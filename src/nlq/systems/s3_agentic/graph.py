"""S3 graph. Ablations are flags: S3-V skips the SQL correction loop, S3-A skips answer verification."""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from nlq.benchmark.schema import BenchmarkQuestion
from nlq.config import S3Settings
from nlq.data.schema_context import SchemaContext
from nlq.engine.results import QueryResult
from nlq.llm.base import LLMClient, LLMResult
from nlq.llm.prompts import PromptLibrary
from nlq.logging_utils import JsonlRunLogger
from nlq.systems.answer import SystemResult, assemble_result
from nlq.systems.s3_agentic import nodes
from nlq.systems.s3_agentic.nodes import PipelineContext
from nlq.systems.s3_agentic.state import S3State
from nlq.systems.sql_validation import SqlValidation


class AgenticSystem:
    """Full pipeline. ``name`` is s3, s3v, or s3a depending on which checks are on."""

    def __init__(
        self,
        llm: LLMClient,
        engine: Any,
        schema: SchemaContext,
        prompts: PromptLibrary,
        settings: S3Settings,
        *,
        name: str | None = None,
        row_cap: int = 200,
        logger: JsonlRunLogger | None = None,
        run_id: str = "s3",
    ) -> None:
        self.name = name or system_name(settings)
        self.row_cap = row_cap
        self.schema = schema
        self._ctx = PipelineContext(
            llm=llm,
            engine=engine,
            schema=schema,
            prompts=prompts,
            max_sql_retries=settings.max_sql_retries,
            max_answer_revisions=settings.max_answer_revisions,
            enable_sql_validation=settings.enable_sql_validation,
            enable_answer_verification=settings.enable_answer_verification,
            row_cap=row_cap,
            logger=logger,
            run_id=run_id,
        )
        self._graph = build_graph(self._ctx)

    def answer(self, question: BenchmarkQuestion) -> SystemResult:
        started = time.perf_counter()
        state: S3State = {
            "question": question.question,
            "question_id": question.id,
            "retry_count": 0,
            "revision_count": 0,
            "feedback": [],
            "timings": {},
            "token_usage": {},
            "llm_calls": [],
            "exec_time_s": 0.0,
        }
        final = self._graph.invoke(state, config={"recursion_limit": 40})
        return _result_from_state(
            final,
            question,
            self.schema,
            started,
            max_sql_retries=self._ctx.max_sql_retries,
        )


def system_name(settings: S3Settings) -> str:
    if not settings.enable_sql_validation:
        return "s3v"
    if not settings.enable_answer_verification:
        return "s3a"
    return "s3"


def build_graph(ctx: PipelineContext):
    """Compile the pipeline. Flags change edges, not the node implementations."""
    graph = StateGraph(S3State)
    graph.add_node("question_understanding", lambda state: nodes.question_understanding(state, ctx))
    graph.add_node("schema_grounding", lambda state: nodes.schema_grounding(state, ctx))
    graph.add_node("query_planning", lambda state: nodes.query_planning(state, ctx))
    graph.add_node("sql_generation", lambda state: nodes.sql_generation(state, ctx))
    graph.add_node("sql_validation", lambda state: nodes.sql_validation(state, ctx))
    graph.add_node("execute", lambda state: nodes.execute(state, ctx))
    graph.add_node("result_analysis", lambda state: nodes.result_analysis(state, ctx))
    graph.add_node("answer_and_provenance", lambda state: nodes.answer_and_provenance(state, ctx))
    graph.add_node("answer_verification", lambda state: nodes.answer_verification(state, ctx))

    graph.add_edge(START, "question_understanding")
    graph.add_edge("question_understanding", "schema_grounding")
    graph.add_edge("schema_grounding", "query_planning")
    graph.add_edge("query_planning", "sql_generation")
    graph.add_conditional_edges(
        "sql_generation",
        lambda state: "execute" if not ctx.enable_sql_validation else "sql_validation",
        {"execute": "execute", "sql_validation": "sql_validation"},
    )
    graph.add_conditional_edges(
        "sql_validation",
        lambda state: _after_validation(state, ctx),
        {"execute": "execute", "sql_generation": "sql_generation", "end": END},
    )
    graph.add_conditional_edges(
        "execute",
        lambda state: _after_execute(state, ctx),
        {"result_analysis": "result_analysis", "sql_generation": "sql_generation", "end": END},
    )
    graph.add_edge("result_analysis", "answer_and_provenance")
    graph.add_conditional_edges(
        "answer_and_provenance",
        lambda state: "end" if not ctx.enable_answer_verification else "answer_verification",
        {"end": END, "answer_verification": "answer_verification"},
    )
    graph.add_conditional_edges(
        "answer_verification",
        lambda state: _after_verification(state, ctx),
        {"answer_and_provenance": "answer_and_provenance", "end": END},
    )
    return graph.compile()


def _after_validation(state: dict[str, Any], ctx: PipelineContext) -> str:
    report = state.get("validation_report") or {}
    if report.get("passed"):
        return "execute"
    if int(state.get("retry_count") or 0) > ctx.max_sql_retries:
        return "end"
    return "sql_generation"


def _after_execute(state: dict[str, Any], ctx: PipelineContext) -> str:
    if not state.get("exec_error"):
        return "result_analysis"
    if not ctx.enable_sql_validation:
        return "end"
    if int(state.get("retry_count") or 0) > ctx.max_sql_retries:
        return "end"
    return "sql_generation"


def _after_verification(state: dict[str, Any], ctx: PipelineContext) -> str:
    report = state.get("claim_check_report") or {}
    if report.get("ok", False):
        return "end"
    if int(state.get("revision_count") or 0) >= ctx.max_answer_revisions:
        return "end"
    return "answer_and_provenance"


def _result_from_state(
    state: dict[str, Any],
    question: BenchmarkQuestion,
    schema: SchemaContext,
    started: float,
    *,
    max_sql_retries: int,
) -> SystemResult:
    report = state.get("validation_report") or {"passed": not bool(state.get("exec_error")), "messages": []}
    validation = SqlValidation(passed=bool(report.get("passed")), messages=list(report.get("messages") or []))
    raw_exec = state.get("exec_result")
    query = QueryResult.model_validate(raw_exec) if raw_exec else None
    calls = [
        LLMResult(
            text=item["text"],
            input_tokens=item["input_tokens"],
            output_tokens=item["output_tokens"],
            latency_s=item["latency_s"],
        )
        for item in state.get("llm_calls") or []
    ]
    failed = query is None
    extra = [state["exec_error"]] if state.get("exec_error") else None
    result = assemble_result(
        question=question,
        schema=schema,
        sql=str(state.get("candidate_sql") or ""),
        validation=validation,
        query=query,
        narrative_payload=state.get("narrative_payload"),
        llm_results=calls,
        exec_time_s=float(state.get("exec_time_s") or 0),
        started=started,
        retries=min(int(state.get("retry_count") or 0), max_sql_retries),
        failed=failed,
        extra_messages=extra,
    )
    result.raw_outputs["timings"] = state.get("timings") or {}
    result.raw_outputs["token_usage"] = state.get("token_usage") or {}
    result.raw_outputs["claims"] = state.get("claims") or []
    return result
