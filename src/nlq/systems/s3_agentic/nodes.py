"""S3 node functions. Each one records its own latency and token use on the state."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from nlq.data.schema_context import SchemaContext, render_schema_context
from nlq.eval.faithfulness import Claim, verify_claims
from nlq.llm.base import LLMClient
from nlq.llm.prompts import PromptLibrary
from nlq.logging_utils import JsonlRunLogger
from nlq.systems.answer import complete_text, execute_capped, parse_json_object
from nlq.systems.sql_validation import validate_sql


@dataclass
class PipelineContext:
    llm: LLMClient
    engine: Any
    schema: SchemaContext
    prompts: PromptLibrary
    max_sql_retries: int
    max_answer_revisions: int
    enable_sql_validation: bool
    enable_answer_verification: bool
    row_cap: int
    logger: JsonlRunLogger | None
    run_id: str


def question_understanding(state: dict[str, Any], ctx: PipelineContext) -> dict[str, Any]:
    result = _timed_prompt(
        ctx,
        state,
        "question_understanding",
        ctx.prompts.render("question_understanding", QUESTION=state["question"]),
    )
    plan = str(parse_json_object(result["text"]).get("plan", ""))
    return _merge(state, result, plan=plan)


def schema_grounding(state: dict[str, Any], ctx: PipelineContext) -> dict[str, Any]:
    result = _timed_prompt(
        ctx,
        state,
        "schema_grounding",
        ctx.prompts.render(
            "schema_grounding",
            SCHEMA=render_schema_context(ctx.schema),
            PLAN=state.get("plan", ""),
        ),
    )
    grounded = json.dumps(parse_json_object(result["text"]))
    return _merge(state, result, grounded_schema=grounded)


def query_planning(state: dict[str, Any], ctx: PipelineContext) -> dict[str, Any]:
    result = _timed_prompt(
        ctx,
        state,
        "query_planning",
        ctx.prompts.render(
            "query_planning",
            GROUNDING=state.get("grounded_schema", ""),
            PLAN=state.get("plan", ""),
        ),
    )
    outline = str(parse_json_object(result["text"]).get("outline", ""))
    return _merge(state, result, outline=outline)


def sql_generation(state: dict[str, Any], ctx: PipelineContext) -> dict[str, Any]:
    errors = "\n".join(state.get("feedback") or []) or "none"
    prompt = ctx.prompts.render(
        "sql_generation",
        SCHEMA=render_schema_context(ctx.schema),
        QUESTION=state["question"] + "\nOutline:\n" + state.get("outline", ""),
        ERRORS=errors,
    )
    result = _timed_prompt(ctx, state, "sql_generation", prompt)
    sql = str(parse_json_object(result["text"]).get("sql", "")).strip()
    return _merge(state, result, candidate_sql=sql, exec_error=None)


def sql_validation(state: dict[str, Any], ctx: PipelineContext) -> dict[str, Any]:
    started = time.perf_counter()
    report = validate_sql(state.get("candidate_sql", ""), ctx.schema.column_map())
    feedback = list(state.get("feedback") or [])
    retry_count = int(state.get("retry_count") or 0)
    if not report.passed:
        feedback.extend(report.messages)
        retry_count += 1
    timings = dict(state.get("timings") or {})
    timings["sql_validation"] = time.perf_counter() - started
    return {
        "validation_report": report.model_dump(),
        "feedback": feedback,
        "retry_count": retry_count,
        "timings": timings,
    }


def execute(state: dict[str, Any], ctx: PipelineContext) -> dict[str, Any]:
    started = time.perf_counter()
    sql = state.get("candidate_sql", "")
    try:
        result = execute_capped(ctx.engine, sql, ctx.row_cap)
    except Exception as exc:
        feedback = list(state.get("feedback") or [])
        feedback.append(str(exc))
        retry_count = int(state.get("retry_count") or 0)
        # Execution failures share the SQL retry cap with validator failures.
        if ctx.enable_sql_validation:
            retry_count += 1
        return {
            "exec_result": None,
            "exec_error": str(exc),
            "exec_time_s": float(state.get("exec_time_s") or 0) + time.perf_counter() - started,
            "feedback": feedback,
            "retry_count": retry_count,
            "timings": {**dict(state.get("timings") or {}), "execute": time.perf_counter() - started},
        }
    return {
        "exec_result": result.model_dump(),
        "exec_error": None,
        "exec_time_s": float(state.get("exec_time_s") or 0) + result.exec_time_s,
        "timings": {**dict(state.get("timings") or {}), "execute": time.perf_counter() - started},
    }


def result_analysis(state: dict[str, Any], ctx: PipelineContext) -> dict[str, Any]:
    rows = (state.get("exec_result") or {}).get("rows", [])
    result = _timed_prompt(
        ctx,
        state,
        "result_analysis",
        ctx.prompts.render(
            "result_analysis",
            QUESTION=state["question"],
            SQL=state.get("candidate_sql", ""),
            ROWS=json.dumps(rows, default=str),
        ),
    )
    analysis = str(parse_json_object(result["text"]).get("analysis", ""))
    return _merge(state, result, analysis=analysis)


def answer_and_provenance(state: dict[str, Any], ctx: PipelineContext) -> dict[str, Any]:
    notes = []
    report = state.get("claim_check_report") or {}
    if report and not report.get("ok", True):
        notes.append(json.dumps(report.get("verdicts", [])))
    rows = (state.get("exec_result") or {}).get("rows", [])
    row_count = (state.get("exec_result") or {}).get("row_count", 0)
    prompt = ctx.prompts.render(
        "narration",
        QUESTION=state["question"],
        SQL=state.get("candidate_sql", ""),
        ROW_COUNT=str(row_count),
        ROWS=json.dumps(rows, default=str),
        ERRORS="\n".join(notes) if notes else "none",
    )
    result = _timed_prompt(ctx, state, "answer_and_provenance", prompt)
    payload = parse_json_object(result["text"])
    revision_count = int(state.get("revision_count") or 0)
    if notes:
        revision_count += 1
    return _merge(state, result, narrative_payload=payload, revision_count=revision_count)


def answer_verification(state: dict[str, Any], ctx: PipelineContext) -> dict[str, Any]:
    narrative = str((state.get("narrative_payload") or {}).get("narrative", ""))
    result = _timed_prompt(
        ctx,
        state,
        "answer_verification",
        ctx.prompts.render(
            "claim_extraction",
            NARRATIVE=narrative,
            SQL=state.get("candidate_sql", ""),
        ),
    )
    raw_claims = parse_json_object(result["text"]).get("claims") or []
    allowed = {"numeric", "comparative_or_ranking", "entity", "procedural", "explanatory_causal"}
    claims = []
    for item in raw_claims:
        if not isinstance(item, dict) or not item.get("claim_text"):
            continue
        claim_type = item.get("claim_type", "numeric")
        if claim_type not in allowed:
            claim_type = "explanatory_causal"
        claims.append(Claim(claim_text=str(item["claim_text"]), claim_type=claim_type))
    rows = (state.get("exec_result") or {}).get("rows", [])
    verdicts = verify_claims(
        claims,
        rows,
        state.get("candidate_sql", ""),
        tolerance=0.01,
        schema_tables=list(ctx.schema.column_map()),
    )
    bad = [item for item in verdicts if item.label in {"contradicted", "unsupported"}]
    report = {
        "ok": not bad,
        "verdicts": [item.model_dump() for item in verdicts],
    }
    return _merge(state, result, claims=[item.model_dump() for item in claims], claim_check_report=report)


def _timed_prompt(ctx: PipelineContext, state: dict[str, Any], node: str, prompt: str) -> dict[str, Any]:
    started = time.perf_counter()
    result = complete_text(
        ctx.llm,
        prompt,
        logger=ctx.logger,
        run_id=ctx.run_id,
        question_id=str(state.get("question_id", "")),
        node=node,
    )
    timings = dict(state.get("timings") or {})
    timings[node] = time.perf_counter() - started
    usage = dict(state.get("token_usage") or {})
    usage[node] = {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens}
    calls = list(state.get("llm_calls") or [])
    calls.append(
        {
            "node": node,
            "text": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_s": result.latency_s,
        }
    )
    return {"timings": timings, "token_usage": usage, "llm_calls": calls, "text": result.text}


def _merge(state: dict[str, Any], stamped: dict[str, Any], **updates: Any) -> dict[str, Any]:
    merged = {
        "timings": stamped["timings"],
        "token_usage": stamped["token_usage"],
        "llm_calls": stamped["llm_calls"],
    }
    merged.update(updates)
    return merged
