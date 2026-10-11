"""Answer object every system returns. Timing fields sit beside the narrative so latency can be decomposed later."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nlq.benchmark.schema import BenchmarkQuestion
from nlq.data.schema_context import SchemaContext
from nlq.engine.results import QueryExecutionError, QueryResult, QueryTimeoutError
from nlq.llm.base import LLMClient, LLMResult
from nlq.llm.prompts import PromptLibrary
from nlq.logging_utils import JsonlRunLogger, NodeCallRecord
from nlq.systems.sql_validation import SqlValidation, extract_filters, extract_tables, validate_sql


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_text: str
    source: Literal["result_row", "sql"]
    ref: str


class ValidationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    messages: list[str] = Field(default_factory=list)


class SystemResult(BaseModel):
    """Common answer. Provenance fields are taken from the executed SQL, not from the narrative."""

    model_config = ConfigDict(extra="forbid")

    answer_values: list[Any] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    sql: str = ""
    tables: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    narrative: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    validation: ValidationInfo = Field(default_factory=lambda: ValidationInfo(passed=False))
    row_count: int = 0
    truncated: bool = False
    failed: bool = False
    llm_time_s: float = 0
    exec_time_s: float = 0
    end_to_end_s: float = 0
    orchestration_s: float = 0
    n_llm_calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    raw_outputs: dict[str, Any] = Field(default_factory=dict)


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model response did not contain a JSON object")
    loaded = json.loads(text[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("Model JSON was not an object")
    return loaded


def complete_text(
    llm: LLMClient,
    prompt: str,
    *,
    logger: JsonlRunLogger | None,
    run_id: str,
    question_id: str,
    node: str,
) -> LLMResult:
    started = datetime.now(timezone.utc)
    result = llm.complete([{"role": "user", "content": prompt}])
    ended = datetime.now(timezone.utc)
    if logger is not None:
        logger.log_node_call(
            NodeCallRecord(
                run_id=run_id,
                question_id=question_id,
                node=node,
                started_at=started,
                ended_at=ended,
                latency_s=result.latency_s,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                status="ok",
                payload={"text": result.text},
            )
        )
    return result


def generate_sql(
    llm: LLMClient,
    prompts: PromptLibrary,
    schema_text: str,
    question: str,
    errors: list[str],
    *,
    logger: JsonlRunLogger | None,
    run_id: str,
    question_id: str,
) -> tuple[str, LLMResult]:
    prompt = prompts.render(
        "sql_generation",
        SCHEMA=schema_text,
        QUESTION=question,
        ERRORS="\n".join(errors) if errors else "none",
    )
    result = complete_text(
        llm, prompt, logger=logger, run_id=run_id, question_id=question_id, node="sql_generation"
    )
    payload = parse_json_object(result.text)
    sql = str(payload.get("sql", "")).strip()
    if not sql:
        raise ValueError("Model did not return SQL")
    return sql, result


def narrate(
    llm: LLMClient,
    prompts: PromptLibrary,
    question: str,
    sql: str,
    query: QueryResult,
    errors: list[str],
    *,
    logger: JsonlRunLogger | None,
    run_id: str,
    question_id: str,
) -> tuple[dict[str, Any], LLMResult]:
    prompt = prompts.render(
        "narration",
        QUESTION=question,
        SQL=sql,
        ROW_COUNT=str(query.row_count),
        ROWS=json.dumps(query.rows, default=str),
        ERRORS="\n".join(errors) if errors else "none",
    )
    result = complete_text(
        llm, prompt, logger=logger, run_id=run_id, question_id=question_id, node="narration"
    )
    return parse_json_object(result.text), result


def execute_capped(engine: Any, sql: str, row_cap: int) -> QueryResult:
    """Run SQL for the narration prompt. The cap is the engine's logged row limit."""
    return engine.execute(sql, apply_row_cap=True, row_cap=row_cap)


def evidence_from_payload(payload: dict[str, Any]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for raw in payload.get("evidence") or []:
        if not isinstance(raw, dict):
            continue
        source = raw.get("source")
        if source not in {"result_row", "sql"}:
            continue
        text = str(raw.get("claim_text", "")).strip()
        if not text:
            continue
        items.append(EvidenceItem(claim_text=text, source=source, ref=str(raw.get("ref", ""))))
    return items


def assemble_result(
    *,
    question: BenchmarkQuestion,
    schema: SchemaContext,
    sql: str,
    validation: SqlValidation,
    query: QueryResult | None,
    narrative_payload: dict[str, Any] | None,
    llm_results: list[LLMResult],
    exec_time_s: float,
    started: float,
    retries: int,
    failed: bool,
    extra_messages: list[str] | None = None,
) -> SystemResult:
    tables = extract_tables(sql, set(schema.column_map())) if sql else []
    filters = extract_filters(sql) if sql else []
    payload = narrative_payload or {}
    messages = list(validation.messages)
    if extra_messages:
        messages.extend(extra_messages)
    llm_time = sum(item.latency_s for item in llm_results)
    end_to_end = time.perf_counter() - started
    return SystemResult(
        answer_values=list(payload.get("answer_values") or []),
        entities=[str(item) for item in payload.get("entities") or []],
        sql=sql,
        tables=tables,
        filters=filters,
        narrative=str(payload.get("narrative") or ""),
        evidence=evidence_from_payload(payload),
        validation=ValidationInfo(passed=validation.passed and not failed, messages=messages),
        row_count=0 if query is None else query.row_count,
        truncated=False if query is None else query.truncated,
        failed=failed,
        llm_time_s=llm_time,
        exec_time_s=exec_time_s,
        end_to_end_s=end_to_end,
        orchestration_s=max(0.0, end_to_end - llm_time - exec_time_s),
        n_llm_calls=len(llm_results),
        retries=retries,
        input_tokens=sum(item.input_tokens for item in llm_results),
        output_tokens=sum(item.output_tokens for item in llm_results),
        raw_outputs={"llm": [item.text for item in llm_results]},
    )


def validation_for(sql: str, schema: SchemaContext) -> SqlValidation:
    return validate_sql(sql, schema.column_map())


def run_sql(engine: Any, sql: str, row_cap: int) -> tuple[QueryResult | None, str | None, float]:
    started = time.perf_counter()
    try:
        result = execute_capped(engine, sql, row_cap)
    except (QueryExecutionError, QueryTimeoutError) as exc:
        return None, str(exc), time.perf_counter() - started
    return result, None, time.perf_counter() - started
