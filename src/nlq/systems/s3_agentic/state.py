"""LangGraph state for S3. Partial dicts from each node overwrite these keys."""

from __future__ import annotations

from typing import Any, TypedDict


class S3State(TypedDict, total=False):
    question: str
    question_id: str
    plan: str
    grounded_schema: str
    outline: str
    candidate_sql: str
    validation_report: dict[str, Any]
    exec_result: dict[str, Any] | None
    exec_error: str | None
    exec_time_s: float
    claims: list[dict[str, Any]]
    claim_check_report: dict[str, Any]
    retry_count: int
    revision_count: int
    timings: dict[str, float]
    token_usage: dict[str, dict[str, int]]
    feedback: list[str]
    narrative_payload: dict[str, Any]
    analysis: str
    llm_calls: list[dict[str, Any]]
    finished: bool
