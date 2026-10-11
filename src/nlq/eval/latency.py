"""Latency decomposition. Orchestration is what is left after model time and SQL time."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LatencyRecord(BaseModel):
    """One measured run. A warm-up run is stored and then dropped by the analysis."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    question_id: str
    system: str
    scale: str
    repeat: int = Field(ge=0)
    end_to_end_s: float = Field(ge=0)
    llm_time_s: float = Field(ge=0)
    exec_time_s: float = Field(ge=0)
    orchestration_s: float = Field(ge=0)
    n_llm_calls: int = Field(ge=0)
    retries: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    failed: bool
    warmup: bool = False


def decompose(end_to_end_s: float, llm_time_s: float, exec_time_s: float) -> float:
    """Orchestration is the remainder. Clock noise cannot make it negative."""
    return max(0.0, end_to_end_s - llm_time_s - exec_time_s)


def from_system_result(run_id: str, question_id: str, system: str, scale: str, repeat: int, result: Any, *, warmup: bool = False) -> LatencyRecord:
    orchestration = decompose(result.end_to_end_s, result.llm_time_s, result.exec_time_s)
    return LatencyRecord(
        run_id=run_id,
        question_id=question_id,
        system=system,
        scale=scale,
        repeat=repeat,
        end_to_end_s=result.end_to_end_s,
        llm_time_s=result.llm_time_s,
        exec_time_s=result.exec_time_s,
        orchestration_s=orchestration,
        n_llm_calls=result.n_llm_calls,
        retries=result.retries,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        failed=result.failed,
        warmup=warmup,
    )

