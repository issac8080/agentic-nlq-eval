"""Append-only JSONL logger for node calls and question runs.

One file per run keeps raw model and SQL outputs so later analysis can be
repeated without calling the API again. Records are validated before any byte
is written, so a bad payload cannot leave a partial line.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

LogRecord = Union["NodeCallRecord", "QuestionRunRecord"]


class LogError(Exception):
    """A JSONL file could not be read back as log records."""


def _require_utc(value: datetime) -> datetime:
    """Reject naive timestamps and store UTC so logs from different machines compare."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


class NodeCallRecord(BaseModel):
    """One graph node invocation, including tokens and the raw payload returned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["node_call"] = "node_call"
    run_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    node: str = Field(min_length=1)
    started_at: datetime
    ended_at: datetime
    latency_s: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    status: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("started_at", "ended_at")
    @classmethod
    def _timestamps_utc(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @model_validator(mode="after")
    def _check_interval(self) -> NodeCallRecord:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must be >= started_at")
        return self


class QuestionRunRecord(BaseModel):
    """One question execution. Identity fields support resume and grouped analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["question_run"] = "question_run"
    run_id: str = Field(min_length=1)
    experiment: int = Field(ge=1)
    system: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    scale: str = Field(min_length=1)
    repeat: int = Field(ge=0)
    config_hash: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompts_version: str = Field(min_length=1)
    started_at: datetime
    ended_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("started_at", "ended_at")
    @classmethod
    def _timestamps_utc(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @model_validator(mode="after")
    def _check_interval(self) -> QuestionRunRecord:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must be >= started_at")
        return self


_LOG_RECORD_ADAPTER: TypeAdapter[LogRecord] = TypeAdapter(
    Annotated[NodeCallRecord | QuestionRunRecord, Field(discriminator="record_type")]
)


def parse_log_record(record: LogRecord | Mapping[str, Any]) -> LogRecord:
    """Validate a model or dict into a node-call or question-run record."""
    if isinstance(record, (NodeCallRecord, QuestionRunRecord)):
        return record
    if isinstance(record, Mapping):
        return _LOG_RECORD_ADAPTER.validate_python(dict(record))
    raise TypeError(f"Expected a log record or mapping, got {type(record).__name__}")


class JsonlRunLogger:
    """Append validated records to one JSONL file. A lock guards concurrent appends."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_node_call(self, record: NodeCallRecord) -> None:
        self.write(record)

    def log_question_run(self, record: QuestionRunRecord) -> None:
        self.write(record)

    def write(self, record: LogRecord | Mapping[str, Any]) -> None:
        """Validate and append one JSON line. Failures leave the file unchanged."""
        validated = parse_log_record(record)
        line = validated.model_dump_json()
        if "\n" in line or "\r" in line:
            raise LogError("Serialised log record must be a single line")
        payload = line + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)


def read_records(path: Path | str) -> list[LogRecord]:
    """Read a JSONL run file back into validated records, preserving order."""
    source = Path(path)
    records: list[LogRecord] = []
    text = source.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "":
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LogError(f"{source}:{line_number}: invalid JSON") from exc
        try:
            records.append(parse_log_record(payload))
        except ValidationError as exc:
            raise LogError(f"{source}:{line_number}: invalid log record") from exc
    return records
