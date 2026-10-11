"""JSONL run logger validation and round-trip."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from nlq.logging_utils import (
    JsonlRunLogger,
    NodeCallRecord,
    QuestionRunRecord,
    parse_log_record,
    read_records,
)


def _interval() -> tuple[datetime, datetime]:
    started = datetime(2026, 10, 9, 5, 0, tzinfo=timezone.utc)
    return started, started + timedelta(seconds=1.5)


def test_appends_node_and_question_records(tmp_path: Path) -> None:
    started, ended = _interval()
    path = tmp_path / "runs" / "run-1.jsonl"
    logger = JsonlRunLogger(path)
    logger.log_node_call(
        NodeCallRecord(
            run_id="run-1",
            question_id="q1",
            node="sql_generation",
            started_at=started,
            ended_at=ended,
            latency_s=1.5,
            input_tokens=10,
            output_tokens=4,
            status="ok",
            payload={"text": "SELECT 1"},
        )
    )
    logger.log_question_run(
        QuestionRunRecord(
            run_id="run-1",
            experiment=1,
            system="s1",
            question_id="q1",
            scale="sf0.01",
            repeat=0,
            config_hash="abc123",
            model_name="REPLACE_ME",
            model_version="REPLACE_ME",
            prompts_version="0.1.0",
            started_at=started,
            ended_at=ended,
            payload={"raw_output": "SELECT 1"},
        )
    )

    records = read_records(path)
    assert [record.record_type for record in records] == ["node_call", "question_run"]
    assert isinstance(records[0], NodeCallRecord)
    assert records[0].node == "sql_generation"
    assert records[0].input_tokens == 10
    assert isinstance(records[1], QuestionRunRecord)
    assert records[1].system == "s1"
    assert records[1].payload["raw_output"] == "SELECT 1"
    assert all("\n" not in line for line in path.read_text(encoding="utf-8").splitlines())


def test_invalid_record_does_not_write_a_partial_line(tmp_path: Path) -> None:
    started, ended = _interval()
    path = tmp_path / "run.jsonl"
    logger = JsonlRunLogger(path)
    logger.log_node_call(
        NodeCallRecord(
            run_id="run-1",
            question_id="q1",
            node="execute",
            started_at=started,
            ended_at=ended,
            latency_s=0.1,
            status="ok",
        )
    )
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValidationError):
        logger.write({"record_type": "node_call", "run_id": "run-1"})

    assert path.read_text(encoding="utf-8") == before
    assert len(read_records(path)) == 1


def test_naive_iso_string_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        parse_log_record(
            {
                "record_type": "node_call",
                "run_id": "run-1",
                "question_id": "q1",
                "node": "execute",
                "started_at": "2026-10-09T05:00:00",
                "ended_at": "2026-10-09T05:00:01",
                "latency_s": 1,
                "status": "ok",
            }
        )


def test_naive_timestamp_is_rejected() -> None:
    started = datetime(2026, 10, 9, 5, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        NodeCallRecord(
            run_id="run-1",
            question_id="q1",
            node="execute",
            started_at=started,
            ended_at=started,
            latency_s=0,
            status="ok",
        )
