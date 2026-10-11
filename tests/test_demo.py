"""The demonstration reads questions and logs. It does not start Spark or a model."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from nlq.demo.server import list_run_files, load_run, question_payload
from nlq.logging_utils import JsonlRunLogger, QuestionRunRecord


def test_questions_include_each_tier() -> None:
    questions = question_payload()
    tiers = {item["tier"] for item in questions}
    assert tiers == {1, 2, 3, 4, 5}
    assert any(item["dataset"] == "nyc_tlc" for item in questions)


def test_run_loader_skips_paths_outside_the_log_directory(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    logger = JsonlRunLogger(runs / "experiment_1.jsonl")
    logger.log_question_run(
        QuestionRunRecord(
            run_id="r1",
            experiment=1,
            system="s1",
            question_id="t1q02",
            scale="sf0.01",
            repeat=1,
            config_hash="abc",
            model_name="m",
            model_version="v",
            prompts_version="1.0.0",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ended_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            payload={"warmup": False, "answer": {"sql": "SELECT 1", "narrative": "one"}, "accuracy": {}, "faithfulness": {}, "latency": {}},
        )
    )
    assert list_run_files(runs)[0]["name"] == "experiment_1.jsonl"
    loaded = load_run(runs, "experiment_1.jsonl")
    assert loaded["rows"][0]["sql"] == "SELECT 1"
    outside = tmp_path / "secret.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    try:
        load_run(runs, "../secret.jsonl")
    except FileNotFoundError:
        return
    raise AssertionError("a path outside runs/ must be rejected")
