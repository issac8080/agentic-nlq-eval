"""Paired tests and figures on synthetic frames. No model and no Spark."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nlq.analysis.reproduce import reproduce
from nlq.eval.run_experiment import _lineitem_rows
from nlq.logging_utils import JsonlRunLogger, QuestionRunRecord

from nlq.analysis.plots import (
    plot_accuracy_by_tier,
    plot_accuracy_faithfulness,
    plot_log_log,
    plot_stacked_latency,
)
from nlq.analysis.stats import (
    aggregate_accuracy,
    cluster_bootstrap_rate,
    holm,
    latency_summary,
    log_log_scaling,
    mcnemar_pairs,
    wilcoxon_pairs,
)


def _accuracy() -> pd.DataFrame:
    rows = []
    for question in range(8):
        for repeat in range(3):
            rows.append(
                {
                    "question_id": f"q{question}",
                    "system": "s1",
                    "tier": question % 5 + 1,
                    "execution_accuracy": True,
                }
            )
            rows.append(
                {
                    "question_id": f"q{question}",
                    "system": "s2",
                    "tier": question % 5 + 1,
                    "execution_accuracy": False,
                }
            )
    return pd.DataFrame(rows)


def test_majority_vote_breaks_ties_as_incorrect() -> None:
    frame = pd.DataFrame(
        [
            {"question_id": "q", "system": "s1", "tier": 1, "execution_accuracy": True},
            {"question_id": "q", "system": "s1", "tier": 1, "execution_accuracy": False},
        ]
    )
    voted = aggregate_accuracy(frame)
    assert int(voted.iloc[0]["execution_accuracy"]) == 0


def test_mcnemar_holm_and_latency(tmp_path: Path) -> None:
    frame = _accuracy()
    pairs = mcnemar_pairs(frame, ["s1", "s2"], seed=1, boots=100)
    assert pairs[0].p_value < 0.05
    assert pairs[0].p_holm <= 1
    assert holm([0.01, 0.04])[0] <= holm([0.01, 0.04])[1] or holm([0.01, 0.04])[0] == 0.02
    latency = pd.DataFrame(
        [
            {
                "question_id": f"q{i}",
                "system": "s1",
                "scale": "sf1",
                "end_to_end_s": i + 1,
                "llm_time_s": 1,
                "exec_time_s": 1,
                "orchestration_s": 1,
            }
            for i in range(8)
        ]
    )
    summary = latency_summary(latency)
    assert set(summary["component"]) == {"end_to_end_s", "llm_time_s", "exec_time_s", "orchestration_s"}
    signed = wilcoxon_pairs(
        pd.concat(
            [
                latency,
                latency.assign(system="s2", end_to_end_s=latency["end_to_end_s"] + 3),
            ]
        ),
        ["s1", "s2"],
    )
    assert signed[0]["p_value"] < 0.05
    fit = log_log_scaling([100, 1000, 10000, 100000], [0.1, 1, 10, 100])
    assert 0.9 < fit["exponent"] < 1.1
    assert fit["exponent_low"] < fit["exponent_high"]
    point, low, high = cluster_bootstrap_rate(["q1", "q1", "q2"], [True, False, True], seed=1, boots=50)
    assert low <= point <= high
    plot_accuracy_by_tier(aggregate_accuracy(frame), tmp_path / "acc.png")
    plot_stacked_latency(latency, tmp_path / "lat.png")
    plot_log_log([100, 1000], [0.1, 1], tmp_path / "log.png")
    plot_accuracy_faithfulness({"correct_faithful": 3, "correct_unfaithful": 1, "incorrect_faithful": 0, "incorrect_unfaithful": 2}, tmp_path / "heat.png")
    for name in ("acc.png", "lat.png", "log.png", "heat.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_lineitem_rows_come_from_the_manifest(tmp_path: Path) -> None:
    directory = tmp_path / "tpch_sf0.01"
    directory.mkdir()
    (directory / "manifest.json").write_text(
        '{"tables": [{"name": "lineitem", "row_count": 60175}]}',
        encoding="utf-8",
    )
    assert _lineitem_rows(tmp_path, "sf0.01") == 60175
    assert _lineitem_rows(tmp_path, "fixture") is None


def test_reproduce_scales_latency_by_lineitem_rows(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    logger = JsonlRunLogger(log_path)
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ended = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    for scale, rows, seconds in (("sf0.01", 100, 1.0), ("sf0.1", 1_000, 10.0), ("sf1", 10_000, 100.0)):
        logger.log_question_run(
            QuestionRunRecord(
                run_id=scale,
                experiment=1,
                system="s1",
                question_id="q",
                scale=scale,
                repeat=1,
                config_hash="abc",
                model_name="m",
                model_version="v",
                prompts_version="1.0.0",
                started_at=started,
                ended_at=ended,
                payload={
                    "warmup": False,
                    "tier": 1,
                    "lineitem_rows": rows,
                    "accuracy": {"execution_accuracy": True, "answer_correct": True},
                    "faithfulness": {"faithful": True},
                    "latency": {
                        "end_to_end_s": seconds,
                        "llm_time_s": 0.1,
                        "exec_time_s": 0.1,
                        "orchestration_s": 0.1,
                    },
                },
            )
        )
    reproduce(log_path, tmp_path / "out")
    assert (tmp_path / "out" / "figures" / "latency_scaling.png").stat().st_size > 0
    assert (tmp_path / "out" / "accuracy.csv").is_file()
