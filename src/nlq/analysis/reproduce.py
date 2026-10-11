"""Turn experiment JSONL into the tables and figures in results/.

Warm-up rows are dropped. Accuracy and latency come from the fields the runner
stored, so this step does not call a model or start Spark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nlq.analysis.plots import (
    plot_accuracy_by_tier,
    plot_accuracy_faithfulness,
    plot_log_log,
    plot_stacked_latency,
    write_result_tables,
)
from nlq.analysis.stats import aggregate_accuracy, latency_summary, log_log_scaling, mcnemar_pairs
from nlq.eval.faithfulness import contingency
from nlq.logging_utils import read_records


def load_measured(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Accuracy, latency, and faithfulness frames. Warm-up repeats are omitted."""
    accuracy_rows = []
    latency_rows = []
    faith_rows = []
    for record in read_records(path):
        if record.record_type != "question_run" or record.payload.get("warmup"):
            continue
        accuracy = record.payload.get("accuracy") or {}
        latency = record.payload.get("latency") or {}
        faith = record.payload.get("faithfulness") or {}
        accuracy_rows.append(
            {
                "question_id": record.question_id,
                "system": record.system,
                "tier": record.payload.get("tier"),
                "scale": record.scale,
                "repeat": record.repeat,
                "execution_accuracy": bool(accuracy.get("execution_accuracy")),
                "answer_correct": bool(accuracy.get("answer_correct")),
            }
        )
        latency_rows.append(
            {
                "question_id": record.question_id,
                "system": record.system,
                "scale": record.scale,
                "end_to_end_s": latency.get("end_to_end_s", 0),
                "llm_time_s": latency.get("llm_time_s", 0),
                "exec_time_s": latency.get("exec_time_s", 0),
                "orchestration_s": latency.get("orchestration_s", 0),
                "lineitem_rows": record.payload.get("lineitem_rows"),
            }
        )
        faith_rows.append(
            {
                "question_id": record.question_id,
                "system": record.system,
                "correct": bool(accuracy.get("execution_accuracy")),
                "faithful": bool(faith.get("faithful")),
            }
        )
    return pd.DataFrame(accuracy_rows), pd.DataFrame(latency_rows), pd.DataFrame(faith_rows)


def reproduce(run_path: Path, output_dir: Path) -> list[Path]:
    """Write tidy CSVs and the four figures. Returns the files created."""
    accuracy, latency, faith = load_measured(run_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    written: list[Path] = []
    if not accuracy.empty:
        aggregated = aggregate_accuracy(accuracy)
        systems = sorted(accuracy["system"].unique())
        pairs = mcnemar_pairs(accuracy, systems, boots=200)
        table = contingency(faith["correct"].tolist(), faith["faithful"].tolist()) if not faith.empty else {}
        written.extend(
            write_result_tables(
                output_dir,
                accuracy=aggregated,
                latency=latency_summary(latency) if not latency.empty else pd.DataFrame(),
                mcnemar=pd.DataFrame([item.model_dump() for item in pairs]),
            )
        )
        written.append(plot_accuracy_by_tier(aggregated, figures / "accuracy_by_tier.png"))
        if table:
            written.append(plot_accuracy_faithfulness(table, figures / "accuracy_faithfulness.png"))
    if not latency.empty:
        written.append(plot_stacked_latency(latency, figures / "latency_components.png"))
        latency = latency.copy()
        latency["lineitem_rows"] = pd.to_numeric(latency["lineitem_rows"], errors="coerce")
        grouped = latency.groupby("scale", as_index=False)[["end_to_end_s", "lineitem_rows"]].median()
        usable = grouped.dropna(subset=["lineitem_rows"])
        usable = usable.loc[usable["lineitem_rows"] > 0]
        if len(usable) >= 3:
            counts = usable["lineitem_rows"].tolist()
            times = usable["end_to_end_s"].tolist()
            fit = log_log_scaling(counts, times)
            (output_dir / "scaling.json").write_text(json.dumps(fit, indent=2) + "\n", encoding="utf-8")
            written.append(plot_log_log(counts, times, figures / "latency_scaling.png"))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce tables and figures from an experiment JSONL file.")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    paths = reproduce(args.runs, args.out)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
