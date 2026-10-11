"""Run every gold query on Spark and DuckDB and require the normalised results to match.

Hashes are stored so a later change in generation or SQL is visible. A mismatch
raises; it is not a warning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nlq.benchmark.schema import BenchmarkQuestion, load_questions
from nlq.data.generate_tpch import dataset_directory, scale_label
from nlq.data.nyc_tlc import write_demonstration_fixture
from nlq.engine.duckdb_engine import DuckDBEngine
from nlq.engine.spark_engine import SparkEngine
from nlq.eval.normalise import normalise_rows, result_hash, results_match


class GoldMismatchError(Exception):
    """Spark and DuckDB disagreed, or the stored reference answer does not match the reference scale."""


def validate_questions(
    questions: list[BenchmarkQuestion],
    *,
    data_dir: Path,
    scales: list[float],
    spark: SparkEngine | None = None,
    output_path: Path | None = None,
    nyc_dir: Path | None = None,
) -> dict[str, str]:
    """Cross-check TPC-H gold SQL at each scale. NYC questions are checked on the fixture."""
    hashes: dict[str, str] = {}
    mismatches: list[str] = []
    owns_spark = spark is None
    engine = spark if spark is not None else SparkEngine(master="local[2]")
    try:
        for scale in scales:
            label = scale_label(scale)
            directory = dataset_directory(data_dir, scale)
            duck = DuckDBEngine()
            duck.register_parquet_dataset(directory)
            duck.register_business_views()
            engine.register_parquet_dataset(directory)
            engine.register_business_views()
            for question in questions:
                if question.dataset != "tpch":
                    continue
                _compare(question, duck, engine, f"{question.id}:{label}", hashes, mismatches)
                if label == question.reference_scale:
                    _check_reference(question, duck, mismatches)
            duck.close()
        if nyc_dir is not None:
            fixture = write_demonstration_fixture(nyc_dir)
            duck = DuckDBEngine()
            duck.register_parquet_dataset(fixture)
            engine.register_parquet_dataset(fixture)
            for question in questions:
                if question.dataset != "nyc_tlc":
                    continue
                _compare(question, duck, engine, f"{question.id}:fixture", hashes, mismatches)
                _check_reference(question, duck, mismatches)
            duck.close()
    finally:
        if owns_spark:
            engine.stop()
    if mismatches:
        raise GoldMismatchError("\n".join(mismatches))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashes


def _compare(
    question: BenchmarkQuestion,
    duck: DuckDBEngine,
    spark: SparkEngine,
    key: str,
    hashes: dict[str, str],
    mismatches: list[str],
) -> None:
    try:
        duck_result = duck.execute(question.gold_sql, timeout_s=180)
        spark_result = spark.execute(question.gold_sql, timeout_s=180)
    except Exception as exc:
        mismatches.append(f"{key}: execution failed: {exc}")
        return
    columns = list(dict.fromkeys([*duck_result.columns, *spark_result.columns]))
    if not results_match(
        duck_result.rows,
        spark_result.rows,
        columns,
        tolerance=question.numeric_tolerance,
        ordered=question.ordered,
        column_order=question.column_order,
    ):
        mismatches.append(f"{key}: Spark and DuckDB results differ")
        return
    normalised = normalise_rows(
        duck_result.rows,
        duck_result.columns,
        tolerance=question.numeric_tolerance,
        ordered=question.ordered,
        column_order=question.column_order,
    )
    hashes[key] = result_hash(normalised)


def _check_reference(question: BenchmarkQuestion, duck: DuckDBEngine, mismatches: list[str]) -> None:
    try:
        result = duck.execute(question.gold_sql, timeout_s=180)
    except Exception as exc:
        mismatches.append(f"{question.id}: reference execution failed: {exc}")
        return
    gold_rows = [row for row in question.gold_answer.values if isinstance(row, dict)]
    columns = list(result.columns)
    if not results_match(
        result.rows,
        gold_rows,
        columns,
        tolerance=question.numeric_tolerance,
        ordered=question.ordered,
        column_order=question.column_order,
    ):
        mismatches.append(f"{question.id}: reference gold_answer does not match {question.reference_scale}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-check gold SQL on Spark and DuckDB.")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.01, 0.1])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data") / "gold_hashes.json")
    parser.add_argument("--nyc-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    questions = load_questions()
    validate_questions(
        questions,
        data_dir=args.data_dir,
        scales=args.scales,
        output_path=args.output,
        nyc_dir=args.nyc_dir,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
