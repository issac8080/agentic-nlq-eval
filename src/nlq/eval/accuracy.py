"""Execution accuracy and answer correctness. Both are deterministic; there is no model judge.

Execution accuracy compares the predicted SQL's result with the gold SQL's result
after normalisation. Answer correctness compares the structured answer object
with the gold answer. At a scale other than the question's reference scale, the
gold values are the gold SQL result at that scale.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from nlq.benchmark.schema import BenchmarkQuestion
from nlq.eval.normalise import normalise_rows, results_match

FAILURE_LABELS = (
    "wrong_join",
    "wrong_filter",
    "wrong_aggregation",
    "schema_linking",
    "time_period",
    "misinterpretation",
)


class AccuracyRecord(BaseModel):
    """One question. ``failure_label`` is left empty for a person to fill in."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    system: str
    execution_accuracy: bool
    answer_correct: bool
    sql_valid: bool
    failure_label: str | None = None

    def with_manual_label(self, label: str) -> AccuracyRecord:
        if label not in FAILURE_LABELS:
            raise ValueError(f"Unknown failure label: {label}")
        return self.model_copy(update={"failure_label": label})


class SqlRunner(Protocol):
    def execute(self, sql: str, *, timeout_s: float | None = None, apply_row_cap: bool = False, row_cap: int | None = None): ...


def execution_accuracy(
    predicted_rows: list[dict[str, Any]],
    predicted_columns: list[str],
    gold_rows: list[dict[str, Any]],
    gold_columns: list[str],
    question: BenchmarkQuestion,
) -> bool:
    columns = gold_columns or predicted_columns
    return results_match(
        predicted_rows,
        gold_rows,
        list(columns),
        tolerance=question.numeric_tolerance,
        ordered=question.ordered,
        column_order=question.column_order,
    )


def answer_correct(
    answer_values: list[Any],
    entities: list[str],
    question: BenchmarkQuestion,
    *,
    expected_values: list[Any] | None = None,
    check_entities: bool = True,
) -> bool:
    """Match values with the same normaliser used for result sets, and entities as a set.

    ``expected_values`` overrides the YAML gold when the scale is not the reference
    scale. Entity names in the YAML are the reference-scale audit, so they are
    checked only when that override is not in use.
    """
    gold_values = question.gold_answer.values if expected_values is None else expected_values
    if not isinstance(answer_values, list) or not isinstance(gold_values, list):
        return False
    if len(answer_values) != len(gold_values):
        return False
    if answer_values and isinstance(answer_values[0], dict) and isinstance(gold_values[0], dict):
        columns = sorted({key for row in gold_values for key in row})
        if not results_match(
            answer_values,
            gold_values,
            columns,
            tolerance=question.numeric_tolerance,
            ordered=question.ordered,
            column_order=question.column_order,
        ):
            return False
    else:
        left = normalise_rows(
            [{"value": value} for value in answer_values],
            ["value"],
            tolerance=question.numeric_tolerance,
            ordered=question.ordered,
        )
        right = normalise_rows(
            [{"value": value} for value in gold_values],
            ["value"],
            tolerance=question.numeric_tolerance,
            ordered=question.ordered,
        )
        if left != right:
            return False
    if not check_entities:
        return True
    return set(entities) == set(question.gold_answer.entities)


def score_question(
    *,
    question: BenchmarkQuestion,
    system: str,
    predicted_sql: str,
    runner: SqlRunner,
    answer_values: list[Any],
    entities: list[str],
    sql_failed: bool,
    use_executed_gold: bool = False,
) -> AccuracyRecord:
    """Run gold SQL and the predicted SQL. A failed predicted SQL is invalid and inaccurate."""
    if sql_failed or not predicted_sql.strip():
        return AccuracyRecord(
            question_id=question.id,
            system=system,
            execution_accuracy=False,
            answer_correct=answer_correct(answer_values, entities, question) if not use_executed_gold else False,
            sql_valid=False,
        )
    try:
        predicted = runner.execute(predicted_sql, timeout_s=120)
        gold = runner.execute(question.gold_sql, timeout_s=120)
    except Exception:
        return AccuracyRecord(
            question_id=question.id,
            system=system,
            execution_accuracy=False,
            answer_correct=False,
            sql_valid=False,
        )
    return AccuracyRecord(
        question_id=question.id,
        system=system,
        execution_accuracy=execution_accuracy(
            predicted.rows, predicted.columns, gold.rows, gold.columns, question
        ),
        answer_correct=answer_correct(
            answer_values,
            entities,
            question,
            expected_values=gold.rows if use_executed_gold else None,
            check_entities=not use_executed_gold,
        ),
        sql_valid=True,
    )
