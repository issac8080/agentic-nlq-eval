"""Pydantic models for benchmark questions.

Gold answers stored in YAML are the reference-scale result. Execution accuracy
at another scale recomputes gold by running ``gold_sql`` on that scale, because
aggregates change as the data grows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

QUESTIONS_DIR = Path(__file__).resolve().parent / "questions"


class GoldAnswer(BaseModel):
    """Structured expectation: result rows and the entities a narrative should name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    values: list[Any]
    entities: list[str] = Field(default_factory=list)


class BenchmarkQuestion(BaseModel):
    """One natural-language question and the Spark SQL that answers it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    tier: int = Field(ge=1, le=5)
    dataset: Literal["tpch", "nyc_tlc"]
    reference_scale: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_sql: str = Field(min_length=1)
    ordered: bool
    numeric_tolerance: float = Field(ge=0)
    rounding: Literal["half_even"] = "half_even"
    column_order: Literal["name", "position"] = "name"
    gold_answer: GoldAnswer

    @field_validator("gold_sql")
    @classmethod
    def _strip_sql(cls, value: str) -> str:
        return " ".join(value.split())


def load_questions(directory: Path | None = None) -> list[BenchmarkQuestion]:
    """Load every YAML file in the questions directory. Ids must be unique."""
    root = Path(directory) if directory is not None else QUESTIONS_DIR
    questions: list[BenchmarkQuestion] = []
    for path in sorted(root.glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"{path} must contain a list of questions")
        questions.extend(BenchmarkQuestion.model_validate(item) for item in loaded)
    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Question ids must be unique")
    return questions


def questions_for_dataset(questions: list[BenchmarkQuestion], dataset: str) -> list[BenchmarkQuestion]:
    return [question for question in questions if question.dataset == dataset]
