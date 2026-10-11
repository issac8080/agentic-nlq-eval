"""Validator, normaliser, faithfulness, latency, systems, and analysis.

The language model is a script. SQL for the system tests is a fake engine, so
these tests do not start Spark. The gold cross-check lives in test_gold.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nlq.benchmark.schema import load_questions
from nlq.config import PricingSettings, S3Settings
from nlq.data.nyc_tlc import clean_trips
from nlq.data.schema_context import build_schema_context
from nlq.data.views import PROFIT_EXPRESSION, VIEW_SELECTS
from nlq.engine.results import QueryExecutionError, QueryResult
from nlq.eval.faithfulness import (
    Claim,
    calculation_correctness,
    cohens_kappa,
    export_annotation_sample,
    provenance_ok,
    verify_claims,
)
from nlq.eval.latency import decompose
from nlq.eval.normalise import canonical_cell, results_match
from nlq.eval.run_experiment import plan_runs, run_id_for, stratified_subset
from nlq.llm.base import LLMClient, LLMResult
from nlq.llm.cost import BudgetExceededError, BudgetGuard
from nlq.llm.prompts import PromptLibrary
from nlq.systems.s1_single import SinglePassSystem
from nlq.systems.s2_retry import RetrySystem
from nlq.systems.s3_agentic.graph import AgenticSystem
from nlq.systems.sql_validation import validate_sql


class ScriptedLLM(LLMClient):
    """Returns JSON chosen from the prompt text. ``sql_mode`` controls the SQL node."""

    def __init__(self, sql_mode: str = "good") -> None:
        self.sql_mode = sql_mode
        self.prompts: list[str] = []
        self._sql_calls = 0

    def complete(self, messages: list[dict[str, str]], **params: object) -> LLMResult:
        text = messages[0]["content"]
        self.prompts.append(text)
        return LLMResult(text=self._body(text), input_tokens=8, output_tokens=4, latency_s=0.01)

    def _body(self, text: str) -> str:
        if text.startswith("You write one Spark SQL"):
            self._sql_calls += 1
            if self.sql_mode == "bad":
                return '{"sql": "SELECT nope FROM geography"}'
            if self.sql_mode == "retry" and self._sql_calls == 1:
                return '{"sql": "SELECT nope FROM geography"}'
            return '{"sql": "SELECT COUNT(*) AS nation_count FROM geography"}'
        if text.startswith("Restate the business question"):
            return '{"plan": "count nations"}'
        if text.startswith("List the schema tables"):
            return '{"tables": ["geography"], "columns": ["geography.nation_name"], "notes": ""}'
        if text.startswith("Turn the grounded plan"):
            return '{"outline": "count geography"}'
        if text.startswith("Describe what the result"):
            return '{"analysis": "one row"}'
        if text.startswith("Write the answer"):
            return (
                '{"answer_values": [{"nation_count": 25}], "entities": [], '
                '"narrative": "There are 25 nations.", '
                '"evidence": [{"claim_text": "There are 25 nations.", "source": "result_row", "ref": "0"}]}'
            )
        if text.startswith("Split the narrative"):
            return '{"claims": [{"claim_text": "There are 25 nations.", "claim_type": "numeric"}]}'
        raise AssertionError(f"unexpected prompt: {text[:80]}")


class FakeEngine:
    def execute(self, sql: str, **kwargs: object) -> QueryResult:
        if "nope" in sql:
            raise QueryExecutionError(sql, "unknown column nope")
        return QueryResult(
            columns=["nation_count"],
            rows=[{"nation_count": 25}],
            row_count=1,
            truncated=False,
            row_cap=200,
            exec_time_s=0.02,
        )


def _question():
    return next(item for item in load_questions() if item.id == "t1q02")


def test_gold_sql_passes_validator() -> None:
    schema = build_schema_context().column_map()
    for question in load_questions():
        if question.dataset != "tpch":
            schema_for = build_schema_context(dataset="nyc_tlc").column_map()
        else:
            schema_for = schema
        report = validate_sql(question.gold_sql, schema_for)
        assert report.passed, (question.id, report.messages)


def test_validator_rejects_ddl_cross_join_and_unknown_column() -> None:
    schema = build_schema_context().column_map()
    assert not validate_sql("DROP TABLE geography", schema).passed
    assert not validate_sql("SELECT * FROM geography CROSS JOIN customers", schema).passed
    assert not validate_sql("SELECT missing FROM geography", schema).passed
    assert validate_sql("SELECT nation_name FROM geography", schema).passed


def test_normaliser_half_even_and_numeric_strings() -> None:
    assert canonical_cell(1.25, 0.1) == "1.2"
    assert canonical_cell("1.35", 0.1) == "1.4"
    assert results_match(
        [{"nation_count": "25"}],
        [{"nation_count": 25}],
        ["nation_count"],
        tolerance=0,
        ordered=False,
    )


def test_claim_verifier_labels() -> None:
    rows = [{"nation_count": 25, "region_name": "ASIA"}]
    verdicts = verify_claims(
        [
            Claim(claim_text="There are 25 nations.", claim_type="numeric"),
            Claim(claim_text="There are 99 nations.", claim_type="numeric"),
            Claim(claim_text="Demand rose because of weather.", claim_type="explanatory_causal"),
            Claim(claim_text="25 is greater than 10.", claim_type="comparative_or_ranking"),
            Claim(claim_text="The region is ASIA.", claim_type="entity"),
            Claim(claim_text="The query reads geography.", claim_type="procedural"),
        ],
        rows,
        "SELECT nation_name FROM geography",
        tolerance=0.01,
        schema_tables=["geography", "customers"],
    )
    labels = [item.label for item in verdicts]
    assert labels == ["supported", "unsupported", "non_verifiable", "supported", "supported", "supported"]
    assert verdicts[1].verification_query is None
    assert calculation_correctness(["2 + 2 = 4", "2 + 2 = 5"], 0.01) == 0.5
    assert provenance_ok(
        reported_sql="SELECT 1",
        executed_sql="SELECT 1",
        reported_tables=["geography"],
        executed_tables=["geography"],
        reported_filters=[],
        executed_filters=[],
        reported_row_count=1,
        executed_row_count=1,
    )


def test_kappa_and_annotation_sample(tmp_path: Path) -> None:
    assert cohens_kappa(["supported", "supported"], ["supported", "supported"]) == 1.0
    path = tmp_path / "sample.csv"
    rows = [{"tier": "1", "question_id": f"q{i}", "claim_text": "c", "claim_type": "numeric", "narrative": "n", "system": "s3"} for i in range(10)]
    count = export_annotation_sample(rows, str(path), fraction=0.22, seed=1)
    text = path.read_text(encoding="utf-8")
    assert count >= 1
    assert "system" not in text.splitlines()[0]
    assert "s3" not in text


def test_latency_decomposition_clamps_at_zero() -> None:
    assert decompose(10, 3, 4) == pytest.approx(3)
    assert decompose(1, 2, 0) == 0


def test_profit_is_defined_once() -> None:
    assert "l_extendedprice" in PROFIT_EXPRESSION
    assert "ps_supplycost" in PROFIT_EXPRESSION
    assert "profit" in VIEW_SELECTS["order_lines"]


def test_nyc_cleaning_drops_bad_rows() -> None:
    raw = pd.DataFrame(
        [
            {
                "tpep_pickup_datetime": "2023-01-02 10:00:00",
                "tpep_dropoff_datetime": "2023-01-02 10:10:00",
                "passenger_count": 1,
                "trip_distance": 2.0,
                "fare_amount": 10.0,
                "tip_amount": 1.0,
                "total_amount": 12.0,
                "payment_type": 1,
                "PULocationID": 1,
                "DOLocationID": 2,
            },
            {
                "tpep_pickup_datetime": "2023-03-02 10:00:00",
                "tpep_dropoff_datetime": "2023-03-02 10:10:00",
                "passenger_count": 1,
                "trip_distance": 2.0,
                "fare_amount": 10.0,
                "tip_amount": 1.0,
                "total_amount": 12.0,
                "payment_type": 1,
                "PULocationID": 1,
                "DOLocationID": 2,
            },
            {
                "tpep_pickup_datetime": "2023-01-03 10:00:00",
                "tpep_dropoff_datetime": "2023-01-03 10:10:00",
                "passenger_count": 1,
                "trip_distance": 2.0,
                "fare_amount": -1.0,
                "tip_amount": 0.0,
                "total_amount": 1.0,
                "payment_type": 1,
                "PULocationID": 1,
                "DOLocationID": 2,
            },
        ]
    )
    zones = pd.DataFrame([{"zone_id": 1}, {"zone_id": 2}])
    cleaned = clean_trips(raw, zones)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["payment_type"] == "credit_card"


def test_budget_guard_stops_before_the_next_call() -> None:
    pricing = PricingSettings(input_usd_per_million=1_000_000, output_usd_per_million=0)
    guard = BudgetGuard(1.0, pricing)
    guard.before_call()
    guard.charge(1, 0)
    with pytest.raises(BudgetExceededError):
        guard.before_call()


def test_experiment_plan_is_seeded_and_resume_ids_are_stable() -> None:
    questions = load_questions()
    subset = stratified_subset(questions, 9, 42)
    assert len(subset) == 25
    planned = plan_runs(subset, ["s1", "s2"], ["sf0.01"], repeats=2, seed=42)
    assert planned[0]["warmup"] is True
    assert planned[0]["repeat"] == 0
    measured = [item for item in planned if not item["warmup"]]
    assert len(measured) == 25 * 2 * 2
    first = run_id_for(
        experiment=1,
        system="s1",
        question_id="t1q02",
        scale="sf0.01",
        repeat=1,
        warmup=False,
        config_hash="abc",
        model="m",
        prompts_version="1.0.0",
    )
    again = run_id_for(
        experiment=1,
        system="s1",
        question_id="t1q02",
        scale="sf0.01",
        repeat=1,
        warmup=False,
        config_hash="abc",
        model="m",
        prompts_version="1.0.0",
    )
    assert first == again


def test_s1_s2_s3_with_scripted_model(project_root: Path) -> None:
    prompts = PromptLibrary(project_root / "prompts")
    schema = build_schema_context()
    question = _question()
    engine = FakeEngine()
    s1 = SinglePassSystem(ScriptedLLM(), engine, schema, prompts).answer(question)
    assert s1.failed is False
    assert s1.answer_values == [{"nation_count": 25}]
    assert s1.n_llm_calls == 2
    assert "geography" in s1.tables

    s2_llm = ScriptedLLM(sql_mode="retry")
    s2 = RetrySystem(s2_llm, engine, schema, prompts, max_sql_retries=3).answer(question)
    assert s2.failed is False
    assert s2.retries == 1
    assert s2_llm._sql_calls == 2

    s3 = AgenticSystem(ScriptedLLM(), engine, schema, prompts, S3Settings()).answer(question)
    assert s3.failed is False
    assert s3.n_llm_calls == 7
    assert s3.retries == 0


def test_s3_ablations_and_retry_cap(project_root: Path) -> None:
    prompts = PromptLibrary(project_root / "prompts")
    schema = build_schema_context()
    question = _question()
    engine = FakeEngine()
    settings = S3Settings()

    no_validation = ScriptedLLM(sql_mode="bad")
    s3v = AgenticSystem(
        no_validation,
        engine,
        schema,
        prompts,
        settings.model_copy(update={"enable_sql_validation": False}),
        name="s3v",
    ).answer(question)
    assert s3v.failed is True
    assert no_validation._sql_calls == 1

    no_verify = ScriptedLLM()
    s3a = AgenticSystem(
        no_verify,
        engine,
        schema,
        prompts,
        settings.model_copy(update={"enable_answer_verification": False}),
        name="s3a",
    ).answer(question)
    assert s3a.failed is False
    assert not any(text.startswith("Split the narrative") for text in no_verify.prompts)

    exhausted = ScriptedLLM(sql_mode="bad")
    failed = AgenticSystem(exhausted, engine, schema, prompts, settings).answer(question)
    assert failed.failed is True
    assert exhausted._sql_calls == 4
    assert failed.retries == 3
