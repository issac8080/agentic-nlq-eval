"""Spark and DuckDB must agree on gold SQL. Skipped when the Parquet is absent."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlq.benchmark.schema import load_questions
from nlq.benchmark.validate_gold import validate_questions
from nlq.data.generate_tpch import dataset_directory
from nlq.engine.duckdb_engine import DuckDBEngine
from nlq.engine.spark_engine import SparkEngine
from nlq.llm.prompts import PromptLibrary
from nlq.systems.s1_single import SinglePassSystem
from nlq.systems.s2_retry import RetrySystem
from nlq.systems.s3_agentic.graph import AgenticSystem
from nlq.config import S3Settings
from nlq.data.schema_context import build_schema_context
from nlq.llm.base import LLMClient, LLMResult


class _NationLLM(LLMClient):
    def complete(self, messages: list[dict[str, str]], **params: object) -> LLMResult:
        text = messages[0]["content"]
        if text.startswith("You write one Spark SQL"):
            body = '{"sql": "SELECT COUNT(*) AS nation_count FROM geography"}'
        elif text.startswith("Write the answer"):
            body = (
                '{"answer_values": [{"nation_count": 25}], "entities": [], '
                '"narrative": "There are 25 nations.", "evidence": []}'
            )
        elif text.startswith("Restate"):
            body = '{"plan": "count"}'
        elif text.startswith("List the schema"):
            body = '{"tables": ["geography"], "columns": [], "notes": ""}'
        elif text.startswith("Turn the grounded"):
            body = '{"outline": "count"}'
        elif text.startswith("Describe what"):
            body = '{"analysis": "25"}'
        elif text.startswith("Split the narrative"):
            body = '{"claims": [{"claim_text": "There are 25 nations.", "claim_type": "numeric"}]}'
        else:
            body = "{}"
        return LLMResult(text=body, input_tokens=4, output_tokens=4, latency_s=0.0)


@pytest.fixture(scope="module")
def spark_engine():
    engine = SparkEngine(master="local[2]")
    yield engine
    engine.stop()


def test_gold_matches_on_spark_and_duckdb(project_root: Path, spark_engine: SparkEngine, tmp_path: Path) -> None:
    data_dir = project_root / "data"
    if not dataset_directory(data_dir, 0.01).exists():
        pytest.skip("SF0.01 Parquet has not been generated")
    if not dataset_directory(data_dir, 0.1).exists():
        pytest.skip("SF0.1 Parquet has not been generated")
    questions = load_questions()
    hashes = validate_questions(
        questions,
        data_dir=data_dir,
        scales=[0.01],
        spark=spark_engine,
        output_path=tmp_path / "gold_hashes.json",
        nyc_dir=tmp_path / "nyc",
    )
    assert "t1q02:sf0.01" in hashes
    assert "nyc01:fixture" in hashes
    one = next(item for item in questions if item.id == "t1q02")
    validate_questions([one], data_dir=data_dir, scales=[0.1], spark=spark_engine)


def test_s1_s2_s3_on_sf001(project_root: Path, spark_engine: SparkEngine) -> None:
    data_dir = project_root / "data"
    directory = dataset_directory(data_dir, 0.01)
    if not directory.exists():
        pytest.skip("SF0.01 Parquet has not been generated")
    spark_engine.register_parquet_dataset(directory)
    spark_engine.register_business_views()
    duck = DuckDBEngine()
    duck.register_parquet_dataset(directory)
    duck.register_business_views()
    question = next(item for item in load_questions() if item.id == "t1q02")
    prompts = PromptLibrary(project_root / "prompts")
    try:
        for engine in (duck, spark_engine):
            schema = build_schema_context(engine)
            s1 = SinglePassSystem(_NationLLM(), engine, schema, prompts).answer(question)
            s2 = RetrySystem(_NationLLM(), engine, schema, prompts).answer(question)
            s3 = AgenticSystem(_NationLLM(), engine, schema, prompts, S3Settings()).answer(question)
            for result in (s1, s2, s3):
                assert result.failed is False
                assert result.row_count == 1
                assert result.answer_values[0]["nation_count"] == 25
    finally:
        duck.close()
