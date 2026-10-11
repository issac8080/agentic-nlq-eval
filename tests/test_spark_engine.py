"""Spark engine against the same SF0.01 Parquet the DuckDB engine reads."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pyspark.sql.types import IntegerType

from nlq.config import SparkSettings
from nlq.data.generate_tpch import generate_tpch
from nlq.engine.duckdb_engine import DuckDBEngine
from nlq.engine.results import QueryExecutionError, QueryTimeoutError
from nlq.engine.spark_engine import SparkEngine

NATION_SQL = "SELECT n_name FROM nation ORDER BY n_nationkey"


@pytest.fixture(scope="module")
def tpch_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("tpch-spark") / "tpch_sf0.01"
    generate_tpch(0.01, directory)
    return directory


@pytest.fixture(scope="module")
def spark_engine(tpch_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[SparkEngine]:
    warehouse = tmp_path_factory.mktemp("spark-warehouse")
    engine = SparkEngine(SparkSettings(), warehouse_dir=warehouse, master="local[2]")
    engine.register_parquet_dataset(tpch_dir)
    engine.execute("SELECT COUNT(*) AS n FROM region", timeout_s=180)
    try:
        yield engine
    finally:
        engine.stop()


def test_counts_match_the_scale(spark_engine: SparkEngine) -> None:
    lineitem = spark_engine.execute("SELECT COUNT(*) AS n FROM lineitem", timeout_s=180)
    orders = spark_engine.execute("SELECT COUNT(*) AS n FROM orders", timeout_s=180)
    assert lineitem.rows == [{"n": 60175}]
    assert orders.rows == [{"n": 15000}]
    assert lineitem.exec_time_s >= 0


def test_nation_rows_match_duckdb(spark_engine: SparkEngine, tpch_dir: Path) -> None:
    spark_rows = spark_engine.execute(NATION_SQL, timeout_s=180)
    duck = DuckDBEngine()
    try:
        duck.register_parquet_dataset(tpch_dir)
        duck_rows = duck.execute(NATION_SQL, timeout_s=60)
    finally:
        duck.close()
    assert spark_rows.rows == duck_rows.rows
    assert len(spark_rows.rows) == 25


def test_row_cap_and_cache_clear(spark_engine: SparkEngine) -> None:
    capped = spark_engine.execute(
        "SELECT r_name FROM region ORDER BY r_regionkey",
        timeout_s=180,
        apply_row_cap=True,
        row_cap=2,
    )
    assert capped.row_count == 5
    assert len(capped.rows) == 2
    assert capped.truncated is True

    spark_engine.session.catalog.cacheTable("region")
    assert spark_engine.cache_state() == ["region"]
    assert spark_engine.clear_cache() == []


def test_bad_sql_raises(spark_engine: SparkEngine) -> None:
    with pytest.raises(QueryExecutionError):
        spark_engine.execute("SELECT * FROM no_such_table", timeout_s=60)


def test_timeout_cancels_a_sleeping_udf(spark_engine: SparkEngine) -> None:
    def sleep_ms(milliseconds: int) -> int:
        time.sleep(int(milliseconds) / 1000.0)
        return int(milliseconds)

    spark_engine.session.udf.register("nlq_test_sleep_ms", sleep_ms, IntegerType())
    started = time.perf_counter()
    with pytest.raises(QueryTimeoutError):
        spark_engine.execute("SELECT nlq_test_sleep_ms(8000) AS waited", timeout_s=2)
    assert time.perf_counter() - started < 12
