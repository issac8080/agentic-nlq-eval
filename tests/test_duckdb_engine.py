"""DuckDB engine against generated TPC-H Parquet. Used only as a gold-SQL cross-check."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from nlq.data.generate_tpch import generate_tpch
from nlq.engine.duckdb_engine import DuckDBEngine
from nlq.engine.results import QueryExecutionError, QueryTimeoutError


@pytest.fixture(scope="module")
def tpch_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("tpch-duck") / "tpch_sf0.01"
    generate_tpch(0.01, directory)
    return directory


@pytest.fixture(scope="module")
def duck_engine(tpch_dir: Path) -> Iterator[DuckDBEngine]:
    engine = DuckDBEngine()
    engine.register_parquet_dataset(tpch_dir)
    try:
        yield engine
    finally:
        engine.close()


def test_lineitem_and_orders_counts(duck_engine: DuckDBEngine) -> None:
    lineitem = duck_engine.execute("SELECT COUNT(*) AS n FROM lineitem", timeout_s=60)
    orders = duck_engine.execute("SELECT COUNT(*) AS n FROM orders", timeout_s=60)
    assert lineitem.rows == [{"n": 60175}]
    assert lineitem.row_count == 1
    assert lineitem.truncated is False
    assert lineitem.exec_time_s >= 0
    assert orders.rows == [{"n": 15000}]


def test_row_cap_reports_the_full_count(duck_engine: DuckDBEngine) -> None:
    result = duck_engine.execute(
        "SELECT r_name FROM region ORDER BY r_regionkey",
        timeout_s=60,
        apply_row_cap=True,
        row_cap=2,
    )
    assert result.row_count == 5
    assert len(result.rows) == 2
    assert result.truncated is True
    assert result.row_cap == 2
    assert result.rows[0]["r_name"] == "AFRICA"


def test_bad_sql_raises(duck_engine: DuckDBEngine) -> None:
    with pytest.raises(QueryExecutionError):
        duck_engine.execute("SELECT * FROM no_such_table", timeout_s=30)


def test_missing_dataset_raises(tmp_path: Path) -> None:
    engine = DuckDBEngine()
    try:
        with pytest.raises(FileNotFoundError):
            engine.register_parquet_dataset(tmp_path / "missing")
    finally:
        engine.close()


def test_timeout_interrupts_a_long_scan() -> None:
    # Uncancelled this scan takes several seconds. Interrupt should stop it immediately.
    engine = DuckDBEngine()
    started = time.perf_counter()
    try:
        with pytest.raises(QueryTimeoutError):
            engine.execute("SELECT SUM(hash(i)) FROM range(200000000) AS t(i)", timeout_s=0.4)
        assert time.perf_counter() - started < 3
    finally:
        engine.close()
