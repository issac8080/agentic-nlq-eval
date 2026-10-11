"""DuckDB execution for gold-SQL cross-checks.

Benchmark answers are executed on Spark. This engine reads the same Parquet
files so a later gold-check can compare result sets on an independent engine.
It is not a second answering system.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from nlq.data.generate_tpch import quote_ident, sql_string_literal
from nlq.data.views import VIEW_SELECTS, create_view_statements
from nlq.engine.results import QueryResult, call_with_timeout, rows_from_records, strip_trailing_semicolon


class DuckDBEngine:
    """In-memory DuckDB database with the TPC-H Parquet files registered as views."""

    def __init__(self, *, timeout_s: float = 120, result_row_cap: int = 200) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if result_row_cap < 1:
            raise ValueError("result_row_cap must be positive")
        self._timeout_s = timeout_s
        self._result_row_cap = result_row_cap
        self._connection = duckdb.connect(database=":memory:")
        self._registered: list[str] = []

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._connection

    def register_parquet_dataset(self, dataset_dir: Path) -> list[str]:
        """Register each ``*.parquet`` file as a view named after the file stem."""
        directory = Path(dataset_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Parquet dataset directory not found: {directory}")
        files = sorted(path for path in directory.glob("*.parquet") if path.is_file())
        if not files:
            raise FileNotFoundError(f"No Parquet files in {directory}")
        names: list[str] = []
        for path in files:
            name = path.stem
            # CREATE VIEW cannot take a prepared parameter, so the path is inlined as a literal.
            literal = sql_string_literal(path.resolve().as_posix())
            self._connection.execute(
                f"CREATE OR REPLACE VIEW {quote_ident(name)} AS SELECT * FROM read_parquet({literal})"
            )
            names.append(name)
        self._registered = names
        return names

    def register_business_views(self) -> list[str]:
        """Add the renamed business views on top of the raw TPC-H tables."""
        for statement in create_view_statements(temporary=False):
            self._connection.execute(statement)
        names = list(VIEW_SELECTS)
        self._registered.extend(names)
        return names

    def execute(
        self,
        sql: str,
        *,
        timeout_s: float | None = None,
        apply_row_cap: bool = False,
        row_cap: int | None = None,
    ) -> QueryResult:
        """Run one statement. ``interrupt`` is the cancel hook when the timeout fires."""
        statement = strip_trailing_semicolon(sql)
        timeout = self._timeout_s if timeout_s is None else timeout_s
        cap = self._result_row_cap if row_cap is None else row_cap
        if apply_row_cap and cap < 1:
            raise ValueError("row_cap must be positive when apply_row_cap is set")

        def action() -> QueryResult:
            started = time.perf_counter()
            if apply_row_cap:
                count_row = self._connection.execute(
                    f"SELECT COUNT(*) AS _nlq_n FROM ({statement}) AS _nlq_count"
                ).fetchone()
                total = int(count_row[0])
                cursor = self._connection.execute(
                    f"SELECT * FROM ({statement}) AS _nlq_cap LIMIT {int(cap)}"
                )
                used_cap: int | None = cap
            else:
                cursor = self._connection.execute(statement)
                used_cap = None
                total = None
            description = cursor.description or []
            columns = [column[0] for column in description]
            records = [tuple(row) for row in cursor.fetchall()]
            if total is None:
                total = len(records)
            truncated = total > len(records)
            return QueryResult(
                columns=columns,
                rows=rows_from_records(columns, records),
                row_count=total,
                truncated=truncated,
                row_cap=used_cap,
                exec_time_s=time.perf_counter() - started,
            )

        def cancel() -> None:
            self._connection.interrupt()

        return call_with_timeout(action, timeout_s=timeout, cancel=cancel, sql=statement)

    def close(self) -> None:
        self._connection.close()
