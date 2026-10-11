"""Shared query result model and the timeout wrapper used by both engines.

Results are JSON-safe so a later run log can store them without a second
conversion step. Decimals stay strings so money values do not become binary floats.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

JsonValue = int | float | str | bool | None


class QueryTimeoutError(Exception):
    """The engine cancelled a query because it exceeded the configured timeout."""

    def __init__(self, sql: str, timeout_s: float, exec_time_s: float) -> None:
        self.sql = sql
        self.timeout_s = timeout_s
        self.exec_time_s = exec_time_s
        super().__init__(f"Query exceeded {timeout_s:.3f}s and was cancelled after {exec_time_s:.3f}s")


class QueryExecutionError(Exception):
    """The engine rejected the SQL or failed while running it."""

    def __init__(self, sql: str, message: str) -> None:
        self.sql = sql
        self.message = message
        super().__init__(message)


class QueryResult(BaseModel):
    """Rows collected from one SQL statement, plus the timing the latency eval needs.

    ``row_count`` is the full result size. ``rows`` may be shorter when a cap is
    applied, which is how narration stays bounded while accuracy still knows
    whether the result was truncated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: list[str]
    rows: list[dict[str, JsonValue]]
    row_count: int = Field(ge=0)
    truncated: bool
    row_cap: int | None = Field(default=None, ge=1)
    exec_time_s: float = Field(ge=0)


def strip_trailing_semicolon(sql: str) -> str:
    """Drop one trailing semicolon so the statement can be wrapped for a row cap."""
    text = sql.strip()
    if text.endswith(";"):
        text = text[:-1].strip()
    if text == "":
        raise ValueError("SQL is empty")
    return text


def json_safe(value: Any) -> JsonValue:
    """Convert an engine cell into a JSON value, preserving decimal text."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    item = getattr(value, "item", None)
    if callable(item):
        return json_safe(item())
    raise TypeError(f"Cannot serialise query cell of type {type(value).__name__}")


def rows_from_records(columns: list[str], records: list[tuple[Any, ...]]) -> list[dict[str, JsonValue]]:
    """Zip columnar names with positional records."""
    return [{column: json_safe(value) for column, value in zip(columns, record, strict=True)} for record in records]


def call_with_timeout(
    action: Callable[[], QueryResult],
    *,
    timeout_s: float,
    cancel: Callable[[], None],
    sql: str,
    cancel_grace_s: float = 5,
) -> QueryResult:
    """Run ``action`` on a daemon thread and call ``cancel`` if it exceeds ``timeout_s``.

    The cancel hook is engine-specific (Spark job group, DuckDB interrupt). The
    grace period gives that hook time to unblock the thread; the error is still
    a timeout when the budget was already spent.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if cancel_grace_s < 0:
        raise ValueError("cancel_grace_s must be >= 0")

    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = action()
        except Exception as exc:  # noqa: BLE001 — re-raised on the caller thread
            box["error"] = exc

    thread = threading.Thread(target=target, name="nlq-query", daemon=True)
    started = time.perf_counter()
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        cancel()
        thread.join(cancel_grace_s)
        raise QueryTimeoutError(sql=sql, timeout_s=timeout_s, exec_time_s=time.perf_counter() - started)
    if "error" in box:
        error = box["error"]
        if isinstance(error, (QueryTimeoutError, QueryExecutionError)):
            raise error
        raise QueryExecutionError(sql=sql, message=str(error)) from error
    value = box.get("value")
    if not isinstance(value, QueryResult):
        raise QueryExecutionError(sql=sql, message="Query finished without a result")
    return value
