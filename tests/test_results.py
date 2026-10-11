"""JSON-safe cells and the shared query timeout wrapper."""

from __future__ import annotations

import math
import threading
import time
from datetime import date
from decimal import Decimal

import pytest

from nlq.engine.results import QueryResult, QueryTimeoutError, call_with_timeout, json_safe


def test_json_safe_preserves_decimals_and_drops_non_finite_floats() -> None:
    assert json_safe(Decimal("123.40")) == "123.40"
    assert json_safe(date(1992, 1, 2)) == "1992-01-02"
    assert json_safe(math.nan) is None
    assert json_safe(True) is True
    with pytest.raises(TypeError):
        json_safe(object())


def test_call_with_timeout_invokes_cancel() -> None:
    stop = threading.Event()

    def action() -> QueryResult:
        stop.wait(5)
        return QueryResult(columns=["n"], rows=[{"n": 1}], row_count=1, truncated=False, exec_time_s=0)

    def cancel() -> None:
        stop.set()

    started = time.perf_counter()
    with pytest.raises(QueryTimeoutError):
        call_with_timeout(action, timeout_s=0.2, cancel=cancel, sql="SELECT 1", cancel_grace_s=1)
    assert stop.is_set()
    assert time.perf_counter() - started < 2
