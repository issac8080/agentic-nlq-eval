"""Canonical result sets for execution accuracy and gold cross-checks.

Floats are rounded with half-even to the question tolerance before comparison.
Numeric strings from decimal engines are treated as numbers so Spark and DuckDB
agree even when one returns a string and the other a float.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any


def canonical_cell(value: Any, tolerance: float) -> str | bool | None:
    """One comparable text form for a cell. Numbers keep a fixed decimal text."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    number = _as_decimal(value)
    if number is None:
        return str(value)
    if tolerance > 0:
        places = max(0, int(round(-math.log10(tolerance))))
        quantum = Decimal("1").scaleb(-places)
        number = number.quantize(quantum, rounding=ROUND_HALF_EVEN)
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def normalise_rows(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    tolerance: float,
    ordered: bool,
    column_order: str = "name",
) -> list[dict[str, str | bool | None]]:
    """Round, order columns, and sort rows unless the question says order matters."""
    keys = sorted(columns) if column_order == "name" else list(columns)
    normalised: list[dict[str, str | bool | None]] = []
    for row in rows:
        normalised.append({key: canonical_cell(row.get(key), tolerance) for key in keys})
    if not ordered:
        normalised.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=True))
    return normalised


def result_hash(rows: list[dict[str, Any]]) -> str:
    """SHA-256 of a normalised result. Used to detect a gold-SQL drift."""
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def results_match(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    columns: list[str],
    *,
    tolerance: float,
    ordered: bool,
    column_order: str = "name",
) -> bool:
    return normalise_rows(
        left, columns, tolerance=tolerance, ordered=ordered, column_order=column_order
    ) == normalise_rows(
        right, columns, tolerance=tolerance, ordered=ordered, column_order=column_order
    )


def _as_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        try:
            return Decimal(text)
        except Exception:
            return None
    return None
