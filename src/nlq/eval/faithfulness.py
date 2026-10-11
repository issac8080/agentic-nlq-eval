"""Claim checks against the logged result and SQL. The model extracts claims; the checks are code.

Labels are supported, contradicted, unsupported, or non_verifiable. Explanatory
claims stay non_verifiable unless they contain a comparison or a number the
result can confirm. A number that is not in the result is unsupported rather
than guessed with a new query; the verdict records that no extra query was run.
"""

from __future__ import annotations

import csv
import random
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nlq.eval.normalise import canonical_cell

ClaimType = Literal[
    "numeric",
    "comparative_or_ranking",
    "entity",
    "procedural",
    "explanatory_causal",
]
ClaimLabel = Literal["supported", "contradicted", "unsupported", "non_verifiable"]
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_COMPARE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:is\s+)?(greater than|higher than|more than|lower than|less than|fewer than)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_ARITHMETIC = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_text: str
    claim_type: ClaimType


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_text: str
    claim_type: ClaimType
    label: ClaimLabel
    verification_query: str | None = None


class FaithfulnessSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_support_rate: float | None
    unsupported_claim_rate: float | None
    contradiction_rate: float | None
    calculation_correctness: float | None
    provenance_integrity: bool
    faithful: bool


def verify_claims(
    claims: list[Claim],
    rows: list[dict[str, Any]],
    sql: str,
    *,
    tolerance: float,
    schema_tables: list[str],
) -> list[ClaimVerdict]:
    cells = _cells(rows, tolerance)
    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        verdicts.append(_verify_one(claim, cells, rows, sql, tolerance, schema_tables))
    return verdicts


def calculation_correctness(texts: list[str], tolerance: float) -> float | None:
    """Recompute arithmetic the narrative states. None when no equation is present."""
    checks: list[bool] = []
    for text in texts:
        for match in _ARITHMETIC.finditer(text):
            left, op, right, result = match.groups()
            expected = _apply(op, float(left), float(right))
            checks.append(canonical_cell(expected, tolerance) == canonical_cell(float(result), tolerance))
    if not checks:
        return None
    return sum(checks) / len(checks)


def provenance_ok(
    *,
    reported_sql: str,
    executed_sql: str,
    reported_tables: list[str],
    executed_tables: list[str],
    reported_filters: list[str],
    executed_filters: list[str],
    reported_row_count: int,
    executed_row_count: int,
) -> bool:
    """Reported provenance has to be what actually ran, not a paraphrase."""
    return (
        " ".join(reported_sql.split()) == " ".join(executed_sql.split())
        and set(reported_tables) == set(executed_tables)
        and set(reported_filters) == set(executed_filters)
        and reported_row_count == executed_row_count
    )


def summarise(
    verdicts: list[ClaimVerdict],
    *,
    calculation: float | None,
    provenance: bool,
) -> FaithfulnessSummary:
    scored = [item for item in verdicts if item.label != "non_verifiable"]
    if not scored:
        support = unsupported = contradiction = None
        faithful = provenance
    else:
        support = _rate(scored, "supported")
        unsupported = _rate(scored, "unsupported")
        contradiction = _rate(scored, "contradicted")
        faithful = provenance and all(item.label == "supported" for item in scored)
    return FaithfulnessSummary(
        claim_support_rate=support,
        unsupported_claim_rate=unsupported,
        contradiction_rate=contradiction,
        calculation_correctness=calculation,
        provenance_integrity=provenance,
        faithful=faithful,
    )


def contingency(correct: list[bool], faithful: list[bool]) -> dict[str, int]:
    """2x2 counts of execution accuracy against faithfulness. Lists are paired."""
    table = {"correct_faithful": 0, "correct_unfaithful": 0, "incorrect_faithful": 0, "incorrect_unfaithful": 0}
    for is_correct, is_faithful in zip(correct, faithful, strict=True):
        key = f"{'correct' if is_correct else 'incorrect'}_{'faithful' if is_faithful else 'unfaithful'}"
        table[key] += 1
    return table


def export_annotation_sample(
    rows: list[dict[str, Any]],
    path: str,
    *,
    fraction: float = 0.22,
    seed: int = 42,
) -> int:
    """Write a stratified sample for a second annotator. The system name is omitted on purpose."""
    if not 0.20 <= fraction <= 0.25:
        raise ValueError("annotation sample fraction must be between 0.20 and 0.25")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("tier", "na")), []).append(row)
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for tier_rows in grouped.values():
        count = max(1, round(len(tier_rows) * fraction)) if tier_rows else 0
        count = min(count, len(tier_rows))
        picked = rng.sample(tier_rows, count)
        for item in picked:
            chosen.append(
                {
                    "sample_id": len(chosen) + 1,
                    "question_id": item.get("question_id", ""),
                    "tier": item.get("tier", ""),
                    "claim_text": item.get("claim_text", ""),
                    "claim_type": item.get("claim_type", ""),
                    "narrative": item.get("narrative", ""),
                    "human_label": "",
                }
            )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "question_id", "tier", "claim_text", "claim_type", "narrative", "human_label"],
        )
        writer.writeheader()
        writer.writerows(chosen)
    return len(chosen)


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa for two annotators. Empty input returns 0."""
    if len(labels_a) != len(labels_b) or not labels_a:
        if not labels_a and not labels_b:
            return 0.0
        if len(labels_a) != len(labels_b):
            raise ValueError("label lists must be the same length")
    categories = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    matrix = {left: {right: 0 for right in categories} for left in categories}
    for left, right in zip(labels_a, labels_b, strict=True):
        matrix[left][right] += 1
    observed = sum(matrix[name][name] for name in categories) / n
    expected = 0.0
    for name in categories:
        row_sum = sum(matrix[name].values()) / n
        col_sum = sum(matrix[other][name] for other in categories) / n
        expected += row_sum * col_sum
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def _verify_one(
    claim: Claim,
    cells: set[str],
    rows: list[dict[str, Any]],
    sql: str,
    tolerance: float,
    schema_tables: list[str],
) -> ClaimVerdict:
    text = claim.claim_text
    if claim.claim_type == "explanatory_causal":
        comparison = _comparison_label(text, tolerance)
        if comparison is not None:
            return ClaimVerdict(claim_text=text, claim_type=claim.claim_type, label=comparison)
        if _NUMBER.search(text) is None:
            return ClaimVerdict(claim_text=text, claim_type=claim.claim_type, label="non_verifiable")
    if claim.claim_type == "comparative_or_ranking":
        comparison = _comparison_label(text, tolerance)
        if comparison is not None:
            return ClaimVerdict(claim_text=text, claim_type=claim.claim_type, label=comparison)
    if claim.claim_type == "procedural":
        return ClaimVerdict(
            claim_text=text,
            claim_type=claim.claim_type,
            label=_procedural_label(text, sql, schema_tables),
        )
    if claim.claim_type == "entity":
        return ClaimVerdict(claim_text=text, claim_type=claim.claim_type, label=_entity_label(text, rows))
    numbers = _NUMBER.findall(text)
    if not numbers:
        label: ClaimLabel = "unsupported"
    else:
        hits = [canonical_cell(number, tolerance) in cells for number in numbers]
        if all(hits):
            label = "supported"
        elif any(hits):
            label = "contradicted"
        else:
            label = "unsupported"
    return ClaimVerdict(
        claim_text=text,
        claim_type=claim.claim_type,
        label=label,
        verification_query=None if label != "unsupported" else None,
    )


def _comparison_label(text: str, tolerance: float) -> ClaimLabel | None:
    match = _COMPARE.search(text)
    if match is None:
        return None
    left, relation, right = match.group(1), match.group(2).lower(), match.group(3)
    left_n, right_n = float(left), float(right)
    greater = relation in {"greater than", "higher than", "more than"}
    holds = left_n > right_n if greater else left_n < right_n
    if canonical_cell(left_n, tolerance) == canonical_cell(right_n, tolerance):
        holds = False
    return "supported" if holds else "contradicted"


def _entity_label(text: str, rows: list[dict[str, Any]]) -> ClaimLabel:
    for row in rows:
        for value in row.values():
            if isinstance(value, str) and len(value) >= 2 and value in text:
                return "supported"
    return "unsupported"


def _procedural_label(text: str, sql: str, schema_tables: list[str]) -> ClaimLabel:
    mentioned = [table for table in schema_tables if table in text]
    if not mentioned:
        return "unsupported"
    missing = [table for table in mentioned if table not in sql]
    if missing:
        return "contradicted"
    return "supported"


def _cells(rows: list[dict[str, Any]], tolerance: float) -> set[str]:
    found: set[str] = set()
    for row in rows:
        for value in row.values():
            cell = canonical_cell(value, tolerance)
            if isinstance(cell, str):
                found.add(cell)
    return found


def _apply(operator: str, left: float, right: float) -> float:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    return left / right


def _rate(verdicts: list[ClaimVerdict], label: str) -> float:
    return sum(item.label == label for item in verdicts) / len(verdicts)
