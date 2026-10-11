"""Paired tests on question-level outcomes, not on individual repeats.

Repeats are majority-voted into one binary accuracy per question so a question
asked three times is not treated as three independent questions. Latency uses
the median repeat. Faithfulness intervals resample whole questions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict
from scipy.stats import binomtest, wilcoxon


class PairTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_a: str
    system_b: str
    p_value: float
    p_holm: float
    accuracy_a: float
    accuracy_b: float
    difference: float
    difference_low: float
    difference_high: float


def aggregate_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per question and system. Ties in the repeat vote count as incorrect."""
    rows = []
    for keys, part in frame.groupby(["question_id", "system", "tier"]):
        votes = [bool(value) for value in part["execution_accuracy"]]
        rows.append(
            {
                "question_id": keys[0],
                "system": keys[1],
                "tier": keys[2],
                "execution_accuracy": int(sum(votes) > len(votes) / 2),
            }
        )
    return pd.DataFrame(rows)


def mcnemar_pairs(frame: pd.DataFrame, systems: list[str], *, seed: int = 42, boots: int = 1000) -> list[PairTest]:
    """Exact McNemar tests for every system pair, then Holm adjustment."""
    aggregated = aggregate_accuracy(frame)
    tests: list[PairTest] = []
    for index, left in enumerate(systems):
        for right in systems[index + 1 :]:
            paired = _align(aggregated, left, right)
            b = int(((paired[left] == 1) & (paired[right] == 0)).sum())
            c = int(((paired[left] == 0) & (paired[right] == 1)).sum())
            p_value = 1.0 if b + c == 0 else float(binomtest(b, b + c, p=0.5, alternative="two-sided").pvalue)
            acc_a = float(paired[left].mean()) if len(paired) else 0.0
            acc_b = float(paired[right].mean()) if len(paired) else 0.0
            low, high = _bootstrap_difference(paired[left].to_numpy(), paired[right].to_numpy(), seed, boots)
            tests.append(
                PairTest(
                    system_a=left,
                    system_b=right,
                    p_value=p_value,
                    p_holm=p_value,
                    accuracy_a=acc_a,
                    accuracy_b=acc_b,
                    difference=acc_a - acc_b,
                    difference_low=low,
                    difference_high=high,
                )
            )
    adjusted = holm([item.p_value for item in tests])
    return [
        item.model_copy(update={"p_holm": adjusted[index]})
        for index, item in enumerate(tests)
    ]


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment. The returned list matches the input order."""
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def latency_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR of each latency component, after taking the median repeat per question."""
    per_question = (
        frame.groupby(["question_id", "system", "scale"], as_index=False)[
            ["end_to_end_s", "llm_time_s", "exec_time_s", "orchestration_s"]
        ].median()
    )
    rows = []
    for (system, scale), part in per_question.groupby(["system", "scale"]):
        for column in ["end_to_end_s", "llm_time_s", "exec_time_s", "orchestration_s"]:
            series = part[column]
            rows.append(
                {
                    "system": system,
                    "scale": scale,
                    "component": column,
                    "median": float(series.median()),
                    "q1": float(series.quantile(0.25)),
                    "q3": float(series.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def wilcoxon_pairs(frame: pd.DataFrame, systems: list[str], component: str = "end_to_end_s") -> list[dict[str, Any]]:
    per_question = frame.groupby(["question_id", "system"], as_index=False)[component].median()
    results = []
    for index, left in enumerate(systems):
        for right in systems[index + 1 :]:
            paired = per_question.pivot(index="question_id", columns="system", values=component).dropna()
            if left not in paired or right not in paired or len(paired) == 0:
                continue
            difference = paired[left] - paired[right]
            if np.allclose(difference, 0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(paired[left], paired[right], zero_method="wilcox").pvalue)
            results.append({"system_a": left, "system_b": right, "component": component, "p_value": p_value})
    return results


def log_log_scaling(row_counts: list[float], latencies: list[float]) -> dict[str, float]:
    """Fit log(latency) = a + b log(rows). ``b`` is the scaling exponent."""
    x = np.log(np.asarray(row_counts, dtype=float))
    y = np.log(np.asarray(latencies, dtype=float))
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    params = np.asarray(fit.params, dtype=float)
    interval = np.asarray(fit.conf_int(alpha=0.05), dtype=float)
    return {
        "intercept": float(params[0]),
        "exponent": float(params[1]),
        "exponent_low": float(interval[1, 0]),
        "exponent_high": float(interval[1, 1]),
    }


def cluster_bootstrap_rate(
    question_ids: list[str],
    supported: list[bool],
    *,
    seed: int = 42,
    boots: int = 1000,
) -> tuple[float, float, float]:
    """Support rate with questions as the resampled clusters."""
    grouped: dict[str, list[bool]] = defaultdict(list)
    for question_id, flag in zip(question_ids, supported, strict=True):
        grouped[question_id].append(bool(flag))
    keys = list(grouped)
    if not keys:
        return 0.0, 0.0, 0.0
    point = sum(sum(values) for values in grouped.values()) / sum(len(values) for values in grouped.values())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(boots):
        draw = rng.choice(keys, size=len(keys), replace=True)
        flags = [flag for key in draw for flag in grouped[key]]
        samples.append(sum(flags) / len(flags))
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _align(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    pivot = frame.pivot(index="question_id", columns="system", values="execution_accuracy")
    return pivot[[left, right]].dropna().astype(int)


def _bootstrap_difference(left: np.ndarray, right: np.ndarray, seed: int, boots: int) -> tuple[float, float]:
    if len(left) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(boots):
        index = rng.integers(0, len(left), size=len(left))
        deltas.append(float(left[index].mean() - right[index].mean()))
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
