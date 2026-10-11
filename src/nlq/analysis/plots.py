"""Figures for the four research questions. Files land in results/figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy_by_tier(frame: pd.DataFrame, path: Path) -> Path:
    """One grouped bar chart: execution accuracy by tier and system."""
    summary = frame.groupby(["tier", "system"], as_index=False)["execution_accuracy"].mean()
    pivot = summary.pivot(index="tier", columns="system", values="execution_accuracy")
    axis = pivot.plot(kind="bar", figsize=(8, 4))
    axis.set_ylabel("Execution accuracy")
    axis.set_xlabel("Tier")
    figure = axis.get_figure()
    return _save(figure, path)


def plot_stacked_latency(frame: pd.DataFrame, path: Path) -> Path:
    """Median LLM, SQL, and orchestration time at each scale, stacked."""
    medians = frame.groupby("scale", as_index=False)[["llm_time_s", "exec_time_s", "orchestration_s"]].median()
    axis = medians.set_index("scale")[["llm_time_s", "exec_time_s", "orchestration_s"]].plot(
        kind="bar", stacked=True, figsize=(8, 4)
    )
    axis.set_ylabel("Seconds")
    figure = axis.get_figure()
    return _save(figure, path)


def plot_log_log(row_counts: list[float], latencies: list[float], path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(row_counts, latencies)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Row count")
    axis.set_ylabel("Latency (s)")
    return _save(figure, path)


def plot_accuracy_faithfulness(table: dict[str, int], path: Path) -> Path:
    """Heatmap of the 2x2 correct/incorrect by faithful/unfaithful counts."""
    matrix = [
        [table.get("correct_faithful", 0), table.get("correct_unfaithful", 0)],
        [table.get("incorrect_faithful", 0), table.get("incorrect_unfaithful", 0)],
    ]
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix)
    axis.set_xticks([0, 1], ["faithful", "unfaithful"])
    axis.set_yticks([0, 1], ["correct", "incorrect"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    figure.colorbar(image)
    return _save(figure, path)


def write_result_tables(directory: Path, **frames: pd.DataFrame) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in frames.items():
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
