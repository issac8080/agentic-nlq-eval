"""Shared answering interface."""

from __future__ import annotations

from typing import Protocol

from nlq.benchmark.schema import BenchmarkQuestion
from nlq.systems.answer import SystemResult


class AnsweringSystem(Protocol):
    name: str

    def answer(self, question: BenchmarkQuestion) -> SystemResult:
        """Answer one benchmark question."""
