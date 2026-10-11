"""Isolate tests from developer environment variables."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlq.config import ENV_OVERRIDES


@pytest.fixture(autouse=True)
def _clear_nlq_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("NLQ_LLM_API_KEY", raising=False)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
