"""Client interface every system calls. A test double implements the same method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LLMResult(BaseModel):
    """One completion. Latency is the provider round trip, not orchestration time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_s: float = Field(ge=0)


class LLMClient(ABC):
    """``complete`` is the only way systems talk to a model."""

    @abstractmethod
    def complete(self, messages: list[dict[str, str]], **params: Any) -> LLMResult:
        """Return the model text and token counts. ``params`` override temperature and similar knobs."""
