"""OpenAI-compatible chat completions over the standard library.

The model name and version come from config. Temperature defaults to 0.
Retries cover transient HTTP failures only, and each retry is reported through ``on_retry``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from nlq.config import LLMSettings
from nlq.llm.base import LLMClient, LLMResult
from nlq.llm.cost import BudgetGuard

TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


class TransientLLMError(Exception):
    """A provider failure that is safe to retry. The message is logged, not hidden."""


class OpenAICompatibleClient(LLMClient):
    """One concrete provider. ``base_url`` defaults to the public OpenAI chat endpoint."""

    def __init__(
        self,
        settings: LLMSettings,
        api_key: str,
        *,
        budget: BudgetGuard | None = None,
        on_retry: Callable[[int, Exception], None] | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 60,
    ) -> None:
        self.settings = settings
        self._api_key = api_key
        self._budget = budget
        self._on_retry = on_retry
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def complete(self, messages: list[dict[str, str]], **params: Any) -> LLMResult:
        if self._budget is not None:
            self._budget.before_call()
        temperature = float(params.get("temperature", self.settings.temperature))
        body = {
            "model": self.settings.model,
            "temperature": temperature,
            "messages": messages,
        }
        started = time.perf_counter()
        payload = self._post_with_retry(body)
        latency = time.perf_counter() - started
        choice = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
        result = LLMResult(
            text=choice or "",
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=latency,
        )
        if self._budget is not None:
            self._budget.charge(result.input_tokens, result.output_tokens)
        return result

    def _post_with_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, self.settings.max_api_retries)

        @retry(
            retry=retry_if_exception_type(TransientLLMError),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=True,
            before_sleep=self._before_sleep,
        )
        def attempt() -> dict[str, Any]:
            return self._post(body)

        return attempt()

    def _before_sleep(self, retry_state: Any) -> None:
        if self._on_retry is None:
            return
        error = retry_state.outcome.exception() if retry_state.outcome else None
        self._on_retry(retry_state.attempt_number, error or TransientLLMError("retry"))

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in TRANSIENT_STATUS:
                raise TransientLLMError(f"HTTP {exc.code}") from exc
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TransientLLMError(str(exc)) from exc
