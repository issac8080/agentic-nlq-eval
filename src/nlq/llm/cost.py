"""Token-cost estimate used by the experiment budget guard.

The guard stops before the next call once spend has reached the cap. A single
call can still finish over the cap, because the token count is known only afterwards.
"""

from __future__ import annotations

from nlq.config import PricingSettings


class BudgetExceededError(Exception):
    """The next model call would start after the configured USD cap was already reached."""


class BudgetGuard:
    def __init__(self, cap_usd: float | None, pricing: PricingSettings) -> None:
        self.cap_usd = cap_usd
        self.pricing = pricing
        self.spent_usd = 0.0

    def before_call(self) -> None:
        if self.cap_usd is not None and self.spent_usd >= self.cap_usd:
            raise BudgetExceededError(
                f"Budget {self.cap_usd:.6f} USD reached; spent {self.spent_usd:.6f} USD"
            )

    def charge(self, input_tokens: int, output_tokens: int) -> float:
        cost = estimate_cost(input_tokens, output_tokens, self.pricing)
        self.spent_usd += cost
        return cost


def estimate_cost(input_tokens: int, output_tokens: int, pricing: PricingSettings) -> float:
    """USD from the configured per-million rates. Placeholder rates of zero keep tests free."""
    return (
        input_tokens / 1_000_000 * pricing.input_usd_per_million
        + output_tokens / 1_000_000 * pricing.output_usd_per_million
    )
