"""
Token Budget Loader
====================
Simple token-counting budget. No GPU model needed — just counts tokens
and applies a per-token rate.

Useful for API-based deployments where the provider charges per token
(e.g., OpenAI, Anthropic).
"""

from typing import Dict, Any, Optional

from ragtune.budget.base import BaseBudgetLoader, BudgetConfig
from ragtune.budget.factory import BudgetLoaderFactory
from ragtune.budget.result import BudgetResult


# Default per-token rates ($/1M tokens)
# Source: OpenAI pricing (openai.com/api/pricing), mid-2026
# Input/output match GPT-4o; cached rate is 50% discount per OpenAI prompt caching.
DEFAULT_RATES = {
    "input": 2.50,
    "output": 10.00,
    "cached_input": 1.25,  # 50% discount on GPT-4o input rate
}


@BudgetLoaderFactory.register("token")
class TokenBudgetLoader(BaseBudgetLoader):
    """Simple token-counting budget.

    Cost = (input_tokens × input_rate) + (output_tokens × output_rate)

    Useful for API-based deployments. Rates can be set in config.
    """

    def calculate(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetResult:
        ctx = context or {}
        prompt_tokens = ctx.get("prompt_tokens", 512)
        completion_tokens = ctx.get("completion_tokens", 256)
        cached_tokens = ctx.get("cached_tokens", 0)

        input_rate = self.config.extra.get("input_rate", DEFAULT_RATES["input"])
        output_rate = self.config.extra.get("output_rate", DEFAULT_RATES["output"])
        cached_rate = self.config.extra.get(
            "cached_rate", DEFAULT_RATES["cached_input"]
        )

        uncached_prompt = max(0, prompt_tokens - cached_tokens)
        cost = (
            (uncached_prompt / 1_000_000 * input_rate)
            + (cached_tokens / 1_000_000 * cached_rate)
            + (completion_tokens / 1_000_000 * output_rate)
        )
        total_tokens = prompt_tokens + completion_tokens

        return BudgetResult(
            cost_usd=round(cost, 6),
            cost_per_million_tokens=round(cost / max(total_tokens, 1) * 1_000_000, 4),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            breakdown={
                "input_rate": input_rate,
                "output_rate": output_rate,
                "cached_rate": cached_rate,
                "uncached_prompt": uncached_prompt,
            },
        )
