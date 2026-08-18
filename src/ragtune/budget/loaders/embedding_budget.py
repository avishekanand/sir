"""
Embedding Budget Loader
========================
Cost estimation for embedding API calls (query encoding).

Supports OpenAI, Cohere, and Voyage AI embedding pricing.
All parameters flow from BudgetConfig.

Source: OpenAI pricing (openai.com/api/pricing), Cohere pricing (cohere.com/pricing),
        Voyage AI pricing (docs.voyageai.com/pricing/)

Formula:
    cost = tokens × price_per_token
    cost_per_query = query_tokens × price_per_token
"""

from typing import Dict, Any, Optional

from ragtune.budget.base import BaseBudgetLoader, BudgetConfig
from ragtune.budget.factory import BudgetLoaderFactory
from ragtune.budget.hardware import estimate_energy_kwh, estimate_carbon_kg
from ragtune.budget.result import BudgetResult


# Default embedding pricing ($/1M tokens)
# Source: OpenAI, Cohere, Voyage AI pricing pages (mid-2026)
EMBEDDING_RATES = {
    "openai/text-embedding-3-small": {"input": 0.02, "dimensions": 1536},
    "openai/text-embedding-3-large": {"input": 0.13, "dimensions": 3072},
    "openai/text-embedding-ada-002": {"input": 0.10, "dimensions": 1536},
    "cohere/embed-v4": {"input": 0.12, "dimensions": 1024},
    "cohere/embed-v3": {"input": 0.10, "dimensions": 1024},
    "voyage/voyage-4": {"input": 0.06, "dimensions": 1024},
    "voyage/voyage-4-lite": {"input": 0.02, "dimensions": 512},
    "voyage/voyage-4-large": {"input": 0.12, "dimensions": 1024},
    "mistral/mistral-embed": {"input": 0.10, "dimensions": 1024},
    "google/gemini-embedding-001": {"input": 0.15, "dimensions": 768},
}


@BudgetLoaderFactory.register("embedding")
class EmbeddingBudgetLoader(BaseBudgetLoader):
    """Cost estimation for embedding API calls.

    Calculates cost based on token count and embedding model pricing.
    All parameters flow from BudgetConfig.

    Usage:
        loader = BudgetLoaderFactory.create("embedding", config={
            "embedding_model": "openai/text-embedding-3-small",
            "embedding_price_per_million": 0.02,
        })
        result = loader.calculate({"tokens": 1000})
    """

    def calculate(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetResult:
        ctx = context or {}

        # 'tokens' is the primary input; fall back to prompt_tokens when
        # callers pass the standard RAGtune context (e.g. the CLI passes
        # prompt_tokens/completion_tokens, not 'tokens').
        tokens = ctx.get("tokens")
        if tokens is None:
            tokens = ctx.get("prompt_tokens", 0)
        prompt_tokens = ctx.get("prompt_tokens", tokens)
        completion_tokens = ctx.get("completion_tokens", 0)

        cfg = self.config

        # Get embedding pricing
        embedding_model = cfg.extra.get(
            "embedding_model", "openai/text-embedding-3-small"
        )
        price_per_million = cfg.extra.get(
            "embedding_price_per_million",
            EMBEDDING_RATES.get(embedding_model, {}).get("input", 0.02),
        )

        # Cost: tokens × price_per_token
        cost = tokens * price_per_million / 1_000_000

        # Energy (minimal for API calls — mostly network overhead)
        # Embedding APIs are typically serverless; energy is negligible
        energy_kwh = 0.0
        carbon_kg = 0.0

        return BudgetResult(
            cost_usd=round(cost, 8),
            cost_per_million_tokens=round(price_per_million, 4),
            embedding_cost_usd=round(cost, 8),
            energy_kwh=energy_kwh,
            carbon_kg=carbon_kg,
            total_tokens=tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            breakdown={
                "embedding_model": embedding_model,
                "price_per_million": price_per_million,
                "tokens": tokens,
            },
        )
