"""
Reranking Budget Loader
========================
Cost estimation for reranking API calls (Cohere, Voyage AI).

Supports per-query pricing (Cohere style) and per-token pricing (Voyage style).
All parameters flow from BudgetConfig.

Source: Cohere pricing (cohere.com/pricing), Voyage AI pricing (docs.voyageai.com/pricing/)

Pricing models:
- Per-query: $X per 1,000 queries (up to 100 docs each)
- Per-token: $X per 1M tokens (tokens = query × docs + doc tokens)
"""

from typing import Dict, Any, Optional

from ragtune.budget.base import BaseBudgetLoader, BudgetConfig
from ragtune.budget.factory import BudgetLoaderFactory
from ragtune.budget.result import BudgetResult


# Default reranking pricing
# Source: Cohere, Voyage AI pricing pages (mid-2026)
RERANKING_RATES = {
    "cohere/rerank-v4-pro": {"per_query": 0.0025, "max_docs": 100},  # $2.50/1k queries
    "cohere/rerank-v4-fast": {"per_query": 0.0020, "max_docs": 100},  # $2.00/1k queries
    "cohere/rerank-v3.5": {"per_query": 0.0020, "max_docs": 100},  # $2.00/1k queries
    "voyage/rerank-2.5": {
        "per_million_tokens": 0.05,
        "max_docs": 100,
    },  # $0.05/1M tokens
    "voyage/rerank-2.5-lite": {
        "per_million_tokens": 0.02,
        "max_docs": 100,
    },  # $0.02/1M tokens
}


@BudgetLoaderFactory.register("reranking")
class RerankingBudgetLoader(BaseBudgetLoader):
    """Cost estimation for reranking API calls.

    Supports two pricing models:
    1. Per-query (Cohere): $X per 1,000 queries, up to 100 docs each
    2. Per-token (Voyage): $X per 1M tokens (query × docs + doc tokens)

    All parameters flow from BudgetConfig.

    Usage:
        # Cohere-style (per-query)
        loader = BudgetLoaderFactory.create("reranking", config={
            "extra": {"reranking_model": "cohere/rerank-v4-pro"}
        })
        result = loader.calculate({"queries": 10, "docs_per_query": 50})

        # Voyage-style (per-token)
        loader = BudgetLoaderFactory.create("reranking", config={
            "extra": {"reranking_model": "voyage/rerank-2.5"}
        })
        result = loader.calculate({
            "queries": 10,
            "docs_per_query": 50,
            "query_tokens": 20,
            "doc_tokens_per_doc": 200,
        })
    """

    def calculate(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetResult:
        ctx = context or {}

        queries = ctx.get("queries", 1)
        docs_per_query = ctx.get("docs_per_query", 10)
        query_tokens = ctx.get("query_tokens", 20)
        doc_tokens_per_doc = ctx.get("doc_tokens_per_doc", 200)

        cfg = self.config

        # Get reranking pricing
        reranking_model = cfg.extra.get("reranking_model", "cohere/rerank-v4-pro")
        rate_info = RERANKING_RATES.get(
            reranking_model, {"per_query": 0.002, "max_docs": 100}
        )

        # Calculate cost based on pricing model
        if "per_query" in rate_info:
            # Per-query pricing (Cohere style)
            # Cost = queries × price_per_query
            # Cap at max_docs per query
            effective_docs = min(docs_per_query, rate_info.get("max_docs", 100))
            cost = queries * rate_info["per_query"]
            total_tokens = queries * (
                query_tokens + effective_docs * doc_tokens_per_doc
            )
        else:
            # Per-token pricing (Voyage style)
            # Tokens = (query_tokens × docs_per_query) + (docs_per_query × doc_tokens_per_doc)
            total_tokens = queries * (
                query_tokens + docs_per_query * doc_tokens_per_doc
            )
            price_per_million = rate_info.get("per_million_tokens", 0.05)
            cost = total_tokens * price_per_million / 1_000_000

        return BudgetResult(
            cost_usd=round(cost, 8),
            cost_per_million_tokens=round(cost / max(total_tokens, 1) * 1_000_000, 4)
            if total_tokens > 0
            else 0.0,
            reranking_cost_usd=round(cost, 8),
            total_tokens=total_tokens,
            prompt_tokens=queries * query_tokens,
            completion_tokens=queries * docs_per_query * doc_tokens_per_doc,
            breakdown={
                "queries": float(queries),
                "docs_per_query": float(docs_per_query),
                "total_tokens": float(total_tokens),
            },
        )
