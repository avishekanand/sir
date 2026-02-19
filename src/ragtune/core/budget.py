import time
from typing import Optional, Dict, Any
from ragtune.core.types import ControllerTrace, CostObject, RemainingBudgetView
from ragtune.utils.config import config
from pydantic import BaseModel, Field, model_validator

class CostBudget(BaseModel):
    """
    Budget limits for various operations.
    Default keys: 'tokens', 'rerank_docs', 'reformulations', 'latency_ms'
    """
    limits: Dict[str, float] = Field(default_factory=lambda: {
        "tokens": 4000,
        "rerank_docs": 50,
        "rerank_calls": 10,
        "retrieval_calls": 10,
        "reformulations": 1,
        "latency_ms": 2000.0
    })

    @model_validator(mode='before')
    @classmethod
    def map_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If 'limits' is not provided but 'max_*' fields are, populate 'limits'
            if "limits" not in data:
                # Start with default limits or empty? 
                # To maintain v0.4 behavior, we should probably start with empty and only set what's provided
                # but if we want to fallback to defaults, we need to be careful.
                # Actually, if the user explicitly provides ANY limit, we should probably use that.
                new_limits = {}
                if "max_tokens" in data: new_limits["tokens"] = data.pop("max_tokens")
                if "max_reranker_docs" in data: new_limits["rerank_docs"] = data.pop("max_reranker_docs")
                if "max_reformulations" in data: new_limits["reformulations"] = data.pop("max_reformulations")
                if "max_latency_ms" in data: new_limits["latency_ms"] = data.pop("max_latency_ms")
                
                if new_limits:
                    data["limits"] = new_limits
        return data

    @classmethod
    def simple(cls, tokens=4000, docs=50, calls=10, reformulations=1, latency=2000.0):
        return cls(limits={
            "tokens": tokens,
            "rerank_docs": docs,
            "rerank_calls": calls,
            "reformulations": reformulations,
            "latency_ms": latency
        })

class CostTracker:
    def __init__(self, budget: CostBudget, trace: ControllerTrace):
        self.budget = budget
        self.trace = trace
        self.consumed: Dict[str, float] = {}
        self._start_time = time.time()

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self._start_time) * 1000

    def is_exhausted(self) -> bool:
        """Check if any critical budget is zero/negative."""
        # For simple v0.54, we just check tokens and docs
        if "tokens" in self.budget.limits and self.consumed.get("tokens", 0) >= self.budget.limits["tokens"]:
            return True
        if "rerank_docs" in self.budget.limits and self.consumed.get("rerank_docs", 0) >= self.budget.limits["rerank_docs"]:
            return True
        if "latency_ms" in self.budget.limits and self.elapsed_ms >= self.budget.limits["latency_ms"]:
            return True
        return False

    def remaining_view(self) -> RemainingBudgetView:
        """Provides an immutable-ish view of what's left for the Scheduler."""
        return RemainingBudgetView(
            remaining_tokens=max(0, int(self.budget.limits.get("tokens", 0) - self.consumed.get("tokens", 0))),
            remaining_rerank_docs=max(0, int(self.budget.limits.get("rerank_docs", 0) - self.consumed.get("rerank_docs", 0))),
            remaining_rerank_calls=max(0, int(self.budget.limits.get("rerank_calls", 0) - self.consumed.get("rerank_calls", 0))),
        )

    def consume(self, cost: CostObject):
        """Standardized consumption of a CostObject."""
        if cost.tokens > 0: self.try_consume("tokens", cost.tokens)
        if cost.docs > 0: self.try_consume("rerank_docs", cost.docs)
        if cost.calls > 0: self.try_consume("rerank_calls", cost.calls)

    def try_consume(self, cost_type: str, amount: float = 1.0) -> bool:
        """Generic consumption method for any cost type."""
        # 1. Check Latency (Global constraint)
        if cost_type != "latency_ms" and "latency_ms" in self.budget.limits:
            if self.elapsed_ms > self.budget.limits["latency_ms"]:
                self.trace.add("budget", f"deny_{cost_type}", reason="latency_exceeded", elapsed=self.elapsed_ms)
                return False

        # 2. Check Capacity
        limit = self.budget.limits.get(cost_type)
        if limit is None:
            # If no limit is defined, we allow it but don't track it as a hard limit?
            # Or should we deny it? Let's allow it but warn in trace.
            current = self.consumed.get(cost_type, 0.0)
            self.consumed[cost_type] = current + amount
            return True

        current = self.consumed.get(cost_type, 0.0)
        self.consumed[cost_type] = current + amount
        
        if self.consumed[cost_type] <= limit:
            self.trace.add("budget", f"consume_{cost_type}", count=amount, total=self.consumed[cost_type])
            return True
        
        self.trace.add("budget", f"over_limit_{cost_type}", count=amount, total=self.consumed[cost_type], limit=limit)
        return False

    # Legacy-style helpers for convenience
    def try_consume_reformulation(self, n=1) -> bool:
        return self.try_consume("reformulations", n)

    def try_consume_retrieval(self, n=1) -> bool:
        return self.try_consume("retrieval_calls", n)

    def try_consume_rerank(self, n_docs: int) -> bool:
        return self.try_consume("rerank_docs", n_docs)

    def try_consume_tokens(self, n_tokens: int) -> bool:
        return self.try_consume("tokens", n_tokens)

    def snapshot(self) -> dict:
        data = self.consumed.copy()
        data["latency"] = self.elapsed_ms
        return data
