from typing import Optional, Dict, Any
import time
from ragtune.core.types import ControllerTrace
from pydantic import BaseModel, Field, model_validator

class CostBudget(BaseModel):
    """
    Budget limits for various operations.
    Default keys: 'tokens', 'rerank_docs', 'reformulations', 'latency_ms'
    """
    limits: Dict[str, float] = Field(default_factory=lambda: {
        "tokens": 4000,
        "rerank_docs": 50,
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
    def simple(cls, tokens=4000, docs=50, reformulations=1, latency=2000.0):
        return cls(limits={
            "tokens": tokens,
            "rerank_docs": docs,
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
        if current + amount <= limit:
            self.consumed[cost_type] = current + amount
            self.trace.add("budget", f"consume_{cost_type}", count=amount, total=self.consumed[cost_type])
            return True
        
        self.trace.add("budget", f"deny_{cost_type}", reason="limit_reached", requested=amount, current=current, limit=limit)
        return False

    # Legacy-style helpers for convenience
    def try_consume_reformulation(self, n=1) -> bool:
        return self.try_consume("reformulations", n)

    def try_consume_rerank(self, n_docs: int) -> bool:
        return self.try_consume("rerank_docs", n_docs)

    def try_consume_tokens(self, n_tokens: int) -> bool:
        return self.try_consume("tokens", n_tokens)

    def snapshot(self) -> dict:
        data = self.consumed.copy()
        data["latency"] = self.elapsed_ms
        return data
