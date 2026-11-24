import time
from typing import Optional, Dict, Any, List, Tuple
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
                # Legacy max_* fields: map to limits dict. Unspecified keys fall through
                # to the Field default_factory (all limits). If no legacy keys are present,
                # data["limits"] is not set and the Field default applies.
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
        # Cost-model accounting (see specs/spec-cost-model.md). Kept separate
        # from `consumed` because these are priced, not budgeted — nothing
        # enforces a limit on them.
        self.llm_usage: Dict[str, List[int]] = {}       # model -> [tok_in, tok_out]
        self.rerank_calls: List[Tuple[str, int]] = []   # (model, n_docs) per call
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
            # No limit defined: allow unconditionally and track for observability.
            current = self.consumed.get(cost_type, 0.0)
            self.consumed[cost_type] = current + amount
            self.trace.add("budget", f"consume_{cost_type}_unlimited", count=amount, total=self.consumed[cost_type])
            return True

        # Always accumulate into consumed — even denied attempts count toward the
        # running total so that is_exhausted() and remaining_view() stay accurate
        # for budget projection. Returns False when the new total exceeds the limit.
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

    # ── Cost-model accounting ─────────────────────────────────────────────────

    def record_llm_usage(self, model: str, tokens_in: int, tokens_out: int) -> None:
        """
        Record tokens billed by an API model, for the dollar term of the cost
        model. Not budget-enforced: `tokens` in `consumed` is the assembly
        context budget, a different quantity.
        """
        bucket = self.llm_usage.setdefault(model, [0, 0])
        bucket[0] += int(tokens_in)
        bucket[1] += int(tokens_out)
        self.trace.add(
            "cost", "llm_usage", model=model,
            tokens_in=int(tokens_in), tokens_out=int(tokens_out),
        )

    def record_rerank_call(self, model: str, n_docs: int) -> None:
        """
        Record one reranker invocation. `n_docs` is the batch the scheduler
        handed over, which sets the effective GPU batch and therefore the
        per-document cost (a ~3.9x factor over the tuned 1-20 range), so calls
        are recorded individually rather than summed.
        """
        if n_docs <= 0:
            return
        self.rerank_calls.append((model, int(n_docs)))
        self.trace.add("cost", "rerank_call", model=model, n_docs=int(n_docs))

    def snapshot(self) -> dict:
        data = self.consumed.copy()
        data["latency"] = self.elapsed_ms
        data["llm_usage"] = {m: tuple(v) for m, v in self.llm_usage.items()}
        data["rerank_calls"] = list(self.rerank_calls)
        return data
