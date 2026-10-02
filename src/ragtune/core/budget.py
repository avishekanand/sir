import math
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, Iterator, List, Tuple
from ragtune.core.types import ControllerTrace, CostObject, RemainingBudgetView, UNLIMITED
from ragtune.utils.config import config
from pydantic import BaseModel, Field, model_validator

# Pipeline stages that component-scoped limits ("<component>.<dimension>") can target.
# "embedding" is opt-in: components wrap their encoder calls in tracker.measure("embedding").
BUDGET_COMPONENTS = ("retrieval", "reformulation", "estimation", "embedding", "rerank", "assembly")
BUDGET_DIMENSIONS = ("latency_ms", "tokens", "docs", "calls")

# Global cost key -> dimension it is attributed to while a component is active.
_ATTRIBUTED_DIMENSION = {
    "tokens": "tokens",
    "rerank_docs": "docs",
    "rerank_calls": "calls",
    "retrieval_calls": "calls",
    "reformulations": "calls",
}


def parse_scoped_key(key: str) -> Tuple[str, str]:
    """Split a '<component>.<dimension>' limit key, validating both parts."""
    component, _, dimension = key.partition(".")
    if component not in BUDGET_COMPONENTS:
        raise ValueError(
            f"Unknown budget component {component!r} in {key!r}. Valid: {list(BUDGET_COMPONENTS)}"
        )
    if dimension not in BUDGET_DIMENSIONS:
        raise ValueError(
            f"Unknown budget dimension {dimension!r} in {key!r}. Valid: {list(BUDGET_DIMENSIONS)}"
        )
    return component, dimension


def group_by_component(state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Regroup scoped keys of a budget snapshot as {component: {dimension: amount}}."""
    grouped: Dict[str, Dict[str, float]] = {}
    for key, value in state.items():
        component, sep, dimension = key.partition(".")
        if sep and component in BUDGET_COMPONENTS and dimension in BUDGET_DIMENSIONS:
            grouped.setdefault(component, {})[dimension] = value
    return grouped

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

    @model_validator(mode="after")
    def validate_scoped_keys(self) -> "CostBudget":
        # A typo in a scoped key would otherwise silently disable that budget.
        for key in self.limits:
            if "." in key:
                parse_scoped_key(key)
        return self

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
        self._active: List[str] = []  # components currently inside measure()

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self._start_time) * 1000

    def is_exhausted(self, component: Optional[str] = None) -> bool:
        """Check if any critical budget is zero/negative.

        When a component is given, its '<component>.*' limits are checked too.
        """
        # For simple v0.54, we just check tokens and docs
        if "tokens" in self.budget.limits and self.consumed.get("tokens", 0) >= self.budget.limits["tokens"]:
            return True
        if "rerank_docs" in self.budget.limits and self.consumed.get("rerank_docs", 0) >= self.budget.limits["rerank_docs"]:
            return True
        if "latency_ms" in self.budget.limits and self.elapsed_ms >= self.budget.limits["latency_ms"]:
            return True
        return component is not None and self.component_exhausted(component)

    def component_exhausted(self, component: str) -> bool:
        """True if any '<component>.*' limit is used up. Global limits are not checked."""
        prefix = f"{component}."
        return any(
            self.consumed.get(key, 0) >= limit
            for key, limit in self.budget.limits.items()
            if key.startswith(prefix)
        )

    def remaining_view(self) -> RemainingBudgetView:
        """Provides an immutable-ish view of what's left for the Scheduler.

        Dimensions without a limit report UNLIMITED.
        """
        return RemainingBudgetView(
            remaining_tokens=self._remaining("tokens"),
            remaining_rerank_docs=self._remaining("rerank_docs"),
            remaining_rerank_calls=self._remaining("rerank_calls"),
            scoped={key: self._remaining(key) for key in self.budget.limits if "." in key},
        )

    def _remaining(self, key: str) -> int:
        limit = self.budget.limits.get(key)
        if limit is None or math.isinf(limit):
            return UNLIMITED
        return max(0, int(limit - self.consumed.get(key, 0)))

    @contextmanager
    def measure(self, component: str) -> Iterator[None]:
        """Time the block as '<component>.latency_ms' and attribute consumption inside it.

        Nested scopes are inclusive for latency; consumption is attributed to
        the innermost component only. Time is always recorded, even when the
        block raises or the global latency limit has passed.
        """
        self._active.append(component)
        start = time.perf_counter()
        try:
            yield
        finally:
            self._active.pop()
            key = f"{component}.latency_ms"
            self.consumed[key] = self.consumed.get(key, 0.0) + (time.perf_counter() - start) * 1000
            limit = self.budget.limits.get(key)
            if limit is not None and self.consumed[key] > limit:
                self.trace.add("budget", f"over_limit_{key}", total=self.consumed[key], limit=limit)

    def consume(self, cost: CostObject):
        """Standardized consumption of a CostObject."""
        if cost.tokens > 0: self.try_consume("tokens", cost.tokens)
        if cost.docs > 0: self.try_consume("rerank_docs", cost.docs)
        if cost.calls > 0: self.try_consume("rerank_calls", cost.calls)

    def try_consume(self, cost_type: str, amount: float = 1.0) -> bool:
        """Generic consumption method for any cost type.

        While a component is active (see measure()), global keys are also
        consumed under '<component>.<dimension>'; both must allow it.
        """
        allowed = self._consume(cost_type, amount)
        dimension = _ATTRIBUTED_DIMENSION.get(cost_type)
        if self._active and dimension:
            allowed = self._consume(f"{self._active[-1]}.{dimension}", amount) and allowed
        return allowed

    def _consume(self, cost_type: str, amount: float) -> bool:
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
            if "." not in cost_type:  # unlimited scoped attribution would double every trace line
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

    def snapshot(self) -> dict:
        data = self.consumed.copy()
        data["latency"] = self.elapsed_ms
        return data
