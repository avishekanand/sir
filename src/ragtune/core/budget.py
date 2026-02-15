from typing import Optional
import time
from ragtune.core.types import ControllerTrace
from pydantic import BaseModel

class CostBudget(BaseModel):
    max_tokens: int = 4000
    max_reranker_docs: int = 50
    max_reformulations: int = 1
    max_latency_ms: float = 2000.0

class CostTracker:
    def __init__(self, budget: CostBudget, trace: ControllerTrace):
        self.budget = budget
        self.trace = trace
        self._tokens_used = 0
        self._rerank_docs_used = 0
        self._reformulations_used = 0
        self._start_time = time.time()

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self._start_time) * 1000

    def try_consume_reformulation(self, n=1) -> bool:
        if self._reformulations_used + n <= self.budget.max_reformulations:
            self._reformulations_used += n
            self.trace.add("budget", "consume_reformulation", count=n)
            return True
        self.trace.add("budget", "deny_reformulation", reason="limit_reached")
        return False

    def try_consume_rerank(self, n_docs: int) -> bool:
        # Check Latency
        if self.elapsed_ms > self.budget.max_latency_ms:
            self.trace.add("budget", "deny_rerank", reason="latency_exceeded", elapsed=self.elapsed_ms)
            return False
        
        # Check Capacity
        remaining = self.budget.max_reranker_docs - self._rerank_docs_used
        if n_docs > remaining:
            self.trace.add("budget", "deny_rerank", reason="doc_limit_exceeded", requested=n_docs, remaining=remaining)
            return False
            
        self._rerank_docs_used += n_docs
        self.trace.add("budget", "consume_rerank", count=n_docs)
        return True

    def try_consume_tokens(self, n_tokens: int) -> bool:
        if self._tokens_used + n_tokens <= self.budget.max_tokens:
            self._tokens_used += n_tokens
            return True
        return False

    def snapshot(self) -> dict:
        return {
            "tokens": self._tokens_used,
            "latency": self.elapsed_ms,
            "rerank_docs": self._rerank_docs_used
        }
