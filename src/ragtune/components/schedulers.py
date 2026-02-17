from typing import List, Optional, Union
from ragtune.core.interfaces import BaseScheduler
from ragtune.core.types import BatchProposal, RerankStrategy, RemainingBudgetView, CostObject
from ragtune.core.pool import CandidatePool, ItemState
from ragtune.registry import registry

@registry.scheduler("active-learning")
class ActiveLearningScheduler(BaseScheduler):
    def __init__(
        self, 
        batch_size: int = 5, 
        strategy: str = "cross_encoder",
    ):
        self.batch_size = batch_size
        self.strategy = strategy

    def select_batch(self, pool: CandidatePool, budget: RemainingBudgetView) -> Optional[BatchProposal]:
        eligible = pool.get_eligible()
        if not eligible or budget.remaining_rerank_docs <= 0:
            return None

        # Sort by priority_value (set by Estimator in controller loop)
        # Tie-break by initial_rank then doc_id
        eligible.sort(key=lambda x: (-x.priority_value, x.initial_rank, x.doc_id))
        
        batch_size = min(self.batch_size, budget.remaining_rerank_docs, len(eligible))
        if batch_size <= 0:
            return None
            
        selected = eligible[:batch_size]
        doc_ids = [it.doc_id for it in selected]
        
        # Strategy escalation logic (v0.54 simplified)
        current_strategy = self.strategy
        if len(selected) >= 2:
            gap = selected[0].priority_value - selected[1].priority_value
            if gap < 0.05 and current_strategy == "cross_encoder":
                current_strategy = "llm"
        
        return BatchProposal(
            doc_ids=doc_ids,
            strategy=current_strategy,
            expected_cost=CostObject(docs=len(doc_ids), calls=1)
        )
@registry.scheduler("graceful-degradation")
class GracefulDegradationScheduler(BaseScheduler):
    def __init__(self, llm_limit: int = 3, cross_encoder_limit: int = 10, batch_size: int = 5):
        self.llm_limit = llm_limit
        self.cross_encoder_limit = cross_encoder_limit
        self.batch_size = batch_size

    def select_batch(self, pool: CandidatePool, budget: RemainingBudgetView) -> Optional[BatchProposal]:
        eligible = pool.get_eligible()
        if not eligible or budget.remaining_rerank_docs <= 0:
            return None

        # Count how many have been reranked by each strategy
        active = pool.get_active_items()
        reranked = [it for it in active if it.state == ItemState.RERANKED]
        num_llm = len([it for it in reranked if it.reranker_strategy == "llm"])
        num_ce = len([it for it in reranked if it.reranker_strategy == "cross_encoder"])
        
        if num_llm < self.llm_limit:
            strategy = "llm"
            rem_strategy = self.llm_limit - num_llm
        elif num_ce < self.cross_encoder_limit:
            strategy = "cross_encoder"
            rem_strategy = self.cross_encoder_limit - num_ce
        else:
            return None

        # Sort eligible by priority_value (from retrieval score usually)
        eligible.sort(key=lambda x: (-x.priority_value, x.initial_rank, x.doc_id))
        
        batch_size = min(self.batch_size, budget.remaining_rerank_docs, rem_strategy, len(eligible))
        if batch_size <= 0:
            return None
            
        selected = eligible[:batch_size]
        return BatchProposal(
            doc_ids=[it.doc_id for it in selected],
            strategy=strategy,
            expected_cost=CostObject(docs=len(selected), calls=1)
        )
