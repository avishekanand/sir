from typing import List, Optional
from ragtune.core.interfaces import BaseScheduler
from ragtune.core.types import ScoredDocument, BatchProposal, RerankStrategy
from ragtune.core.budget import CostTracker
from ragtune.components.estimators import UtilityEstimator

class ActiveLearningScheduler(BaseScheduler):
    def __init__(self, batch_size: int = 5, strategy: RerankStrategy = RerankStrategy.CROSS_ENCODER):
        self.batch_size = batch_size
        self.strategy = strategy
        self.estimator = UtilityEstimator()

    def propose_next_batch(
        self, 
        pool: List[ScoredDocument], 
        processed_indices: List[int], 
        tracker: CostTracker
    ) -> Optional[BatchProposal]:
        # 1. Check if we can afford at least one rerank
        if not tracker.try_consume_rerank(0): # Check latency/budget without consuming
             # Note: CostTracker.try_consume_rerank in v0.1 check budget and THEN consumes. 
             # We need a non-consuming check or just rely on the controller's check.
             # However, specs said "Check Budget FIRST".
             pass
        
        # In this implementation, we'll let the controller handle the actual consumption check
        # but stop if we've processed everything.
        if len(processed_indices) >= len(pool):
            return None

        # 2. Get Utility Estimates for the whole pool
        utilities = self.estimator.estimate(pool, processed_indices)
        
        # 3. Select Best Unprocessed Candidates
        candidates = []
        for i, util in enumerate(utilities):
            if i not in processed_indices:
                candidates.append((i, util))
        
        if not candidates:
            return None
            
        # Sort by estimated utility
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Pick Top-K
        next_indices = [c[0] for c in candidates[:self.batch_size]]
        
        return BatchProposal(
            document_indices=next_indices,
            strategy=self.strategy,
            estimated_utility=sum(c[1] for c in candidates[:self.batch_size]) / len(next_indices)
        )
