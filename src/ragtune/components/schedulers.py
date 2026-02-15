from typing import List, Optional, Union
from ragtune.core.interfaces import BaseScheduler
from ragtune.core.types import ScoredDocument, BatchProposal, RerankStrategy, RAGtuneContext
from ragtune.components.estimators import UtilityEstimator, SimilarityEstimator
from ragtune.registry import registry

@registry.scheduler("active-learning")
class ActiveLearningScheduler(BaseScheduler):
    def __init__(
        self, 
        batch_size: int = 5, 
        initial_strategy: RerankStrategy = RerankStrategy.CROSS_ENCODER,
        estimator: Optional[Union[UtilityEstimator, SimilarityEstimator]] = None
    ):
        self.batch_size = batch_size
        self.strategy = initial_strategy
        self.estimator = estimator or UtilityEstimator()

    def propose_next_batch(
        self, 
        pool: List[ScoredDocument], 
        processed_indices: List[int], 
        context: RAGtuneContext
    ) -> Optional[BatchProposal]:
        if len(processed_indices) >= len(pool):
            return None

        # 1. Get Utility Estimates
        # If using SimilarityEstimator, this will involve semantic boosting
        utilities = self.estimator.estimate(pool, processed_indices, context)
        
        # 2. Filter & Sort Candidates
        candidates = []
        for i, util in enumerate(utilities):
            if i not in processed_indices:
                candidates.append((i, util))
        
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Hybrid Strategy Logic (Ambiguity Escalation)
        # If the top candidates have very low utility scores, 
        # or if they are very close together (ambiguous), 
        # we might want to escalate to a stronger model (LLM).
        
        current_strategy = self.strategy
        if len(candidates) >= 2:
            top_util = candidates[0][1]
            gap = top_util - candidates[1][1]
            
            # Escalation Heuristic: 
            # If the best doc isn't clearly better (gap < 0.05) 
            # AND we are currently in CROSS_ENCODER mode, 
            # switch this batch to LLM.
            if gap < 0.05 and current_strategy == RerankStrategy.CROSS_ENCODER:
                current_strategy = RerankStrategy.LLM
        
        # 4. Pick Top-K
        next_indices = [c[0] for c in candidates[:self.batch_size]]
        avg_utility = sum(c[1] for c in candidates[:self.batch_size]) / len(next_indices)
        
        return BatchProposal(
            document_indices=next_indices,
            strategy=current_strategy,
            estimated_utility=avg_utility
        )
