from typing import List
from ragtune.core.types import ScoredDocument

class UtilityEstimator:
    """
    Predicts the utility of unranked docs based on feedback from ranked docs.
    """
    def estimate(
        self, 
        pool: List[ScoredDocument], 
        ranked_indices: List[int]
    ) -> List[float]:
        # 1. Baseline: Start with current scores
        estimates = [d.score for d in pool]
        
        if not ranked_indices:
            return estimates

        # 2. Feedback Loop (The "Dynamic Adaptation")
        # Identify "Winners" (high reranker scores)
        winners = [pool[i] for i in ranked_indices if (pool[i].reranker_score or 0) > 0.8]
        
        if winners:
            # Boost unranked docs that share metadata/source with winners
            for i, doc in enumerate(pool):
                if i in ranked_indices:
                    continue
                
                # Heuristic: Boost if from same source/metadata
                for winner in winners:
                    # Check for metadata overlap (e.g. 'source' or 'category')
                    for key in ['source', 'section', 'category']:
                        if key in doc.metadata and key in winner.metadata:
                            if doc.metadata[key] == winner.metadata[key]:
                                estimates[i] *= 1.2 # 20% Boost
                                break # Boost once per doc
                        
        return estimates
