from typing import List, Optional, Dict
import numpy as np
from ragtune.core.types import RAGtuneContext, EstimatorOutput
from ragtune.core.interfaces import BaseEstimator
from ragtune.core.pool import CandidatePool, ItemState
from ragtune.registry import registry

@registry.estimator("baseline")
class BaselineEstimator(BaseEstimator):
    """
    Returns the maximum retrieval score across all sources.
    """
    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, EstimatorOutput]:
        priorities = {}
        for item in pool.get_eligible():
            score = max(item.sources.values()) if item.sources else 0.0
            priorities[item.doc_id] = EstimatorOutput(priority=score)
        return priorities

    def needs_reformulation(self, context: RAGtuneContext, current_pool: CandidatePool) -> bool:
        return len(current_pool) < 5

@registry.estimator("utility")
class UtilityEstimator(BaseEstimator):
    """
    Predicts utility based on simple metadata overlap with already reranked winners.
    """
    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, EstimatorOutput]:
        eligible = pool.get_eligible()
        if not eligible:
            return {}

        active = pool.get_active_items()
        winners = [it for it in active if it.state == ItemState.RERANKED and (it.reranker_score or 0) > 0.8]
        
        priorities = {}
        for it in eligible:
            score = max(it.sources.values()) if it.sources else 0.0
            if winners:
                for winner in winners:
                    for key in ['source', 'section', 'category']:
                        if key in it.metadata and key in winner.metadata:
                            if it.metadata[key] == winner.metadata[key]:
                                score *= 1.2
                                break
            priorities[it.doc_id] = EstimatorOutput(priority=score, predicted_quality=score)
        return priorities

@registry.estimator("similarity")
class SimilarityEstimator(BaseEstimator):
    """
    Intelligence: Predicts utility using semantic similarity (Embeddings).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, EstimatorOutput]:
        eligible = pool.get_eligible()
        if not eligible:
            return {}

        active = pool.get_active_items()
        reranked = [it for it in active if it.state == ItemState.RERANKED]
        winners = [it for it in reranked if (it.reranker_score or 0) > 0.8]
        
        priorities = {}
        for it in eligible:
            priorities[it.doc_id] = EstimatorOutput(priority=(max(it.sources.values()) if it.sources else 0.0))
        
        if not winners:
            return priorities

        # Encode eligible and winners
        eligible_texts = [it.content for it in eligible]
        winner_texts = [it.content for it in winners]
        
        eligible_embs = self.model.encode(eligible_texts, convert_to_numpy=True)
        winner_embs = self.model.encode(winner_texts, convert_to_numpy=True)

        # Compute cosine similarity
        eligible_norms = np.linalg.norm(eligible_embs, axis=1, keepdims=True)
        winner_norms = np.linalg.norm(winner_embs, axis=1, keepdims=True)
        
        eligible_norms[eligible_norms == 0] = 1.0
        winner_norms[winner_norms == 0] = 1.0
        
        norm_eligible = eligible_embs / eligible_norms
        norm_winners = winner_embs / winner_norms
        
        similarities = np.dot(norm_eligible, norm_winners.T)
        max_sims = np.max(similarities, axis=1)
        
        boost_weight = 0.5 
        for i, it in enumerate(eligible):
            priorities[it.doc_id].priority *= (1.0 + max_sims[i] * boost_weight)
            priorities[it.doc_id].predicted_quality = max_sims[i]

        return priorities

@registry.estimator("composite")
class CompositeEstimator(BaseEstimator):
    """Combines multiple estimators with weighted or logical aggregation."""
    def __init__(self, estimators: List[BaseEstimator], weights: Optional[List[float]] = None, mode: str = "any"):
        self.estimators = estimators
        self.weights = weights or [1.0] * len(estimators)
        self.mode = mode # "any" or "all" for needs_reformulation

    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, EstimatorOutput]:
        combined_priorities = {}
        for est, weight in zip(self.estimators, self.weights):
            outputs = est.value(pool, context)
            for doc_id, out in outputs.items():
                if doc_id not in combined_priorities:
                    combined_priorities[doc_id] = EstimatorOutput(priority=0.0)
                combined_priorities[doc_id].priority += (out.priority * weight)
                # Average other metrics if they exist
                if out.predicted_quality is not None:
                    curr_q = combined_priorities[doc_id].predicted_quality or 0.0
                    combined_priorities[doc_id].predicted_quality = curr_q + out.predicted_quality / len(self.estimators)
        return combined_priorities

    def needs_reformulation(self, context: RAGtuneContext, current_pool: CandidatePool) -> bool:
        results = [est.needs_reformulation(context, current_pool) for est in self.estimators]
        if self.mode == "all":
            return all(results)
        return any(results)
