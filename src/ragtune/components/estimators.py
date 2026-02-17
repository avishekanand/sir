from typing import List, Optional, Dict
import numpy as np
from ragtune.core.types import RAGtuneContext
from ragtune.core.interfaces import BaseEstimator
from ragtune.core.pool import CandidatePool, ItemState

class BaselineEstimator(BaseEstimator):
    """
    Returns the maximum retrieval score across all sources.
    """
    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, float]:
        priorities = {}
        for item in pool.get_eligible():
            priorities[item.doc_id] = max(item.sources.values()) if item.sources else 0.0
        return priorities

class UtilityEstimator(BaseEstimator):
    """
    Predicts utility based on simple metadata overlap with already reranked winners.
    """
    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, float]:
        eligible = pool.get_eligible()
        if not eligible:
            return {}

        active = pool.get_active_items()
        winners = [it for it in active if it.state == ItemState.RERANKED and (it.reranker_score or 0) > 0.8]
        
        priorities = {it.doc_id: (max(it.sources.values()) if it.sources else 0.0) for it in eligible}
        
        if winners:
            for it in eligible:
                for winner in winners:
                    for key in ['source', 'section', 'category']:
                        if key in it.metadata and key in winner.metadata:
                            if it.metadata[key] == winner.metadata[key]:
                                priorities[it.doc_id] *= 1.2
                                break
        return priorities

class SimilarityEstimator(BaseEstimator):
    """
    Intelligence: Predicts utility using semantic similarity (Embeddings).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._cache_embeddings: Dict[str, np.ndarray] = {}

    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, float]:
        eligible = pool.get_eligible()
        if not eligible:
            return {}

        active = pool.get_active_items()
        reranked = [it for it in active if it.state == ItemState.RERANKED]
        winners = [it for it in reranked if (it.reranker_score or 0) > 0.8]
        
        priorities = {it.doc_id: (max(it.sources.values()) if it.sources else 0.0) for it in eligible}
        
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
            priorities[it.doc_id] *= (1.0 + max_sims[i] * boost_weight)

        return priorities
