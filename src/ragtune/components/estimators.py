from typing import List, Optional
import numpy as np
from ragtune.core.types import ScoredDocument

class UtilityEstimator:
    """
    Baseline: Predicts utility based on simple metadata overlap.
    """
    def estimate(
        self, 
        pool: List[ScoredDocument], 
        ranked_indices: List[int]
    ) -> List[float]:
        estimates = [d.score for d in pool]
        if not ranked_indices:
            return estimates

        winners = [pool[i] for i in ranked_indices if (pool[i].reranker_score or 0) > 0.8]
        if winners:
            for i, doc in enumerate(pool):
                if i in ranked_indices: continue
                for winner in winners:
                    for key in ['source', 'section', 'category']:
                        if key in doc.metadata and key in winner.metadata:
                            if doc.metadata[key] == winner.metadata[key]:
                                estimates[i] *= 1.2
                                break
        return estimates

class SimilarityEstimator:
    """
    Intelligence: Predicts utility using semantic similarity (Embeddings).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._pool_embeddings: Optional[np.ndarray] = None
        self._pool_ids: List[str] = []

    def _ensure_embeddings(self, pool: List[ScoredDocument]):
        """Caches embeddings for the current pool to avoid re-computation."""
        current_ids = [d.id for d in pool]
        if self._pool_ids != current_ids:
            texts = [d.content for d in pool]
            self._pool_embeddings = self.model.encode(texts, convert_to_numpy=True)
            self._pool_ids = current_ids

    def estimate(
        self, 
        pool: List[ScoredDocument], 
        ranked_indices: List[int]
    ) -> List[float]:
        self._ensure_embeddings(pool)
        estimates = np.array([d.score for d in pool], dtype=float)
        
        if not ranked_indices or self._pool_embeddings is None:
            return estimates.tolist()

        # Identify "Winners" and their embeddings
        winner_indices = [i for i in ranked_indices if (pool[i].reranker_score or 0) > 0.8]
        if not winner_indices:
            return estimates.tolist()

        winner_embeddings = self._pool_embeddings[winner_indices]
        
        # Compute cosine similarity between ALL docs and winners
        # Handle zero-norm embeddings to avoid NaN
        pool_norms = np.linalg.norm(self._pool_embeddings, axis=1, keepdims=True)
        winner_norms = np.linalg.norm(winner_embeddings, axis=1, keepdims=True)
        
        # Replace 0 norms with 1 to avoid division by zero (similarity will be 0 anyway)
        pool_norms[pool_norms == 0] = 1.0
        winner_norms[winner_norms == 0] = 1.0
        
        norm_pool = self._pool_embeddings / pool_norms
        norm_winners = winner_embeddings / winner_norms
        
        # similarities shape: (len(pool), len(winners))
        similarities = np.dot(norm_pool, norm_winners.T)
        
        # Max similarity to any winner
        max_sims = np.max(similarities, axis=1)
        
        # Boost unranked docs based on similarity
        # We use a linear boost: score * (1 + similarity * weight)
        boost_weight = 0.5 
        for i in range(len(pool)):
            if i not in ranked_indices:
                estimates[i] *= (1.0 + max_sims[i] * boost_weight)

        return estimates.tolist()
