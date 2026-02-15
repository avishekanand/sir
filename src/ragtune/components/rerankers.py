from typing import List
from ragtune.core.interfaces import BaseReranker
from ragtune.core.types import ScoredDocument

class NoOpReranker(BaseReranker):
    """Identity reranker that returns documents as is."""
    def rerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        return documents

class SimulatedReranker(BaseReranker):
    """
    Reranker that simulates scoring and provides high scores for specific signals.
    Used for testing the Active Learning feedback loop.
    """
    def rerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        results = []
        for doc in documents:
            # Simulate high scores for documents containing the query
            is_match = query.lower() in doc.content.lower()
            reranker_score = 0.95 if is_match else 0.3
            
            results.append(doc.model_copy(update={
                "score": reranker_score,
                "reranker_score": reranker_score
            }))
        return results
