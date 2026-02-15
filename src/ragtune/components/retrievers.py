from typing import List
from ragtune.core.interfaces import BaseRetriever
from ragtune.core.types import ScoredDocument

class InMemoryRetriever(BaseRetriever):
    def __init__(self, documents: List[ScoredDocument]):
        self.docs = documents

    def retrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        # Simple keyword match or just return top_k for testing
        results = [d for d in self.docs if query.lower() in d.content.lower()]
        if not results:
            results = self.docs # Fallback for "fake" tests
        return results[:top_k]
