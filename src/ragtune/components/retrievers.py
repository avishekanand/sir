from typing import List, Union
from ragtune.core.interfaces import BaseRetriever
from ragtune.core.types import ScoredDocument, RAGtuneContext
from ragtune.registry import registry

@registry.retriever("in-memory")
class InMemoryRetriever(BaseRetriever):
    def __init__(self, documents: List[Union[ScoredDocument, dict]]):
        self.docs = []
        for d in documents:
            if isinstance(d, dict):
                self.docs.append(ScoredDocument(**d))
            else:
                self.docs.append(d)

    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        # Simple keyword match or just return top_k for testing
        results = [d for d in self.docs if context.query.lower() in d.content.lower()]
        if not results:
            results = self.docs # Fallback for "fake" tests
        return results[:top_k]
