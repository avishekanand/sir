from typing import List
from ragtune.core.types import ScoredDocument, ReformulationResult

class SimpleConcatFusion:
    """Simple fusion that flattens candidates from multiple queries."""
    def fuse(self, results: List[ReformulationResult]) -> List[ScoredDocument]:
        all_docs = []
        seen = set()
        for res in results:
            for doc in res.candidates:
                if doc.id not in seen:
                    all_docs.append(doc)
                    seen.add(doc.id)
        return all_docs
