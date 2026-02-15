from typing import List
from ragtune.core.interfaces import BaseAssembler
from ragtune.core.types import ScoredDocument
from ragtune.core.budget import CostTracker

class GreedyAssembler(BaseAssembler):
    """
    Greedy assembler that selects documents based on score, 
    fitting as many as the token budget allows.
    """
    def assemble(self, candidates: List[ScoredDocument], tracker: CostTracker) -> List[ScoredDocument]:
        # Sort candidates by score descending
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        result = []
        for doc in sorted_candidates:
            if tracker.try_consume_tokens(doc.token_count):
                result.append(doc)
        return result
