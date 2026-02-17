from typing import List
from ragtune.core.interfaces import BaseAssembler
from ragtune.core.types import ScoredDocument, RAGtuneContext
from ragtune.core.pool import PoolItem
from ragtune.registry import registry

@registry.assembler("greedy")
class GreedyAssembler(BaseAssembler):
    """
    Greedy assembler that selects documents based on score, 
    fitting as many as the token budget allows.
    """
    def assemble(self, candidates: List[PoolItem], context: RAGtuneContext) -> List[ScoredDocument]:
        # Sort candidates by final_score descending
        # Stable tie-break by initial_rank
        sorted_candidates = sorted(candidates, key=lambda x: (-x.final_score(), x.initial_rank))
        
        result = []
        for it in sorted_candidates:
            # Conversion to output format
            doc = ScoredDocument(
                id=it.doc_id,
                content=it.content,
                metadata=it.metadata,
                score=it.final_score(),
                reranker_score=it.reranker_score,
                initial_rank=it.initial_rank,
                original_score=max(it.sources.values()) if it.sources else 0.0
            )
            
            # Simple token estimation for v0.54
            token_count = it.metadata.get("token_count", 0)
            if doc.content and token_count == 0:
                token_count = len(doc.content.split()) * 1.3
                
            if context.tracker.try_consume_tokens(int(token_count)):
                doc.token_count = int(token_count)
                result.append(doc)
        return result
