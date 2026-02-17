import pytest
from ragtune.core.pool import PoolItem, ItemState
from ragtune.core.types import RAGtuneContext, ScoredDocument, ControllerTrace
from ragtune.core.budget import CostTracker, CostBudget
from ragtune.components.assemblers import GreedyAssembler

def test_greedy_assembler_precedence_invariant():
    """
    Test that every reranked document is ranked higher than any candidate doc
    in the final list, even if the candidate has a higher retrieval score.
    """
    assembler = GreedyAssembler()
    tracker = CostTracker(CostBudget(), ControllerTrace())
    ctx = RAGtuneContext(query="test", tracker=tracker)
    
    # d1: Candidate with high retrieval score (0.9)
    # d2: Reranked with low reranker score (0.1)
    # d3: Reranked with high reranker score (0.8)
    # d4: Candidate with low retrieval score (0.2)
    items = [
        PoolItem(doc_id="d1", content="c1", sources={"ret": 0.9}, initial_rank=0),
        PoolItem(doc_id="d2", content="c2", sources={"ret": 0.5}, initial_rank=1, reranker_score=0.1, state=ItemState.RERANKED),
        PoolItem(doc_id="d3", content="c3", sources={"ret": 0.5}, initial_rank=2, reranker_score=0.8, state=ItemState.RERANKED),
        PoolItem(doc_id="d4", content="c4", sources={"ret": 0.2}, initial_rank=3)
    ]
    
    results = assembler.assemble(items, ctx)
    
    # Expected order: Reranked first (sorted by reranker_score desc), then Candidates (sorted by retrieval desc)
    # 1. d3 (Reranked, 0.8)
    # 2. d2 (Reranked, 0.1)
    # 3. d1 (Candidate, 0.9)
    # 4. d4 (Candidate, 0.2)
    
    assert results[0].id == "d3"
    assert results[1].id == "d2"
    assert results[2].id == "d1"
    assert results[3].id == "d4"
    
    # Verify that all reranked items precede all non-reranked items
    reranked_indices = [i for i, r in enumerate(results) if r.reranker_score is not None]
    candidate_indices = [i for i, r in enumerate(results) if r.reranker_score is None]
    
    if reranked_indices and candidate_indices:
        assert max(reranked_indices) < min(candidate_indices)
