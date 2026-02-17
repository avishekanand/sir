import pytest
from ragtune.core.pool import CandidatePool, PoolItem, ItemState
from ragtune.core.types import IllegalTransitionError

def test_pool_initialization():
    items = [
        PoolItem(doc_id="d1", content="content 1", sources={"s1": 0.5}),
        PoolItem(doc_id="d2", content="content 2", sources={"s1": 0.4})
    ]
    pool = CandidatePool(items)
    assert len(pool) == 2
    assert all(it.state == ItemState.CANDIDATE for it in pool)

def test_legal_transitions():
    items = [PoolItem(doc_id="d1", content="c1")]
    pool = CandidatePool(items)
    
    # CANDIDATE -> IN_FLIGHT
    pool.transition(["d1"], ItemState.IN_FLIGHT)
    assert pool.get_items(["d1"])[0].state == ItemState.IN_FLIGHT
    
    # IN_FLIGHT -> RERANKED
    pool.update_scores({"d1": 0.9}, strategy="cross-encoder")
    assert pool.get_items(["d1"])[0].state == ItemState.RERANKED
    assert pool.get_items(["d1"])[0].reranker_score == 0.9

def test_illegal_transitions():
    items = [PoolItem(doc_id="d1", content="c1")]
    pool = CandidatePool(items)
    
    # Cannot go RERANKED directly from CANDIDATE
    with pytest.raises(IllegalTransitionError):
        pool.transition(["d1"], ItemState.RERANKED)
        
    pool.transition(["d1"], ItemState.IN_FLIGHT)
    
    # Cannot go back to CANDIDATE
    with pytest.raises(IllegalTransitionError):
        pool.transition(["d1"], ItemState.CANDIDATE)

def test_final_score_precedence():
    item = PoolItem(doc_id="d1", content="c1", sources={"ret": 0.5})
    # Initial precedence: Retrieval
    assert item.final_score() == 0.5
    
    # Priority value set by estimator
    item.priority_value = 0.7
    assert item.final_score() == 0.7
    
    # Reranker score set
    item.reranker_score = 0.9
    assert item.final_score() == 0.9

def test_active_items_filtering():
    items = [
        PoolItem(doc_id="d1", content="c1", state=ItemState.CANDIDATE),
        PoolItem(doc_id="d2", content="c2", state=ItemState.RERANKED),
        PoolItem(doc_id="d3", content="c3", state=ItemState.DROPPED),
        PoolItem(doc_id="d4", content="c4", state=ItemState.IN_FLIGHT)
    ]
    pool = CandidatePool(items)
    active = pool.get_active_items()
    active_ids = {it.doc_id for it in active}
    assert active_ids == {"d1", "d2"}
    assert "d3" not in active_ids
    assert "d4" not in active_ids
