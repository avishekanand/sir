import pytest
from ragtune.core.pool import CandidatePool, PoolItem, ItemState
from ragtune.core.types import IllegalTransitionError

def test_a1_allowed_transitions_succeed():
    item = PoolItem(doc_id="d1", content="c1")
    pool = CandidatePool([item])
    
    # CANDIDATE -> IN_FLIGHT
    pool.transition(["d1"], ItemState.IN_FLIGHT)
    assert item.state == ItemState.IN_FLIGHT
    
    # IN_FLIGHT -> RERANKED
    pool.update_scores({"d1": 0.9}, strategy="cross_encoder")
    assert item.state == ItemState.RERANKED
    
    # RERANKED -> DROPPED (if allowed)
    pool.transition(["d1"], ItemState.DROPPED)
    assert item.state == ItemState.DROPPED

def test_a2_illegal_transition_raises():
    item = PoolItem(doc_id="d1", content="c1")
    pool = CandidatePool([item])
    
    # CANDIDATE -> RERANKED should fail (must go through IN_FLIGHT)
    with pytest.raises(IllegalTransitionError):
        pool.transition(["d1"], ItemState.RERANKED)
        
    pool.transition(["d1"], ItemState.IN_FLIGHT)
    pool.update_scores({"d1": 0.9}, strategy="cross_encoder")
    
    # RERANKED -> IN_FLIGHT should fail
    with pytest.raises(IllegalTransitionError):
        pool.transition(["d1"], ItemState.IN_FLIGHT)

def test_a3_unknown_id_handling():
    pool = CandidatePool([])
    # Decision: consistent behavior - currently pool.transition ignores missing IDs
    # If we want it to raise, we'd change it. For now, testing current behavior:
    # pool.transition ignores missing.
    pool.transition(["missing"], ItemState.IN_FLIGHT) 
    # No exception = success (consistent with current code)

def test_a4_update_scores_exclusivity():
    item = PoolItem(doc_id="d1", content="c1")
    pool = CandidatePool([item])
    
    # CANDIDATE -> UPDATE_SCORES fails
    with pytest.raises(IllegalTransitionError):
        pool.update_scores({"d1": 0.5}, strategy="test")
        
    pool.transition(["d1"], ItemState.IN_FLIGHT)
    pool.update_scores({"d1": 0.8}, strategy="test")
    assert item.state == ItemState.RERANKED
    assert item.reranker_score == 0.8

def test_a5_state_exclusivity():
    items = [PoolItem(doc_id=f"d{i}", content="c") for i in range(5)]
    pool = CandidatePool(items)
    
    pool.transition(["d0", "d1"], ItemState.IN_FLIGHT)
    pool.transition(["d2"], ItemState.DROPPED)
    pool.update_scores({"d0": 0.9}, strategy="s")
    
    states = [it.state for it in pool]
    assert states.count(ItemState.RERANKED) == 1 # d0
    assert states.count(ItemState.IN_FLIGHT) == 1 # d1
    assert states.count(ItemState.DROPPED) == 1 # d2
    assert states.count(ItemState.CANDIDATE) == 2 # d3, d4

def test_a6_get_eligible_filtering():
    items = [
        PoolItem(doc_id="d1", state=ItemState.CANDIDATE, content=""),
        PoolItem(doc_id="d2", state=ItemState.IN_FLIGHT, content=""),
        PoolItem(doc_id="d3", state=ItemState.RERANKED, content=""),
        PoolItem(doc_id="d4", state=ItemState.DROPPED, content="")
    ]
    pool = CandidatePool(items)
    eligible = pool.get_eligible()
    assert len(eligible) == 1
    assert eligible[0].doc_id == "d1"

def test_a7_no_lost_docs_invariant():
    n = 10
    items = [PoolItem(doc_id=str(i), content="c") for i in range(n)]
    pool = CandidatePool(items)
    
    import random
    ids = [str(i) for i in range(n)]
    
    # Random walk
    for _ in range(20):
        target_ids = random.sample(ids, 3)
        try:
            pool.transition(target_ids, random.choice(list(ItemState)))
        except IllegalTransitionError:
            pass
            
    assert len(list(pool)) == n

def test_a8_stable_access_and_order():
    ids = ["z", "a", "m", "b"]
    items = [PoolItem(doc_id=did, content="") for did in ids]
    pool = CandidatePool(items)
    
    # get_items should maintain requested order
    requested = ["m", "z", "a"]
    fetched = pool.get_items(requested)
    assert [it.doc_id for it in fetched] == requested
    
    # No duplicates if requested twice? 
    # Pool.get_items implementation: [self._items[did] for did in doc_ids if did in self._items]
    # Yes, it allows duplicates if requested.
    assert len(pool.get_items(["m", "m"])) == 2
