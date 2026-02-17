import pytest
from ragtune.core.pool import CandidatePool, PoolItem, ItemState
from ragtune.core.types import RAGtuneContext
from ragtune.core.interfaces import BaseEstimator
from typing import Dict

class FakeEstimator(BaseEstimator):
    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, float]:
        # Simple metadata overlap boost (like UtilityEstimator but fixed)
        eligible = pool.get_eligible()
        active = pool.get_active_items()
        winners = [it for it in active if it.state == ItemState.RERANKED and it.metadata.get("tag") == "win"]
        
        priorities = {it.doc_id: 0.5 for it in eligible}
        for it in eligible:
            for w in winners:
                if it.metadata.get("tag") == w.metadata.get("tag"):
                    priorities[it.doc_id] += 0.4
        return priorities

def test_b9_eligible_only_impact():
    items = [
        PoolItem(doc_id="d1", state=ItemState.CANDIDATE, content=""),
        PoolItem(doc_id="d2", state=ItemState.RERANKED, content="", reranker_score=0.9),
        PoolItem(doc_id="d3", state=ItemState.DROPPED, content="")
    ]
    pool = CandidatePool(items)
    ctx = RAGtuneContext(query="test", tracker=None)
    est = FakeEstimator()
    
    vals = est.value(pool, ctx)
    pool.apply_priorities(vals)
    
    assert items[0].priority_value == 0.5
    assert items[1].priority_value == 0.0 # Untouched
    assert items[2].priority_value == 0.0 # Untouched

def test_b10_estimator_determinism():
    items = [PoolItem(doc_id="d1", content="c1"), PoolItem(doc_id="d2", content="c2")]
    pool = CandidatePool(items)
    ctx = RAGtuneContext(query="test", tracker=None)
    est = FakeEstimator()
    
    v1 = est.value(pool, ctx)
    v2 = est.value(pool, ctx)
    assert v1 == v2

def test_b11_estimator_uses_reranked_evidence():
    items = [
        PoolItem(doc_id="win1", state=ItemState.RERANKED, content="", metadata={"tag": "win"}),
        PoolItem(doc_id="cand1", state=ItemState.CANDIDATE, content="", metadata={"tag": "win"}),
        PoolItem(doc_id="cand2", state=ItemState.CANDIDATE, content="", metadata={"tag": "other"})
    ]
    pool = CandidatePool(items)
    ctx = RAGtuneContext(query="test", tracker=None)
    est = FakeEstimator()
    
    priorities = est.value(pool, ctx)
    assert priorities["cand1"] == 0.9 # 0.5 + 0.4
    assert priorities["cand2"] == 0.5

def test_b12_estimator_no_mutation():
    items = [PoolItem(doc_id="d1", content="c1")]
    pool = CandidatePool(items)
    ctx = RAGtuneContext(query="test", tracker=None)
    est = FakeEstimator()
    
    initial_states = [it.state for it in pool]
    est.value(pool, ctx)
    current_states = [it.state for it in pool]
    assert initial_states == current_states
