import pytest
from ragtune.core.pool import CandidatePool, PoolItem, ItemState
from ragtune.core.types import RAGtuneContext, ControllerTrace
from ragtune.core.budget import CostTracker, CostBudget
from ragtune.components.estimators import UtilityEstimator

def test_estimator_boost_by_source():
    estimator = UtilityEstimator()
    
    # Create pool items
    items = [
        PoolItem(doc_id="a", content="A", metadata={"source": "wiki"}),
        PoolItem(doc_id="b", content="B", metadata={"source": "wiki"}),
        PoolItem(doc_id="c", content="C", metadata={"source": "news"})
    ]
    pool = CandidatePool(items)
    
    tracker = CostTracker(CostBudget(), ControllerTrace())
    context = RAGtuneContext(query="test", tracker=tracker)
    
    # Round 1: No ones reranked yet. 
    # UtilityEstimator uses retrieval score as baseline if no winners.
    # We didn't set sources, so scores are 0.
    estimates = estimator.value(pool, context)
    assert estimates["a"].priority == 0.0
    
    # Simulate doc_a being reranked with high score
    # First move A to IN_FLIGHT then RERANKED
    pool.transition(["a"], ItemState.IN_FLIGHT)
    pool.update_scores({"a": 0.9}, strategy="test")
    
    # doc_b should be boosted because it shares "wiki" source with winner doc_a
    new_estimates = estimator.value(pool, context)
    
    # Default boost is 1.2x of fallback (which is 0.0 here, so 1.2*0.0 = 0.0?)
    # Wait, UtilityEstimator.value has: score = max(it.sources.values()) * boost?
    # Let's give them some source scores.
    pool.get_items(["b", "c"])[0].sources["bm25"] = 0.5
    pool.get_items(["b", "c"])[1].sources["bm25"] = 0.5
    
    estimates_with_sources = estimator.value(pool, context)
    assert estimates_with_sources["b"].priority > 0.5  # Boosted
    assert estimates_with_sources["c"].priority == 0.5  # Not boosted
