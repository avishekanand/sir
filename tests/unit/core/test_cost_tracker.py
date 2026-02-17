import pytest
from ragtune.core.budget import CostTracker, CostBudget
from ragtune.core.types import CostObject, RemainingBudgetView, ControllerTrace

def test_d18_consume_respects_hard_limits():
    budget = CostBudget(limits={"rerank_docs": 10})
    tracker = CostTracker(budget, ControllerTrace())
    
    # consume 7 ok
    assert tracker.try_consume("rerank_docs", 7) is True
    assert tracker.consumed["rerank_docs"] == 7
    
    # consume 4 more - returns False but tracks overage for termination logic
    assert tracker.try_consume("rerank_docs", 4) is False
    assert tracker.is_exhausted() is True
    assert tracker.consumed["rerank_docs"] == 11

def test_d19_controller_only_mutation():
    budget = CostBudget(limits={"rerank_docs": 10})
    tracker = CostTracker(budget, ControllerTrace())
    view = tracker.remaining_view()
    
    # RemainingBudgetView is a Pydantic model (immutable-ish in usage)
    view.remaining_rerank_docs = 0
    assert tracker.remaining_view().remaining_rerank_docs == 10

def test_d20_monotonicity():
    budget = CostBudget(limits={"rerank_docs": 100})
    tracker = CostTracker(budget, ControllerTrace())
    tracker.consume(CostObject(docs=10))
    c1 = tracker.consumed["rerank_docs"]
    tracker.consume(CostObject(docs=20))
    c2 = tracker.consumed["rerank_docs"]
    assert c2 > c1
    
    # Negative consumption should be avoided/ignored by logic
    tracker.consume(CostObject(docs=-5))
    assert tracker.consumed["rerank_docs"] == 30

def test_d21_latency_budget():
    from unittest.mock import patch
    budget = CostBudget(limits={"latency_ms": 1000})
    tracker = CostTracker(budget, ControllerTrace())
    
    # Use patch directly on time.time used inside CostTracker methods
    with patch("ragtune.core.budget.time.time") as mock_time:
        # Start time was recorded at init (real time)
        # We need to know what that was.
        # Let's mock the start time too by patching time.time during tracker init.
        pass

    # Simplified test for logic:
    tracker._start_time = 1000.0
    with patch("ragtune.core.budget.time.time") as mock_time:
        mock_time.return_value = 1001.1 # 1100ms elapsed
        assert tracker.is_exhausted() is True
