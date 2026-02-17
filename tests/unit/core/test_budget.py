import pytest
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.types import ControllerTrace

def test_budget_token_consumption():
    trace = ControllerTrace()
    budget = CostBudget.simple(tokens=25)
    tracker = CostTracker(budget, trace)
    
    assert tracker.try_consume_tokens(10) is True
    assert tracker.try_consume_tokens(10) is True
    assert tracker.try_consume_tokens(10) is False # Total 30 > 25

def test_budget_rerank_consumption():
    trace = ControllerTrace()
    budget = CostBudget.simple(docs=5)
    tracker = CostTracker(budget, trace)
    
    assert tracker.try_consume_rerank(3) is True
    assert tracker.try_consume_rerank(3) is False # Total 6 > 5

def test_budget_reformulation_consumption():
    trace = ControllerTrace()
    budget = CostBudget.simple(reformulations=1)
    tracker = CostTracker(budget, trace)
    
    assert tracker.try_consume_reformulation() is True
    assert tracker.try_consume_reformulation() is False

def test_custom_cost_type():
    trace = ControllerTrace()
    budget = CostBudget(limits={"usd": 0.50})
    tracker = CostTracker(budget, trace)

    assert tracker.try_consume("usd", 0.20) is True
    assert tracker.try_consume("usd", 0.40) is False
    assert tracker.consumed["usd"] == pytest.approx(0.60)
