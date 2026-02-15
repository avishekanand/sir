import pytest
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.types import ControllerTrace

def test_budget_token_consumption():
    trace = ControllerTrace()
    budget = CostBudget(max_tokens=25)
    tracker = CostTracker(budget, trace)
    
    assert tracker.try_consume_tokens(10) is True
    assert tracker.try_consume_tokens(10) is True
    assert tracker.try_consume_tokens(10) is False # Total 30 > 25

def test_budget_rerank_consumption():
    trace = ControllerTrace()
    budget = CostBudget(max_reranker_docs=5)
    tracker = CostTracker(budget, trace)
    
    assert tracker.try_consume_rerank(3) is True
    assert tracker.try_consume_rerank(3) is False # Total 6 > 5

def test_budget_reformulation_consumption():
    trace = ControllerTrace()
    budget = CostBudget(max_reformulations=1)
    tracker = CostTracker(budget, trace)
    
    assert tracker.try_consume_reformulation() is True
    assert tracker.try_consume_reformulation() is False
