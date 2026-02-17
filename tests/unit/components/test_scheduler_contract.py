import pytest
from ragtune.core.pool import CandidatePool, PoolItem, ItemState
from ragtune.core.types import RemainingBudgetView, BatchProposal, CostObject
from ragtune.components.schedulers import ActiveLearningScheduler

def test_c13_selects_subset_of_eligible():
    items = [
        PoolItem(doc_id="e1", state=ItemState.CANDIDATE, content="", initial_rank=1),
        PoolItem(doc_id="r1", state=ItemState.RERANKED, content=""),
        PoolItem(doc_id="f1", state=ItemState.IN_FLIGHT, content=""),
        PoolItem(doc_id="d1", state=ItemState.DROPPED, content=""),
        PoolItem(doc_id="e2", state=ItemState.CANDIDATE, content="", initial_rank=2)
    ]
    pool = CandidatePool(items)
    budget = RemainingBudgetView(
        remaining_tokens=1000, remaining_rerank_docs=10, 
        remaining_rerank_calls=10, assembly_token_buffer=0
    )
    scheduler = ActiveLearningScheduler(batch_size=5)
    
    proposal = scheduler.select_batch(pool, budget)
    assert proposal is not None
    assert set(proposal.doc_ids).issubset({"e1", "e2"})

def test_c14_no_batch_if_no_eligible():
    items = [PoolItem(doc_id="r1", state=ItemState.RERANKED, content="")]
    pool = CandidatePool(items)
    budget = RemainingBudgetView(
        remaining_tokens=1000, remaining_rerank_docs=10, 
        remaining_rerank_calls=10, assembly_token_buffer=0
    )
    scheduler = ActiveLearningScheduler(batch_size=5)
    
    assert scheduler.select_batch(pool, budget) is None

def test_c15_stable_tie_breaking():
    # Both identical priority. Should sort by initial_rank.
    items = [
        PoolItem(doc_id="z", initial_rank=10, content="", priority_value=0.5),
        PoolItem(doc_id="a", initial_rank=5, content="", priority_value=0.5)
    ]
    pool = CandidatePool(items)
    budget = RemainingBudgetView(
        remaining_tokens=1000, remaining_rerank_docs=10, 
        remaining_rerank_calls=10, assembly_token_buffer=0
    )
    scheduler = ActiveLearningScheduler(batch_size=5)
    
    proposal = scheduler.select_batch(pool, budget)
    assert proposal.doc_ids == ["a", "z"] # 5 before 10

def test_c16_budget_aware_batching():
    items = [PoolItem(doc_id=str(i), content="") for i in range(10)]
    pool = CandidatePool(items)
    # Only 3 doc budget left
    budget = RemainingBudgetView(
        remaining_tokens=1000, remaining_rerank_docs=3, 
        remaining_rerank_calls=10, assembly_token_buffer=0
    )
    scheduler = ActiveLearningScheduler(batch_size=5)
    
    proposal = scheduler.select_batch(pool, budget)
    assert len(proposal.doc_ids) == 3
    assert proposal.expected_cost.docs == 3

def test_c17_strategy_escalation_logic():
    # Gap < 0.05 → LLM
    items = [
        PoolItem(doc_id="d1", content="", priority_value=0.9, initial_rank=1),
        PoolItem(doc_id="d2", content="", priority_value=0.88, initial_rank=2)
    ]
    pool = CandidatePool(items)
    budget = RemainingBudgetView(
        remaining_tokens=1000, remaining_rerank_docs=10, 
        remaining_rerank_calls=10, assembly_token_buffer=0
    )
    scheduler = ActiveLearningScheduler(batch_size=5)
    
    prop = scheduler.select_batch(pool, budget)
    assert prop.strategy == "llm"
    
    # Gap > 0.05 → cross_encoder (default)
    pool.get_items(["d2"])[0].priority_value = 0.8
    prop2 = scheduler.select_batch(pool, budget)
    assert prop2.strategy == "cross_encoder"
