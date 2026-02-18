import pytest
import asyncio
from typing import List, Dict, Optional
from ragtune.core.controller import RAGtuneController
from ragtune.core.types import RAGtuneContext, ScoredDocument, BatchProposal
from ragtune.core.pool import CandidatePool, PoolItem
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.interfaces import BaseRetriever, BaseReranker, BaseReformulator, BaseAssembler, BaseEstimator, BaseScheduler
from ragtune.utils.config import config

# --- MOCKS ---

class MockRetriever(BaseRetriever):
    def __init__(self, mapping: Dict[str, List[ScoredDocument]]):
        self.mapping = mapping
        self.call_count = 0
        self.last_top_k = None

    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        self.call_count += 1
        self.last_top_k = top_k
        return self.mapping.get(context.query, [])

class MockReranker(BaseReranker):
    def rerank(self, documents: List[PoolItem], context: RAGtuneContext, strategy: Optional[str] = None) -> Dict[str, float]:
        return {doc.doc_id: 1.0 for doc in documents}

class MockReformulator(BaseReformulator):
    def __init__(self, rewrites: List[str]):
        self.rewrites = rewrites
        self.call_count = 0

    def generate(self, context: RAGtuneContext) -> List[str]:
        self.call_count += 1
        if not context.tracker.try_consume_reformulation():
            return []
        return self.rewrites

class MockEstimator(BaseEstimator):
    def __init__(self, needs_ref: bool = True):
        self.needs_ref_flag = needs_ref
        self.call_count = 0

    def value(self, pool, context):
        return {it.doc_id: 1.0 for it in pool}

    def needs_reformulation(self, context, current_pool):
        self.call_count += 1
        return self.needs_ref_flag

class MockScheduler(BaseScheduler):
    def select_batch(self, pool, budget) -> Optional[BatchProposal]:
        # Just return None to stop loop after 1 iteration or immediately
        return None

class MockAssembler(BaseAssembler):
    def assemble(self, candidates, context):
        return [ScoredDocument(id=it.doc_id, content=it.content, score=it.final_score()) for it in candidates]

# --- TESTS ---

def test_e2e_01_no_rewrite_budget_fallback():
    """No rewrite budget -> original-only fallback."""
    retriever = MockRetriever({"original": [ScoredDocument(id="d1", content="c")]})
    reformulator = MockReformulator(["q1", "q2"])
    estimator = MockEstimator(needs_ref=True)
    
    budget = CostBudget(limits={"retrieval_calls": 5, "reformulations": 0})
    controller = RAGtuneController(
        retriever=retriever, reformulator=reformulator, estimator=estimator,
        reranker=MockReranker(), scheduler=MockScheduler(), assembler=MockAssembler()
    )
    controller.budget = budget
    
    output = controller.run("original")
    
    # Assertions
    assert reformulator.call_count == 1 
    assert retriever.call_count == 1
    
    trace_actions = [e.action for e in output.trace.events]
    assert "consume_retrieval_calls" in trace_actions
    assert "over_limit_reformulations" in trace_actions
    
    pool_init_event = next(e for e in output.trace.events if e.action == "pool_init")
    assert pool_init_event.details["count"] == 1
    assert pool_init_event.details["reformulations"] == []
    
    # Observability
    assert output.final_budget_state["retrieval_calls"] == 1
    assert output.final_budget_state["reformulations"] == 1 # consumed/attempted

def test_e2e_02_no_extra_retrieval_budget():
    """Rewrite budget available but no extra retrieval budget."""
    retriever = MockRetriever({
        "original": [ScoredDocument(id="d_orig", content="c")],
        "q1": [ScoredDocument(id="d_q1", content="c")]
    })
    reformulator = MockReformulator(["q1"])
    estimator = MockEstimator(needs_ref=True)
    
    budget = CostBudget(limits={"retrieval_calls": 1, "reformulations": 1})
    controller = RAGtuneController(
        retriever=retriever, reformulator=reformulator, estimator=estimator,
        reranker=MockReranker(), scheduler=MockScheduler(), assembler=MockAssembler()
    )
    controller.budget = budget
    
    output = controller.run("original")
    
    assert reformulator.call_count == 1
    assert retriever.call_count == 1
    
    trace_actions = [e.action for e in output.trace.events]
    assert "retrieval_skipped" in trace_actions
    
    # Observability
    assert output.final_budget_state["retrieval_calls"] == 2 # 1 consumed, 1 over limit attempted? 
    # Actually tracker records every attempt in consumed if no limit, but here limit is 1.
    # try_consume increments then checks. So 1+1=2.
    assert output.final_budget_state["reformulations"] == 1

def test_e2e_03_partial_supplemental_retrieval():
    """Partial supplemental retrieval allowed."""
    retriever = MockRetriever({
        "original": [ScoredDocument(id="d0", content="c")],
        "q1": [ScoredDocument(id="d1", content="c")],
        "q2": [ScoredDocument(id="d2", content="c")]
    })
    reformulator = MockReformulator(["q1", "q2", "q3"])
    
    budget = CostBudget(limits={"retrieval_calls": 2, "reformulations": 1})
    controller = RAGtuneController(
        retriever=retriever, reformulator=reformulator, estimator=MockEstimator(),
        reranker=MockReranker(), scheduler=MockScheduler(), assembler=MockAssembler()
    )
    controller.budget = budget
    
    output = controller.run("original")
    
    assert retriever.call_count == 2
    assert "retrieval_skipped" in [e.action for e in output.trace.events]
    
    # Observability
    assert output.final_budget_state["retrieval_calls"] == 4 # 1 (orig) + 3 (rewrites)
    assert output.final_budget_state["reformulations"] == 1

def test_e2e_04_rerank_budget_exhausted_early():
    """Rerank budget exhausted after rewrites."""
    retriever = MockRetriever({
        "original": [ScoredDocument(id="d0", content="c")],
        "q1": [ScoredDocument(id="d1", content="c")]
    })
    
    class LimitedScheduler(MockScheduler):
        def select_batch(self, pool, budget):
            if budget.remaining_rerank_docs > 0:
                # Propose more than allowed
                return BatchProposal(doc_ids=["d0"], strategy="cross_encoder", expected_cost={"docs": 100}, estimated_utility=1.0)
            return None

    budget = CostBudget(limits={"retrieval_calls": 5, "reformulations": 1, "rerank_docs": 50})
    controller = RAGtuneController(
        retriever=retriever, reformulator=MockReformulator(["q1"]), estimator=MockEstimator(),
        reranker=MockReranker(), scheduler=LimitedScheduler(), assembler=MockAssembler()
    )
    controller.budget = budget
    
    output = controller.run("original")
    
    assert retriever.call_count == 2
    assert "over_limit_rerank_docs" in [e.action for e in output.trace.events]
    assert output.final_budget_state["rerank_docs"] == 100

def test_e2e_05_estimator_skip_rewrites():
    """High-confidence estimator -> skip rewrites."""
    reformulator = MockReformulator(["q1"])
    estimator = MockEstimator(needs_ref=False)
    
    controller = RAGtuneController(
        retriever=MockRetriever({"original": []}), reformulator=reformulator, estimator=estimator,
        reranker=MockReranker(), scheduler=MockScheduler(), assembler=MockAssembler()
    )
    
    output = controller.run("original")
    
    assert reformulator.call_count == 0
    assert output.final_budget_state.get("reformulations", 0) == 0

def test_e2e_06_low_confidence_triggers_rewrites():
    """Low-confidence estimator -> rewrites triggered."""
    retriever = MockRetriever({
        "original": [ScoredDocument(id="junk", content="bad", score=0.1)],
        "q1": [ScoredDocument(id="gold", content="good", score=0.9)]
    })
    reformulator = MockReformulator(["q1"])
    
    # Custom estimator that signals needs_ref if only junk is present
    class SmartEstimator(MockEstimator):
        def needs_reformulation(self, context, current_pool):
            self.call_count += 1
            # If pool has only low scores, return True
            max_score = max([it.final_score() for it in current_pool], default=0)
            return max_score < 0.5

    controller = RAGtuneController(
        retriever=retriever, reformulator=reformulator, estimator=SmartEstimator(),
        reranker=MockReranker(), scheduler=MockScheduler(), assembler=MockAssembler()
    )
    
    output = controller.run("original")
    
    assert reformulator.call_count == 1
    pool_init_event = next(e for e in output.trace.events if e.action == "pool_init")
    assert pool_init_event.details["count"] == 2
    assert pool_init_event.details["metrics"]["rewrite_utility_ratio"] == 0.5

def test_e2e_07_misleading_high_confidence():
    """Misleading estimator (false high confidence) -> ensure system doesn't rewrite."""
    estimator = MockEstimator(needs_ref=False)
    reformulator = MockReformulator(["q1"])
    
    controller = RAGtuneController(
        retriever=MockRetriever({"original": [ScoredDocument(id="junk", content="bad")]}),
        reformulator=reformulator,
        estimator=estimator,
        reranker=MockReranker(),
        scheduler=MockScheduler(),
        assembler=MockAssembler()
    )
    
    output = controller.run("original")
    
    assert reformulator.call_count == 0
    assert "q1" not in [e.action for e in output.trace.events]

def test_e2e_08_estimator_disagreement_policy():
    """Estimator disagreement ensemble -> deterministic policy (Pessimistic: any True -> True)."""
    class EnsembleEstimator(MockEstimator):
        def __init__(self, signals):
            self.signals = signals
            self.call_count = 0
        def needs_reformulation(self, context, current_pool):
            self.call_count += 1
            return any(self.signals) # Or all(), depending on policy

    # One says Yes, one says No -> Final Yes
    controller = RAGtuneController(
        retriever=MockRetriever({"original": []}),
        reformulator=MockReformulator(["q1"]),
        estimator=EnsembleEstimator([True, False]),
        reranker=MockReranker(), scheduler=MockScheduler(), assembler=MockAssembler()
    )
    controller.run("original")
    assert controller.reformulator.call_count == 1

def test_e2e_09_budget_aware_estimator_choice():
    """Estimator wants rewrites, but budget low -> selects cheaper action (Rerank)."""
    # This specifically tests if the Controller/Estimator can see remaining budget
    # and "agree" to skip rewrite if it's too expensive.
    
    class BudgetSensitiveEstimator(MockEstimator):
        def needs_reformulation(self, context, current_pool):
            self.call_count += 1
            budget = context.tracker.remaining_view()
            # If we don't have many tokens or calls left, yield to reranking
            if budget.remaining_tokens < 1000:
                return False
            return True

    retriever = MockRetriever({"original": []})
    reformulator = MockReformulator(["q1"])
    
    # CASE A: Plenty of budget -> rewrite
    budget_high = CostBudget(limits={"tokens": 5000, "retrieval_calls": 5, "reformulations": 1})
    c1 = RAGtuneController(retriever=retriever, reformulator=reformulator, estimator=BudgetSensitiveEstimator(),
                           reranker=MockReranker(), scheduler=MockScheduler(), assembler=MockAssembler())
    c1.budget = budget_high
    c1.run("original")
    assert reformulator.call_count == 1
    
    # CASE B: Low token budget -> skip rewrite (save for rerank)
    reformulator.call_count = 0
    budget_low = CostBudget(limits={"tokens": 500, "retrieval_calls": 5, "reformulations": 1})
    c2 = RAGtuneController(retriever=retriever, reformulator=reformulator, estimator=BudgetSensitiveEstimator(),
                           reranker=MockReranker(), scheduler=MockScheduler(), assembler=MockAssembler())
    c2.budget = budget_low
    c2.run("original")
    assert reformulator.call_count == 0

def test_e2e_10_rewrite_dedup_bounded_spend():
    """Rewrite output contains duplicates -> wasted spend is bounded."""
    # LLM returns [q1, q1, original]
    reformulator = MockReformulator(["q1", "q1", "original"])
    retriever = MockRetriever({"original": [], "q1": []})
    
    controller = RAGtuneController(
        retriever=retriever, reformulator=reformulator, estimator=MockEstimator(),
        reranker=MockReranker(), scheduler=MockScheduler(), assembler=MockAssembler()
    )
    
    output = controller.run("original")
    
    # Retrieval calls: 1 (original) + 1 (q1) = 2
    # The second q1 and the 'original' rewrite should be skipped by Controller's seen_queries logic.
    assert retriever.call_count == 2

def test_e2e_11_off_topic_rewrite_noise():
    """One rewrite is off-topic -> noise handled."""
    retriever = MockRetriever({
        "original": [ScoredDocument(id="d1", content="c1", score=0.8)],
        "junk_q": [ScoredDocument(id=f"noise_{i}", content="noise", score=0.1) for i in range(5)]
    })
    
    controller = RAGtuneController(
        retriever=retriever, reformulator=MockReformulator(["junk_q"]), estimator=MockEstimator(),
        reranker=MockReranker(), scheduler=MockScheduler(), assembler=MockAssembler()
    )
    
    output = controller.run("original")
    
    pool_init_event = next(e for e in output.trace.events if e.action == "pool_init")
    assert pool_init_event.details["metrics"]["overlap_count"] == 0
    assert pool_init_event.details["metrics"]["rewrite_only_count"] == 5

