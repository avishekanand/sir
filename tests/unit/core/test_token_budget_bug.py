"""
Tests for token budget misintegration in the RAGtune iterative loop (refs #1).

These tests verify that the token budget dimension is properly integrated into
the iterative reranking loop. Tests that assert desired behavior are marked
xfail(strict=True) because the current implementation consumes tokens only
during assembly, not during the iterative loop itself. When the bug is fixed,
these tests will start passing (XPASS), signalling that the xfail markers
should be removed.
"""
import pytest
from typing import List, Optional
from ragtune.core.controller import RAGtuneController
from ragtune.core.types import ScoredDocument, RAGtuneContext, CostObject, EstimatorOutput
from ragtune.core.interfaces import BaseRetriever, BaseReformulator, BaseReranker, BaseScheduler, BaseEstimator
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.pool import CandidatePool
from ragtune.core.types import ControllerTrace, BatchProposal, RemainingBudgetView
from ragtune.components.assemblers import GreedyAssembler


class ManyDocRetriever(BaseRetriever):
    """Retriever that returns N documents, each with estimated token_count in metadata."""
    def __init__(self, n_docs: int = 20, tokens_per_doc: int = 20):
        self.n_docs = n_docs
        self.tokens_per_doc = tokens_per_doc

    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        return [
            ScoredDocument(
                id=f"doc_{i}",
                content=f"Document {i} about machine learning topics",
                score=1.0 - (i * 0.05),
                metadata={"token_count": self.tokens_per_doc}
            )
            for i in range(self.n_docs)
        ]


class IdentityReformulator(BaseReformulator):
    def generate(self, context: RAGtuneContext) -> List[str]:
        context.tracker.try_consume_reformulation()
        return []


class CountingReranker(BaseReranker):
    """Reranker that tracks how many times it was called, using instance state."""
    def __init__(self):
        self.call_count = 0
        self.batch_sizes: List[int] = []

    def rerank(self, documents: List[ScoredDocument], context: RAGtuneContext, strategy: Optional[str] = None) -> dict:
        self.call_count += 1
        self.batch_sizes.append(len(documents))
        return {doc.doc_id: doc.final_score() + 0.1 for doc in documents}


class CountingScheduler(BaseScheduler):
    """Scheduler that selects batches of fixed size until budget exhausted."""
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size

    def select_batch(self, pool: CandidatePool, budget: RemainingBudgetView) -> Optional[BatchProposal]:
        eligible = pool.get_eligible()
        if not eligible or budget.remaining_rerank_docs <= 0:
            return None
        batch = eligible[: min(self.batch_size, budget.remaining_rerank_docs, len(eligible))]
        return BatchProposal(
            doc_ids=[item.doc_id for item in batch],
            strategy="noop",
            expected_cost=CostObject(docs=len(batch), calls=1)
        )


class TestEstimator(BaseEstimator):
    """Simple baseline estimator for testing."""
    def value(self, pool: CandidatePool, context: RAGtuneContext) -> dict:
        return {
            item.doc_id: EstimatorOutput(priority=max(item.sources.values()))
            for item in pool.get_eligible()
        }


def test_token_budget_zero_prevents_all_reranking():
    """
    Test that tokens=0 at initialization prevents the loop from running entirely.

    This is the correct current behavior: is_exhausted() returns True immediately
    when tokens=0, and the loop never executes.
    """
    reranker = CountingReranker()

    controller = RAGtuneController(
        retriever=ManyDocRetriever(n_docs=10),
        reformulator=IdentityReformulator(),
        reranker=reranker,
        assembler=GreedyAssembler(),
        scheduler=CountingScheduler(batch_size=5),
        estimator=TestEstimator(),
        budget=CostBudget(limits={
            "tokens": 0,
            "rerank_docs": 50,
            "rerank_calls": 10,
        }),
    )

    output = controller.run("test query")

    assert reranker.call_count == 0, (
        f"Expected 0 reranker calls with tokens=0, got {reranker.call_count}"
    )
    assert output.final_budget_state.get("rerank_docs", 0) == 0


@pytest.mark.xfail(
    strict=True,
    reason="Token budget not consumed during iterative loop (refs #1). "
           "Currently tokens=3 does not limit reranking because tokens "
           "are only consumed post-loop during assembly."
)
def test_positive_token_budget_limits_reranking_iterations():
    """
    Test that a positive token budget constrains the number of reranking iterations.

    With tokens=3, rerank_docs=20, and documents costing ~10 tokens each, the loop
    should exhaust the token budget before consuming all 20 rerank_docs. Currently
    the loop ignores the token budget and consumes all rerank_docs.

    When this bug is fixed, rerank_docs_consumed will be strictly less than the
    configured rerank_docs limit because token exhaustion will stop the loop early.
    """
    reranker = CountingReranker()

    controller = RAGtuneController(
        retriever=ManyDocRetriever(n_docs=20, tokens_per_doc=10),
        reformulator=IdentityReformulator(),
        reranker=reranker,
        assembler=GreedyAssembler(),
        scheduler=CountingScheduler(batch_size=4),
        estimator=TestEstimator(),
        budget=CostBudget(limits={
            "tokens": 3,
            "rerank_docs": 20,
            "rerank_calls": 10,
        }),
    )

    output = controller.run("test query")

    rerank_docs_limit = 20
    rerank_docs_consumed = output.final_budget_state["rerank_docs"]

    assert rerank_docs_consumed < rerank_docs_limit, (
        f"Token budget of 3 should prevent consuming all {rerank_docs_limit} "
        f"rerank_docs, but {rerank_docs_consumed} were consumed. "
        "Tokens are ignored during the loop (bug #1)."
    )


@pytest.mark.xfail(
    strict=True,
    reason="Token budget not consumed during iterative loop (refs #1). "
           "The token budget should constrain reranking independently of "
           "rerank_docs, but currently tokens and rerank_docs are decoupled."
)
def test_token_budget_constrains_reranking_before_docs_exhausted():
    """
    Test that the token budget stops reranking even when rerank_docs remain.

    With tokens=3 and rerank_docs=50, the loop should halt due to token exhaustion,
    not due to rerank_docs exhaustion. Currently all 50 rerank_docs are consumed
    because the token check in is_exhausted() is ineffective for positive values.
    """
    reranker = CountingReranker()

    controller = RAGtuneController(
        retriever=ManyDocRetriever(n_docs=50, tokens_per_doc=100),
        reformulator=IdentityReformulator(),
        reranker=reranker,
        assembler=GreedyAssembler(),
        scheduler=CountingScheduler(batch_size=5),
        estimator=TestEstimator(),
        budget=CostBudget(limits={
            "tokens": 3,
            "rerank_docs": 50,
            "rerank_calls": 20,
        }),
    )

    output = controller.run("test query")

    rerank_docs_limit = 50
    rerank_docs_consumed = output.final_budget_state["rerank_docs"]

    assert rerank_docs_consumed < rerank_docs_limit, (
        f"Token budget of 3 should limit reranking before consuming all "
        f"{rerank_docs_limit} rerank_docs, but {rerank_docs_consumed} were consumed. "
        "Token budget does not constrain the loop (bug #1)."
    )


@pytest.mark.xfail(
    strict=True,
    reason="Token budget not consumed during iterative loop (refs #1). "
           "Tokens should accumulate via tracker.consume() during the loop, "
           "not just via try_consume_tokens() during assembly."
)
def test_tokens_consumed_during_loop_not_just_assembly():
    """
    Test that token consumption reflects loop activity, not just assembly activity.

    Currently tokens_consumed == len(documents_returned) * token_count, which means
    tokens are only consumed by GreedyAssembler.assemble() after the loop exits.
    After the fix, tokens should also reflect per-batch consumption during the loop,
    so tokens_consumed should exceed the assembly-only estimate.
    """
    reranker = CountingReranker()

    controller = RAGtuneController(
        retriever=ManyDocRetriever(n_docs=10, tokens_per_doc=50),
        reformulator=IdentityReformulator(),
        reranker=reranker,
        assembler=GreedyAssembler(),
        scheduler=CountingScheduler(batch_size=3),
        estimator=TestEstimator(),
        budget=CostBudget(limits={
            "tokens": 1000,
            "rerank_docs": 50,
            "rerank_calls": 10,
        }),
    )

    output = controller.run("test query")

    rerank_calls_consumed = output.final_budget_state.get("rerank_calls", 0)
    tokens_consumed = output.final_budget_state.get("tokens", 0)

    assembly_only_token_estimate = len(output.documents) * 50

    assert rerank_calls_consumed > 0, "Reranking happened during the loop"

    assert tokens_consumed > assembly_only_token_estimate, (
        f"Tokens consumed ({tokens_consumed}) should exceed the assembly-only "
        f"estimate ({assembly_only_token_estimate}) because per-batch token costs "
        "should accumulate during the loop via tracker.consume(proposal.expected_cost). "
        "Currently tokens are consumed only during assembly (bug #1)."
    )


@pytest.mark.xfail(
    strict=True,
    reason="Token budget not consumed during iterative loop (refs #1). "
           "The loop should not perform reranking work that cannot produce "
           "useful output due to token exhaustion."
)
def test_low_token_budget_prevents_wasted_reranking():
    """
    Test that a token budget insufficient for any document prevents reranking.

    With tokens=1 and documents costing 10 tokens each, no document can fit in
    the token budget during assembly. A properly integrated token budget should
    prevent the loop from reranking documents that cannot be returned. Currently
    all documents are reranked and then assembly returns zero results, wasting
    the reranking computation.
    """
    reranker = CountingReranker()

    controller = RAGtuneController(
        retriever=ManyDocRetriever(n_docs=10, tokens_per_doc=10),
        reformulator=IdentityReformulator(),
        reranker=reranker,
        assembler=GreedyAssembler(),
        scheduler=CountingScheduler(batch_size=5),
        estimator=TestEstimator(),
        budget=CostBudget(limits={
            "tokens": 1,
            "rerank_docs": 50,
            "rerank_calls": 10,
        }),
    )

    output = controller.run("test query")

    rerank_docs_consumed = output.final_budget_state["rerank_docs"]

    assert rerank_docs_consumed == 0, (
        f"Token budget of 1 should prevent all reranking because no document "
        f"can fit in 1 token, but {rerank_docs_consumed} docs were reranked. "
        "Reranking work was performed despite being guaranteed to produce no "
        "useful output (bug #1)."
    )


@pytest.mark.xfail(
    strict=True,
    reason="Schedulers do not populate per-batch token estimates (refs #1). "
           "BatchProposal.expected_cost.tokens should reflect estimated token "
           "consumption for the selected batch to allow token-aware scheduling."
)
def test_schedulers_provide_per_batch_token_estimates():
    """
    Test that schedulers populate expected_cost.tokens with per-batch estimates.

    Currently all schedulers create CostObject with tokens=0, which prevents
    token consumption during tracker.consume(proposal.expected_cost). After the
    fix, schedulers should compute estimated token costs based on document lengths
    and strategy-specific prompt overhead.
    """
    from ragtune.components.schedulers import ActiveLearningScheduler
    from ragtune.core.pool import CandidatePool

    pool = CandidatePool()
    for i in range(5):
        pool.add_items([
            ScoredDocument(id=f"doc_{i}", content=f"doc content {i}", score=1.0 - i * 0.1)
        ], source="original")
    pool.apply_priorities({f"doc_{i}": 1.0 - i * 0.1 for i in range(5)})

    budget = RemainingBudgetView(remaining_tokens=100, remaining_rerank_docs=10, remaining_rerank_calls=5)
    scheduler = ActiveLearningScheduler(batch_size=2)
    proposal = scheduler.select_batch(pool, budget)
    assert proposal is not None

    assert proposal.expected_cost.tokens > 0, (
        f"Schedulers should estimate per-batch token cost, but "
        f"expected_cost.tokens={proposal.expected_cost.tokens}. "
        "Token estimation in schedulers is required for budget-aware "
        "loop control (refs #1)."
    )


def test_cost_tracker_consume_skips_zero_tokens():
    """
    Unit test verifying that CostTracker.consume() does not accumulate a cost
    dimension when the corresponding CostObject field is zero.

    This is correct behavior: consume() checks cost.tokens > 0 before calling
    try_consume('tokens', ...). When token estimation is added to schedulers,
    the non-zero tokens field will flow through consume() correctly.
    """
    budget = CostBudget(limits={"tokens": 100, "rerank_docs": 50, "rerank_calls": 10})
    tracker = CostTracker(budget, ControllerTrace())

    cost_with_tokens = CostObject(tokens=10, docs=0, calls=0)
    tracker.consume(cost_with_tokens)
    assert tracker.consumed.get("tokens", 0) == 10

    cost_without_tokens = CostObject(tokens=0, docs=5, calls=1)
    tracker.consume(cost_without_tokens)
    assert tracker.consumed.get("tokens", 0) == 10, (
        "Tokens should not change when cost.tokens=0. "
        "consume() only calls try_consume('tokens', ...) when cost.tokens > 0."
    )
    assert tracker.consumed["rerank_docs"] == 5
    assert tracker.consumed["rerank_calls"] == 1