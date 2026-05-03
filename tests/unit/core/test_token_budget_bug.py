"""
Tests for token budget behavior in the RAGtune iterative loop.

These tests identify and document a bug where the token budget dimension
in CostTracker is not properly integrated into the iterative reranking loop:
tokens are only consumed during assembly (after the loop exits), not during
the loop itself via BatchProposal.expected_cost.

This means:
1. is_exhausted() checking tokens at loop entry is only meaningful if tokens=0
   at initialization (which prevents the loop entirely), not if tokens are
   exhausted during loop iterations.
2. A positive token budget does not constrain the number of reranking iterations.
3. Tokens only limit how many documents are returned during assembly, after all
   reranking work has already been performed.
"""
from typing import List
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
    """Reranker that tracks how many times it was called and what batch sizes, using instance state."""
    def __init__(self):
        self.call_count = 0
        self.batch_sizes = []

    def rerank(self, documents: list, context: RAGtuneContext, strategy: str = None) -> dict:
        self.call_count += 1
        self.batch_sizes.append(len(documents))
        return {doc.doc_id: doc.final_score() + 0.1 for doc in documents}


class CountingScheduler(BaseScheduler):
    """Scheduler that selects batches of fixed size until budget exhausted."""
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size

    def select_batch(self, pool: CandidatePool, budget: RemainingBudgetView) -> BatchProposal | None:
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
    Test that tokens=0 at initialization causes is_exhausted() to return True
    immediately and prevents the loop from running entirely.

    This is the ONLY scenario where the token budget currently affects loop entry.
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
        f"Expected 0 reranker calls with tokens=0, got {reranker.call_count}. "
        "Token budget of 0 prevents loop entry entirely."
    )
    # rerank_docs is only consumed during the loop, so it stays at 0
    assert output.final_budget_state.get("rerank_docs", 0) == 0
    # tokens are consumed during assembly even when loop doesn't run
    # because the token budget only gates assembly, not loop entry


def test_positive_token_budget_does_not_constrain_reranking_iterations():
    """
    Test that a positive but very small token budget (e.g., 3 tokens) does NOT
    prevent the reranking loop from running its full course based on rerank_docs.

    This demonstrates the core bug: tokens are not consumed during the loop,
    so is_exhausted() cannot return True on token depletion mid-loop.
    The token budget only affects assembly, not the number of reranking iterations.
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

    rerank_docs_consumed = output.final_budget_state["rerank_docs"]
    tokens_consumed = output.final_budget_state.get("tokens", 0)

    assert rerank_docs_consumed > 0, (
        f"Expected rerank_docs > 0 with positive token budget, got {rerank_docs_consumed}"
    )

    assert reranker.call_count >= 4, (
        f"Expected at least 4 reranker calls (20 docs / batch_size=4), "
        f"got {reranker.call_count}. "
        "The reranking loop ran despite a token budget of only 3 tokens, "
        "demonstrating that tokens are NOT consumed during the iterative loop."
    )

    assert tokens_consumed >= 3, (
        f"Token budget of 3 should be exhausted or nearly exhausted after assembly, "
        f"but tokens_consumed={tokens_consumed}. Tokens are consumed AFTER the loop "
        "during assembly, not during the loop itself."
    )


def test_token_budget_versus_rerank_docs_budget_independence():
    """
    Test that rerank_docs and tokens are independent budget dimensions during the loop.

    With rerank_docs=20 and tokens=3, the loop should consume all 20 rerank_docs
    budget without ever checking or consuming tokens mid-loop.
    """
    reranker = CountingReranker()

    controller = RAGtuneController(
        retriever=ManyDocRetriever(n_docs=20, tokens_per_doc=100),
        reformulator=IdentityReformulator(),
        reranker=reranker,
        assembler=GreedyAssembler(),
        scheduler=CountingScheduler(batch_size=5),
        estimator=TestEstimator(),
        budget=CostBudget(limits={
            "tokens": 3,
            "rerank_docs": 20,
            "rerank_calls": 10,
        }),
    )

    output = controller.run("test query")

    rerank_docs_consumed = output.final_budget_state["rerank_docs"]
    tokens_consumed = output.final_budget_state.get("tokens", 0)

    assert rerank_docs_consumed == 20, (
        f"Expected all 20 rerank_docs consumed, got {rerank_docs_consumed}. "
        "The rerank_docs budget fully constrained the loop iterations."
    )

    assert tokens_consumed >= 3, (
        f"Tokens only consumed during assembly (post-loop): tokens_consumed={tokens_consumed}. "
        "If this value is 0, it means assembly returned 0 documents due to token exhaustion. "
        "If this value > 0, it reflects documents that fit within the 3-token budget during assembly. "
        "Either way, tokens did not constrain the reranking loop itself."
    )


def test_tokens_consumed_only_during_assembly():
    """
    Test that token consumption occurs during GreedyAssembler.assemble(), not
    during the controller's iterative loop via tracker.consume(proposal.expected_cost).

    This verifies the root cause: BatchProposal.expected_cost carries tokens=0
    from all schedulers, so tracker.consume() never touches the token budget.
    """
    reranker = CountingReranker()

    controller = RAGtuneController(
        retriever=ManyDocRetriever(n_docs=10, tokens_per_doc=5),
        reformulator=IdentityReformulator(),
        reranker=reranker,
        assembler=GreedyAssembler(),
        scheduler=CountingScheduler(batch_size=3),
        estimator=TestEstimator(),
        budget=CostBudget(limits={
            "tokens": 50,
            "rerank_docs": 50,
            "rerank_calls": 10,
        }),
    )

    output = controller.run("test query")

    rerank_docs_consumed = output.final_budget_state["rerank_docs"]
    rerank_calls_consumed = output.final_budget_state.get("rerank_calls", 0)
    tokens_consumed = output.final_budget_state.get("tokens", 0)

    assert reranker.call_count == rerank_calls_consumed, (
        f"Reranker call count {reranker.call_count} should match "
        f"rerank_calls consumed {rerank_calls_consumed}"
    )

    assert rerank_calls_consumed > 0, "Reranking happened during the loop"

    assert tokens_consumed > 0, (
        f"Tokens consumed ({tokens_consumed}) must come from assembly, not from "
        "tracker.consume(proposal.expected_cost) because all schedulers create "
        "BatchProposal with expected_cost=CostObject(docs=..., calls=..., tokens=0)"
    )

    estimated_tokens_from_assembly = len(output.documents) * 5
    assert tokens_consumed <= estimated_tokens_from_assembly, (
        f"Tokens consumed ({tokens_consumed}) should be at most the assembly estimate "
        f"(docs returned: {len(output.documents)} x tokens_per_doc=5 = {estimated_tokens_from_assembly}), "
        "confirming tokens are consumed during assembly only."
    )


def test_assembly_token_filtering_with_low_token_budget():
    """
    Test that with a very low token budget, assembly filters documents but
    the reranking loop already completed its work.

    Scenario: Set tokens=1, rerank_docs=20. Loop runs and reranks all 20 docs
    (consuming all rerank_docs budget). Then assembly can only return documents
    whose estimated token count fits in 1 token (essentially 0 documents).
    The reranking work was done but the result is filtered to nothing.
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
            "rerank_docs": 10,
            "rerank_calls": 10,
        }),
    )

    output = controller.run("test query")

    rerank_docs_consumed = output.final_budget_state["rerank_docs"]

    assert rerank_docs_consumed == 10, (
        f"All rerank_docs budget (10) should be consumed: got {rerank_docs_consumed}. "
        "The loop ran to completion based on rerank_docs, not tokens."
    )

    assert reranker.call_count >= 2, (
        f"Reranking happened despite low token budget: "
        f"{reranker.call_count} calls, {rerank_docs_consumed} docs reranked."
    )

    assert len(output.documents) == 0, (
        f"Expected 0 documents returned (1 token budget insufficient for any doc at ~13 estimated tokens), "
        f"got {len(output.documents)}. Token budget filtered at assembly, but all reranking was already done."
    )


def test_batch_proposal_expected_cost_carries_no_tokens():
    """
    Test that scheduler.create_batch returns CostObject with tokens=0 by default.

    This is the root cause: if expected_cost.tokens were populated by schedulers
    and consumed during the loop, the token budget would properly constrain iterations.
    Currently CostObject(tokens=0) from schedulers means consume() skips token consumption.
    """
    from ragtune.components.schedulers import ActiveLearningScheduler, GracefulDegradationScheduler
    from ragtune.core.pool import CandidatePool

    pool = CandidatePool()
    for i in range(5):
        pool.add_items([
            ScoredDocument(id=f"doc_{i}", content=f"doc {i}", score=1.0 - i * 0.1)
        ], source="original")

    pool.apply_priorities({f"doc_{i}": 1.0 - i * 0.1 for i in range(5)})

    budget = RemainingBudgetView(remaining_tokens=100, remaining_rerank_docs=10, remaining_rerank_calls=5)

    scheduler1 = ActiveLearningScheduler(batch_size=2)
    proposal1 = scheduler1.select_batch(pool, budget)
    assert proposal1 is not None
    assert proposal1.expected_cost.tokens == 0, (
        f"ActiveLearningScheduler creates CostObject with tokens={proposal1.expected_cost.tokens}, "
        "not a per-batch token estimate. Tokens cannot be consumed during the loop."
    )
    assert proposal1.expected_cost.docs == 2
    assert proposal1.expected_cost.calls == 1

    scheduler2 = GracefulDegradationScheduler(batch_size=3)
    proposal2 = scheduler2.select_batch(pool, budget)
    if proposal2 is not None:
        assert proposal2.expected_cost.tokens == 0, (
            f"GracefulDegradationScheduler creates CostObject with tokens={proposal2.expected_cost.tokens}"
        )


def test_cost_tracker_consume_skips_zero_tokens():
    """
    Test that CostTracker.consume() does not accumulate tokens when cost.tokens=0.

    This confirms the mechanism: consume() only calls try_consume("tokens", ...) when
    cost.tokens > 0. Since schedulers always set tokens=0, tokens never accumulate.
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