"""Budget-exhaustion edge cases for component-scoped limits, through RAGtuneController."""

from typing import Dict, List

import pytest

from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.retrievers import InMemoryRetriever
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.core.budget import CostBudget
from ragtune.core.controller import RAGtuneController
from ragtune.core.interfaces import BaseEstimator, BaseReformulator, BaseReranker
from ragtune.core.types import EstimatorOutput, RAGtuneContext, ScoredDocument


class ClockReranker(BaseReranker):
    """Takes a fixed number of (fake) milliseconds per batch."""

    def __init__(self, clock, ms_per_batch: float = 0.0):
        self.clock, self.ms, self.batches = clock, ms_per_batch, 0

    def rerank(self, documents, context, strategy=None) -> Dict[str, float]:
        self.clock.advance_ms(self.ms)
        self.batches += 1
        return {d.doc_id: 1.0 for d in documents}


class ClockEstimator(BaseEstimator):
    def __init__(self, clock, ms_per_call: float = 0.0, wants_reformulation: bool = False):
        self.clock, self.ms, self.calls = clock, ms_per_call, 0
        self.wants_reformulation = wants_reformulation

    def value(self, pool, context) -> Dict[str, EstimatorOutput]:
        self.clock.advance_ms(self.ms)
        self.calls += 1
        return {it.doc_id: EstimatorOutput(priority=1.0 - it.initial_rank * 0.01) for it in pool.get_eligible()}

    def needs_reformulation(self, context, current_pool) -> bool:
        return self.wants_reformulation


class ListReformulator(BaseReformulator):
    def __init__(self, queries: List[str]):
        self.queries, self.calls = queries, 0

    def generate(self, context: RAGtuneContext) -> List[str]:
        self.calls += 1
        return list(self.queries)


DOCS = [ScoredDocument(id=f"d{i}", content=f"doc {i}", score=1.0 - i * 0.05) for i in range(10)]


def build(fake_clock, limits, rerank_ms=0.0, estimator=None, reformulator=None, batch_size=2):
    reranker = ClockReranker(fake_clock, rerank_ms)
    controller = RAGtuneController(
        retriever=InMemoryRetriever(DOCS),
        reformulator=reformulator or ListReformulator([]),
        reranker=reranker,
        assembler=GreedyAssembler(max_docs=10),
        scheduler=ActiveLearningScheduler(batch_size=batch_size),
        estimator=estimator or ClockEstimator(fake_clock),
        budget=CostBudget(limits=limits),
    )
    return controller, reranker


def actions(output) -> List[str]:
    return [e.action for e in output.trace.events]


def test_latency_only_rerank_budget_still_reranks(fake_clock):
    # Venky's example: budget only reranking, only latency. Without the
    # absent-limit-is-unlimited rule this reranked zero documents.
    controller, reranker = build(fake_clock, {"rerank.latency_ms": 1000}, rerank_ms=10)
    out = controller.run("doc")
    assert out.final_budget_state["rerank_docs"] == len(DOCS)
    assert out.final_budget_state["rerank.latency_ms"] == pytest.approx(10 * reranker.batches)


def test_rerank_latency_budget_stops_loop_after_crossing_batch(fake_clock):
    # 100ms per batch, 250ms budget: batches at 100 and 200 are under the limit,
    # the third crosses it, and no fourth batch starts.
    controller, reranker = build(fake_clock, {"rerank.latency_ms": 250}, rerank_ms=100)
    out = controller.run("doc")
    assert reranker.batches == 3
    assert out.final_budget_state["rerank_docs"] == 6
    assert "over_limit_rerank.latency_ms" in actions(out)


def test_scoped_rerank_docs_limit_is_exact(fake_clock):
    controller, reranker = build(fake_clock, {"rerank.docs": 5}, batch_size=2)
    out = controller.run("doc")
    assert out.final_budget_state["rerank.docs"] == 5  # batches of 2, 2, then 1
    assert reranker.batches == 3


def test_assembly_token_budget_only_limits_context(fake_clock):
    controller, _ = build(fake_clock, {"assembly.tokens": 5})
    out = controller.run("doc")
    assert out.final_budget_state["rerank_docs"] == len(DOCS)  # reranking unaffected
    assert sum(d.token_count for d in out.documents) <= 5
    assert 0 < len(out.documents) < len(DOCS)


def test_zero_reformulation_calls_skips_reformulator(fake_clock):
    reformulator = ListReformulator(["other query"])
    estimator = ClockEstimator(fake_clock, wants_reformulation=True)
    controller, _ = build(fake_clock, {"reformulation.calls": 0}, estimator=estimator, reformulator=reformulator)
    out = controller.run("doc")
    assert reformulator.calls == 0
    assert "reformulation_skipped" in actions(out)


def test_retrieval_calls_budget_skips_supplemental_retrieval(fake_clock):
    reformulator = ListReformulator(["alt one", "alt two"])
    estimator = ClockEstimator(fake_clock, wants_reformulation=True)
    controller, _ = build(fake_clock, {"retrieval.calls": 1}, estimator=estimator, reformulator=reformulator)
    out = controller.run("doc")
    assert out.final_budget_state["retrieval.calls"] == 1
    assert actions(out).count("retrieval_skipped") == 2


def test_exhausted_estimation_budget_keeps_reranking(fake_clock):
    estimator = ClockEstimator(fake_clock, ms_per_call=50)
    controller, reranker = build(fake_clock, {"estimation.latency_ms": 100}, estimator=estimator)
    out = controller.run("doc")
    assert estimator.calls == 2
    assert out.final_budget_state["rerank_docs"] == len(DOCS)
    assert actions(out).count("estimation_skipped") == 1


def test_final_state_reports_every_instrumented_stage(fake_clock):
    controller, _ = build(fake_clock, {})
    state = controller.run("doc").final_budget_state
    for stage in ("retrieval", "estimation", "rerank", "assembly"):
        assert f"{stage}.latency_ms" in state
