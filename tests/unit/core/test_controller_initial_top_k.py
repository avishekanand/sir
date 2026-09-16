import pytest
from typing import List, Dict, Optional

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.core.pool import CandidatePool, PoolItem
from ragtune.core.types import (
    ScoredDocument, RAGtuneContext, BatchProposal,
    CostObject, RemainingBudgetView, EstimatorOutput,
)
from ragtune.core.interfaces import (
    BaseRetriever, BaseReformulator, BaseReranker,
    BaseScheduler, BaseEstimator, BaseAssembler,
)
from ragtune.utils.config import config


# --- Fakes ---

class CapturingRetriever(BaseRetriever):
    """Records every top_k value it is called with."""
    def __init__(self):
        self.calls: List[int] = []

    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        self.calls.append(top_k)
        return [ScoredDocument(id=f"d{i}", content=f"c{i}", score=0.1) for i in range(top_k)]


class NoOpReformulator(BaseReformulator):
    def generate(self, context: RAGtuneContext) -> List[str]:
        return []


class NoOpEstimator(BaseEstimator):
    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, EstimatorOutput]:
        return {it.doc_id: EstimatorOutput(priority=0.5) for it in pool.get_eligible()}


class NoOpScheduler(BaseScheduler):
    def select_batch(self, pool: CandidatePool, budget: RemainingBudgetView) -> Optional[BatchProposal]:
        return None


class NoOpReranker(BaseReranker):
    def rerank(self, documents: List[PoolItem], context: RAGtuneContext, strategy=None) -> Dict[str, float]:
        return {doc.doc_id: 0.5 for doc in documents}


class NoOpAssembler(BaseAssembler):
    def assemble(self, candidates: List[PoolItem], context: RAGtuneContext) -> List[ScoredDocument]:
        return [ScoredDocument(id=it.doc_id, content=it.content, score=it.final_score()) for it in candidates]


def _make_controller(retriever, initial_top_k=None):
    return RAGtuneController(
        retriever=retriever,
        reformulator=NoOpReformulator(),
        reranker=NoOpReranker(),
        assembler=NoOpAssembler(),
        scheduler=NoOpScheduler(),
        estimator=NoOpEstimator(),
        budget=CostBudget.simple(docs=0),
        initial_top_k=initial_top_k,
    )


# --- Tests ---

def test_initial_top_k_is_passed_to_retriever():
    retriever = CapturingRetriever()
    controller = _make_controller(retriever, initial_top_k=30)
    controller.run("query")
    assert retriever.calls[0] == 30


def test_initial_top_k_none_falls_back_to_config_default():
    # Config default for original_query_depth is 10 (set in defaults.yaml).
    retriever = CapturingRetriever()
    controller = _make_controller(retriever, initial_top_k=None)
    controller.run("query")
    expected = config.get("retrieval.original_query_depth", 10)
    assert retriever.calls[0] == expected


def test_initial_top_k_overrides_config():
    # Even if config has been mutated, initial_top_k takes precedence.
    original = config.get("retrieval.original_query_depth", 10)
    config.set("retrieval.original_query_depth", 99)
    try:
        retriever = CapturingRetriever()
        controller = _make_controller(retriever, initial_top_k=7)
        controller.run("query")
        assert retriever.calls[0] == 7
    finally:
        config.set("retrieval.original_query_depth", original)


def test_initial_top_k_zero_is_not_treated_as_none():
    # 0 is a valid (if degenerate) value — must not fall back to config.
    retriever = CapturingRetriever()
    controller = _make_controller(retriever, initial_top_k=0)
    controller.run("query")
    assert retriever.calls[0] == 0


def test_default_initial_top_k_is_none():
    controller = RAGtuneController(
        retriever=CapturingRetriever(),
        reformulator=NoOpReformulator(),
        reranker=NoOpReranker(),
        assembler=NoOpAssembler(),
        scheduler=NoOpScheduler(),
        estimator=NoOpEstimator(),
    )
    assert controller.initial_top_k is None
