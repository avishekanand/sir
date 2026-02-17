import pytest
from typing import List, Dict
from ragtune.core.controller import RAGtuneController
from ragtune.core.pool import PoolItem, ItemState
from ragtune.core.types import ScoredDocument, RAGtuneContext, BatchProposal, CostObject
from ragtune.core.interfaces import BaseRetriever, BaseReformulator, BaseReranker, BaseScheduler, BaseEstimator, BaseAssembler
from ragtune.core.budget import CostBudget
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import GracefulDegradationScheduler
from ragtune.components.estimators import BaselineEstimator

class DeterministicRetriever(BaseRetriever):
    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        # Return docs where 'match' in content is better
        return [
            ScoredDocument(id="d1", content="bad", score=0.9),
            ScoredDocument(id="d2", content="match", score=0.8),
            ScoredDocument(id="d3", content="bad", score=0.7),
            ScoredDocument(id="d4", content="match", score=0.6),
            ScoredDocument(id="d5", content="bad", score=0.5)
        ]

class IdentityReformulator(BaseReformulator):
    def generate(self, context: RAGtuneContext) -> List[str]:
        return [context.query]

def test_i1_rerank_improves_ordering():
    controller = RAGtuneController(
        retriever=DeterministicRetriever(),
        reformulator=IdentityReformulator(),
        reranker=SimulatedReranker(), # Simple 'match' in content → high score reranker
        assembler=GreedyAssembler(),
        scheduler=GracefulDegradationScheduler(batch_size=5, llm_limit=5),
        estimator=BaselineEstimator(),
        budget=CostBudget.simple(docs=10)
    )
    
    # query is 'match'
    output = controller.run("match")
    
    # Original: d1(0.9), d2(0.8), d3(0.7), d4(0.6), d5(0.5)
    # Reranked: d2(0.95), d4(0.95), d1(0.3), d3(0.3), d5(0.3)
    # Expected top-2: d2, d4
    doc_ids = [d.id for d in output.documents[:2]]
    assert "d2" in doc_ids
    assert "d4" in doc_ids
    assert output.documents[0].score == 0.95

def test_i2_budget_tradeoff():
    # Scenario A: No reranking budget
    controller_no_budget = RAGtuneController(
        retriever=DeterministicRetriever(),
        reformulator=IdentityReformulator(),
        reranker=SimulatedReranker(),
        assembler=GreedyAssembler(),
        scheduler=GracefulDegradationScheduler(batch_size=5, llm_limit=5),
        estimator=BaselineEstimator(),
        budget=CostBudget.simple(docs=0)
    )
    out_a = controller_no_budget.run("match")
    assert out_a.documents[0].id == "d1" # Original top
    assert out_a.final_budget_state.get("rerank_docs", 0) == 0
    
    # Scenario B: Budget for 5
    controller_with_budget = RAGtuneController(
        retriever=DeterministicRetriever(),
        reformulator=IdentityReformulator(),
        reranker=SimulatedReranker(),
        assembler=GreedyAssembler(),
        scheduler=GracefulDegradationScheduler(batch_size=5, llm_limit=5),
        estimator=BaselineEstimator(),
        budget=CostBudget.simple(docs=5)
    )
    out_b = controller_with_budget.run("match")
    assert out_b.documents[0].id == "d2" # Reranked top
    assert out_b.final_budget_state["rerank_docs"] == 5

def test_i3_union_provenance():
    class MultiSourceRetriever(BaseRetriever):
        def retrieve(self, context, top_k):
            if context.query == "q1":
                return [ScoredDocument(id="shared", content="c", score=0.9), ScoredDocument(id="unique1", content="c", score=0.8)]
            return [ScoredDocument(id="shared", content="c", score=0.5), ScoredDocument(id="unique2", content="c", score=0.4)]

    # We need a reformulator that returns 2 queries
    class MultiQueryReformulator(BaseReformulator):
        def generate(self, context): return ["q1", "q2"]

    controller = RAGtuneController(
        retriever=MultiSourceRetriever(),
        reformulator=MultiQueryReformulator(),
        reranker=SimulatedReranker(),
        assembler=GreedyAssembler(),
        scheduler=GracefulDegradationScheduler(batch_size=10, llm_limit=10),
        estimator=BaselineEstimator(),
        budget=CostBudget.simple(docs=10)
    )
    
    # We need to manually debug if CandidatePool dedups.
    # The pool initialization in run() should handle deduping and source merging.
    output = controller.run("unused")
    
    # Expected doc_ids: shared, unique1, unique2
    doc_ids = {d.id for d in output.documents}
    assert len(doc_ids) == 3
    assert "shared" in doc_ids

def test_i4_determinism_under_ties():
    # Same query twice should yield same results
    controller = RAGtuneController(
        retriever=DeterministicRetriever(),
        reformulator=IdentityReformulator(),
        reranker=SimulatedReranker(),
        assembler=GreedyAssembler(),
        scheduler=GracefulDegradationScheduler(batch_size=5, llm_limit=5),
        estimator=BaselineEstimator(),
        budget=CostBudget.simple(docs=5)
    )
    out1 = controller.run("match")
    out2 = controller.run("match")
    ids1 = [d.id for d in out1.documents]
    ids2 = [d.id for d in out2.documents]
    assert ids1 == ids2
