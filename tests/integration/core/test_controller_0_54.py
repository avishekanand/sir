import pytest
from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.core.types import ScoredDocument, RAGtuneContext
from ragtune.core.interfaces import BaseRetriever, BaseReformulator
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import GracefulDegradationScheduler
from ragtune.components.estimators import BaselineEstimator
from typing import List

class MockRetriever(BaseRetriever):
    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        return [
            ScoredDocument(id="d1", content="This matches query", score=0.5),
            ScoredDocument(id="d2", content="No match here", score=0.4),
            ScoredDocument(id="d3", content="Another match for query", score=0.3)
        ]

class IdentityReformulator(BaseReformulator):
    def generate(self, context: RAGtuneContext) -> List[str]:
        return [context.query]

def test_controller_iterative_loop():
    controller = RAGtuneController(
        retriever=MockRetriever(),
        reformulator=IdentityReformulator(),
        reranker=SimulatedReranker(),
        assembler=GreedyAssembler(),
        scheduler=GracefulDegradationScheduler(llm_limit=2, batch_size=2),
        estimator=BaselineEstimator(),
        budget=CostBudget.simple(docs=10)
    )
    
    output = controller.run("query")
    
    # We expect d1 and d3 (matches) and d2 (non-match)
    # Scheduler should have picked d1, d2 first (top scores)
    # LLM Limit is 2, so [d1, d2] get reranked.
    # SimulatedReranker gives 0.95 to d1 (match), 0.3 to d2 (no match)
    # d3 stayed at 0.3 (original score)
    
    doc_ids = [d.id for d in output.documents[:2]]
    assert "d1" in doc_ids 
    assert "d3" in doc_ids # Both d1 and d3 matched and were reranked to 1000.95
    
    assert output.documents[0].score == 1000.95
    assert output.documents[1].score == 1000.95
    assert output.documents[2].score == 1000.3 # d2 also reranked (1000.3)
    
    # Check trace
    rerank_events = [e for e in output.trace.events if e.action == "rerank_batch"]
    assert len(rerank_events) == 2
    
    # Batch 1: LLM
    assert rerank_events[0].details["count"] == 2
    assert set(rerank_events[0].details["doc_ids"]) == {"d1", "d2"}
    assert rerank_events[0].details["strategy"] == "llm"
    
    # Batch 2: Cross-Encoder (Degradation)
    assert rerank_events[1].details["count"] == 1
    assert rerank_events[1].details["doc_ids"] == ["d3"]
    assert rerank_events[1].details["strategy"] == "cross_encoder"
