import pytest
from ragtune.adapters.langchain import RAGtuneLangChainAdapter
from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import GracefulDegradationScheduler
from ragtune.components.estimators import BaselineEstimator
from ragtune.components.reformulators import IdentityReformulator
from typing import List
from ragtune.core.types import ScoredDocument

class FakeRetriever:
    def retrieve(self, context, top_k):
        return [ScoredDocument(id="d1", content="c1", score=0.9, metadata={"meta": "val"})]

def test_i6_langchain_smoke():
    controller = RAGtuneController(
        retriever=FakeRetriever(),
        reformulator=IdentityReformulator(),
        reranker=SimulatedReranker(),
        assembler=GreedyAssembler(),
        scheduler=GracefulDegradationScheduler(batch_size=1, llm_limit=1),
        estimator=BaselineEstimator(),
        budget=CostBudget.simple(docs=1)
    )
    
    lc_adapter = RAGtuneLangChainAdapter(controller)
    
    # LangChain invoke
    results = lc_adapter.invoke("test query")
    
    from langchain_core.documents import Document
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert results[0].page_content == "c1"
    assert results[0].metadata["id"] == "d1"
    assert "final_score" in results[0].metadata
