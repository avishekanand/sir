import pytest
import asyncio
from typing import List, Optional
from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.types import ScoredDocument, BatchProposal, RAGtuneContext
from ragtune.core.interfaces import BaseRetriever, BaseReformulator, BaseReranker, BaseAssembler, BaseScheduler

class AsyncRetriever(BaseRetriever):
    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        return [ScoredDocument(id="1", content="Async works")]
    
    async def aretrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        await asyncio.sleep(0.01)
        return [ScoredDocument(id="1", content="Async works")]

class AsyncReranker(BaseReranker):
    def rerank(self, documents: List[ScoredDocument], context: RAGtuneContext) -> List[ScoredDocument]:
        return documents
    
    async def arerank(self, documents: List[ScoredDocument], context: RAGtuneContext) -> List[ScoredDocument]:
        await asyncio.sleep(0.01)
        return [d.model_copy(update={"reranker_score": 0.9}) for d in documents]

@pytest.mark.asyncio
async def test_controller_arun(fake_reformulator, fake_assembler, fake_scheduler):
    retriever = AsyncRetriever()
    reranker = AsyncReranker()
    
    controller = RAGtuneController(
        retriever=retriever,
        reformulator=fake_reformulator,
        reranker=reranker,
        assembler=fake_assembler,
        scheduler=fake_scheduler,
        budget=CostBudget(max_reranker_docs=10)
    )
    
    output = await controller.arun("test query")
    
    assert output.query == "test query"
    assert len(output.documents) > 0
    assert output.documents[0].reranker_score == 0.9
    assert any(e.action == "rerank_batch" for e in output.trace.events)
