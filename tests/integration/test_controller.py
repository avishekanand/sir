import pytest
from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.core.interfaces import BaseReranker

def test_happy_path(fake_retriever, fake_reformulator, fake_reranker, fake_assembler, fake_scheduler):
    budget = CostBudget(max_tokens=1000, max_reranker_docs=100, max_reformulations=1)
    controller = RAGtuneController(
        fake_retriever, 
        fake_reformulator, 
        fake_reranker, 
        fake_assembler, 
        fake_scheduler,
        budget
    )
    
    output = controller.run("text")
    
    assert len(output.documents) > 0
    # Trace should show rerank rounds
    rerank_events = [e for e in output.trace.events if e.action == "rerank_batch"]
    assert len(rerank_events) > 0

def test_active_loop_stops_on_budget(fake_retriever, fake_reformulator, fake_reranker, fake_assembler, fake_scheduler):
    # Set reranker doc limit to 3. Pool has 10. Scheduler does 1 at a time.
    budget = CostBudget(max_tokens=1000, max_reranker_docs=3)
    controller = RAGtuneController(
        fake_retriever, 
        fake_reformulator, 
        fake_reranker, 
        fake_assembler, 
        fake_scheduler,
        budget
    )
    
    output = controller.run("text")
    
    # Trace should show exactly 3 rerank rounds and one skip/halt
    rerank_events = [e for e in output.trace.events if e.action == "rerank_batch"]
    assert len(rerank_events) == 3
    
    skip_events = [e for e in output.trace.events if e.action == "skip_batch"]
    assert len(skip_events) == 1

def test_token_limit_assembly(fake_retriever, fake_reformulator, fake_reranker, fake_assembler, fake_scheduler):
    # Each doc has 10 tokens. Limit to 25. Should get 2 docs.
    budget = CostBudget(max_tokens=25, max_reranker_docs=100)
    controller = RAGtuneController(
        fake_retriever, 
        fake_reformulator, 
        fake_reranker, 
        fake_assembler, 
        fake_scheduler,
        budget
    )
    
    output = controller.run("text")
    
    assert len(output.documents) == 2
    assert output.final_budget_state["tokens"] == 20

def test_feedback_propagation(fake_reformulator, fake_assembler):
    # Use a real ActiveLearningScheduler for this one to test feedback
    from ragtune.components.schedulers import ActiveLearningScheduler
    from ragtune.core.types import ScoredDocument
    from ragtune.components.retrievers import InMemoryRetriever

    # doc_1 and doc_2 are same section. doc_1 will be winner.
    # doc_3 is different section.
    custom_pool = [
        ScoredDocument(id="doc_1", content="winner", metadata={"section": "A"}, score=0.5, token_count=10),
        ScoredDocument(id="doc_2", content="boost me", metadata={"section": "A"}, score=0.4, token_count=10),
        ScoredDocument(id="doc_3", content="static", metadata={"section": "B"}, score=0.45, token_count=10),
    ]
    
    retriever = InMemoryRetriever(custom_pool)
    
    # Mock Reranker that returns 0.9 for "winner"
    class MockWinningReranker(BaseReranker):
        def rerank(self, docs, query):
            results = []
            for d in docs:
                score = 0.9 if "winner" in d.content else 0.3
                results.append(d.model_copy(update={"score": score, "reranker_score": score}))
            return results

    scheduler = ActiveLearningScheduler(batch_size=1)
    budget = CostBudget(max_reranker_docs=10)
    
    controller = RAGtuneController(
        retriever, 
        fake_reformulator, 
        MockWinningReranker(), 
        fake_assembler, 
        scheduler,
        budget
    )
    
    output = controller.run("text")
    
    # Check execution order in trace
    rounds = [e.details['doc_ids'][0] for e in output.trace.events if e.action == "rerank_batch"]
    
    # Order should be doc_1 (best initial), then doc_2 (boosted by doc_1), then doc_3
    assert rounds == ["doc_1", "doc_2", "doc_3"]
