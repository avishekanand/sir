import pytest
from typing import List, Dict, Optional
from ragtune.core.controller import RAGtuneController
from ragtune.core.pool import CandidatePool, PoolItem, ItemState
from ragtune.core.types import ScoredDocument, RAGtuneContext, BatchProposal, CostObject, RemainingBudgetView, ControllerTrace
from ragtune.core.interfaces import BaseRetriever, BaseReformulator, BaseReranker, BaseScheduler, BaseEstimator, BaseAssembler
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.components.assemblers import GreedyAssembler

# --- Fakes/Mocks ---

class FakeRetriever(BaseRetriever):
    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        return [ScoredDocument(id=f"d{i}", content=f"c{i}", score=0.1) for i in range(top_k)]

class FakeReformulator(BaseReformulator):
    def generate(self, context: RAGtuneContext) -> List[str]:
        return [context.query]

class FakeEstimator(BaseEstimator):
    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, float]:
        return {it.doc_id: 0.5 for it in pool.get_eligible()}

class FakeScheduler(BaseScheduler):
    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.exhausted = False
        self.run_once = False # If True, will exhaust after one batch
    def select_batch(self, pool: CandidatePool, budget: RemainingBudgetView) -> Optional[BatchProposal]:
        if self.exhausted: return None
        eligible = pool.get_eligible()
        if not eligible: return None
        
        limit = min(self.batch_size, budget.remaining_rerank_docs)
        if limit <= 0: return None
        
        ids = [it.doc_id for it in eligible[:limit]]
        if self.run_once:
            self.exhausted = True
        return BatchProposal(
            doc_ids=ids, 
            strategy="identity", 
            expected_cost=CostObject(docs=len(ids))
        )

class FakeReranker(BaseReranker):
    def rerank(self, documents: List[PoolItem], context, strategy=None) -> Dict[str, float]:
        return {doc.doc_id: 0.9 for doc in documents}

class FakeAssembler(BaseAssembler):
    def assemble(self, candidates: List[PoolItem], context) -> List[ScoredDocument]:
        return [ScoredDocument(id=it.doc_id, content=it.content, score=it.final_score()) for it in candidates]

# --- Tests ---

def test_e22_and_e24_pop_batch_transitions_and_clean_exit():
    scheduler = FakeScheduler(batch_size=2)
    controller = RAGtuneController(
        retriever=FakeRetriever(),
        reformulator=FakeReformulator(),
        reranker=FakeReranker(),
        assembler=FakeAssembler(),
        scheduler=scheduler,
        estimator=FakeEstimator(),
        budget=CostBudget.simple(docs=3) # Limit to 3 docs
    )
    
    # Run. Should take 2 batches: [d0, d1] then [d2]
    # Then stop because budget only has 1 docs left but scheduler asks for 2? 
    # Actually Scheduler asks for 2, budget view says 3 left -> 2 docs.
    # Second batch: budget view says 1 left -> 1 doc.
    # Third batch: budget view says 0 docs left -> Scheduler returns None (if implemented correctly)
    
    output = controller.run("test")
    # budget = 3. Batch 1 = 2 docs. Batch 2 = 1 doc. 
    # Total doc_ids in trace: d0, d1, d2
    rerank_events = [e for e in output.trace.events if e.action == "rerank_batch"]
    assert len(rerank_events) == 2
    assert rerank_events[0].details["doc_ids"] == ["d0", "d1"]
    assert rerank_events[1].details["doc_ids"] == ["d2"]

def test_e25_partial_rerank_results_missing_become_dropped():
    class PartialReranker(FakeReranker):
        def rerank(self, documents, context, strategy=None):
            # Only return score for the first doc in batch
            return {documents[0].doc_id: 1.0}

    controller = RAGtuneController(
        retriever=FakeRetriever(),
        reformulator=FakeReformulator(),
        reranker=PartialReranker(),
        assembler=GreedyAssembler(),
        scheduler=FakeScheduler(batch_size=2),
        estimator=FakeEstimator(),
        budget=CostBudget.simple(docs=10)
    )
    
    # We need a way to stop the loop after 1 batch for this test
    controller.scheduler.run_once = True
    output = controller.run("test")
        
    # d1 should be DROPPED
    doc_ids = [d.id for d in output.documents]
    assert "d0" in doc_ids
    assert "d1" not in doc_ids

def test_e26_exception_path_dropped():
    class CrashingReranker(FakeReranker):
        def rerank(self, documents, context, strategy=None):
            raise ValueError("Boom")

    controller = RAGtuneController(
        retriever=FakeRetriever(),
        reformulator=FakeReformulator(),
        reranker=CrashingReranker(),
        assembler=FakeAssembler(),
        scheduler=FakeScheduler(batch_size=2),
        estimator=FakeEstimator(),
        budget=CostBudget.simple(docs=10)
    )
    controller.scheduler.run_once = True
    output = controller.run("test")
    # Batch [d0, d1] should be DROPPED
    # Wait, RAGtuneController.run has:
    # except Exception as e:
    #     pool.transition(proposal.doc_ids, ItemState.DROPPED)
    # This is implemented!
    
    # Trace should have rerank_error
    errors = [e for e in output.trace.events if e.action == "rerank_error"]
    assert len(errors) == 1
    assert errors[0].details["doc_ids"] == ["d0", "d1"]

def test_e29_traceability():
    controller = RAGtuneController(
        retriever=FakeRetriever(),
        reformulator=FakeReformulator(),
        reranker=FakeReranker(),
        assembler=FakeAssembler(),
        scheduler=FakeScheduler(batch_size=1),
        estimator=FakeEstimator(),
        budget=CostBudget.simple(docs=1)
    )
    output = controller.run("test")
    rerank_event = [e for e in output.trace.events if e.action == "rerank_batch"][0]
    assert rerank_event.details["strategy"] == "identity"
    assert "d0" in rerank_event.details["doc_ids"]

def test_e28_final_score_precedence():
    # doc1: reranker=0.2, priority=0.9
    # doc2: reranker=None, priority=0.8
    # doc3: reranker=None, priority=0.0 (retrieval=0.5)
    items = [
        PoolItem(doc_id="d1", content="c1", reranker_score=0.2, priority_value=0.9),
        PoolItem(doc_id="d2", content="c2", priority_value=0.8),
        PoolItem(doc_id="d3", content="c3", sources={"ret": 0.5})
    ]
    # Test precedence on individual items
    assert items[0].final_score() == 1000.2
    assert items[1].final_score() == 0.8
    assert items[2].final_score() == 0.5
    
    # Assert assembler sorts them correctly
    from ragtune.components.assemblers import GreedyAssembler
    assembler = GreedyAssembler()
    # CostTracker needed for GreedyAssembler
    tracker = CostTracker(CostBudget(), ControllerTrace())
    ctx = RAGtuneContext(query="t", tracker=tracker)
    results = assembler.assemble(items, ctx)
    
    # Sorted by final_score descending: d1 (1000.2), d2 (0.8), d3 (0.5)
    assert results[0].id == "d1"
    assert results[1].id == "d2"
    assert results[2].id == "d3"
