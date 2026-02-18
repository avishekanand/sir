import pytest
from unittest.mock import MagicMock, patch
from ragtune.core.controller import RAGtuneController
from ragtune.core.types import RAGtuneContext, ScoredDocument
from ragtune.core.budget import CostBudget
from ragtune.components.reformulators import IdentityReformulator, LLMReformulator
from ragtune.utils.config import config

@pytest.fixture
def mock_components():
    retriever = MagicMock()
    reformulator = MagicMock()
    reranker = MagicMock()
    assembler = MagicMock()
    scheduler = MagicMock()
    estimator = MagicMock()
    
    # Default behavior
    retriever.retrieve.return_value = [ScoredDocument(id=f"doc_{i}", content="c", score=0.5, metadata={}) for i in range(5)]
    retriever.aretrieve.return_value = retriever.retrieve.return_value
    
    reformulator.generate.return_value = ["rewrite 1", "rewrite 2"]
    reformulator.agenerate.return_value = ["rewrite 1", "rewrite 2"]
    
    reranker.rerank.return_value = {}
    reranker.arerank.return_value = {}
    
    assembler.assemble.return_value = []
    assembler.aassemble.return_value = []
    
    scheduler.select_batch.return_value = None
    scheduler.aselect_batch.return_value = None
    
    return {
        "retriever": retriever,
        "reformulator": reformulator,
        "reranker": reranker,
        "assembler": assembler,
        "scheduler": scheduler,
        "estimator": estimator
    }

def test_controller_happy_path_multi_round(mock_components):
    controller = RAGtuneController(**mock_components)
    
    with patch("ragtune.utils.config.config.get") as mock_get:
        # original_query_depth=10, depth_per_reformulation=5, num_reformulations=2
        mock_get.side_effect = lambda k, d=None: 10 if "original_query_depth" in k else (5 if "depth_per_reformulation" in k else (50 if "max_pool_size" in k else d))
        
        controller.run("test query")
        
        # Verify 3 retrieval calls (1 original + 2 rewrites)
        assert mock_components["retriever"].retrieve.call_count == 3
        
        # Verify reformulator was called
        assert mock_components["reformulator"].generate.called

def test_controller_llm_failure_graceful(mock_components):
    mock_components["reformulator"].generate.return_value = [] # Failure/Empty
    controller = RAGtuneController(**mock_components)
    
    controller.run("test query")
    
    # Only original retrieval should run
    assert mock_components["retriever"].retrieve.call_count == 1

def test_controller_budget_exhaustion(mock_components):
    # Setup budget that denies reformulation
    budget = CostBudget(limits={"reformulations": 0})
    controller = RAGtuneController(**mock_components, budget=budget)
    
    # Use real LLMReformulator to see budget logic
    controller.reformulator = LLMReformulator()
    
    controller.run("test query")
    
    # Should only have 1 retrieval call (original) because reformulator returns empty list on budget deny
    assert mock_components["retriever"].retrieve.call_count == 1

def test_controller_partial_retrieval_success(mock_components):
    # Round 1 returns docs, Round 2 (rewrite 1) returns empty, Round 3 (rewrite 2) returns docs
    mock_components["retriever"].retrieve.side_effect = [
        [ScoredDocument(id="orig", content="c", score=0.5)], # original
        [], # rewrite 1
        [ScoredDocument(id="rewrite2", content="c", score=0.4)] # rewrite 2
    ]
    controller = RAGtuneController(**mock_components)
    
    output = controller.run("test query")
    
    # Should have 2 docs in pool
    assert len(output.documents) == 0 # Assembler returns empty list in mock
    # Check trace for pool size
    trace_events = output.trace.events
    pool_init_event = next(e for e in trace_events if e.action == "pool_init")
    assert pool_init_event.details["count"] == 2

def test_controller_caching(mock_components):
    controller = RAGtuneController(**mock_components)
    
    controller.run("same query")
    controller.run("same query")
    
    # Reformulator should only be called once
    assert mock_components["reformulator"].generate.call_count == 1
    # Retrieval still happens twice for the query, but queries were cached.

def test_controller_trace_richness(mock_components):
    controller = RAGtuneController(**mock_components)
    output = controller.run("test query")
    
    trace_events = output.trace.events
    pool_init_event = next(e for e in trace_events if e.action == "pool_init")
    
    assert "count" in pool_init_event.details
    assert "reformulations" in pool_init_event.details
    assert len(pool_init_event.details["reformulations"]) == 2
