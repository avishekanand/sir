import pytest
from unittest.mock import MagicMock
from ragtune.adapters.langchain import LangChainRetriever
from ragtune.core.types import ScoredDocument, RAGtuneContext
from ragtune.core.budget import CostTracker, CostBudget
from ragtune.core.types import ControllerTrace

def test_langchain_retriever_conversion():
    # 1. Setup Mock LangChain Documents
    # We mock Document because we don't want a hard dependency on langchain for this unit test
    mock_doc = MagicMock()
    mock_doc.page_content = "Hello RAGtune"
    mock_doc.metadata = {"id": "test_1", "tokens": 5, "extra": "data"}
    
    # 2. Setup Mock Retriever
    mock_lc_retriever = MagicMock()
    # Newer LC interface uses invoke()
    mock_lc_retriever.invoke.return_value = [mock_doc]
    
    # 3. Use Adapter
    tracker = CostTracker(CostBudget(), ControllerTrace())
    context = RAGtuneContext(query="test query", tracker=tracker)
    adapter = LangChainRetriever(mock_lc_retriever)
    results = adapter.retrieve(context, top_k=1)
    
    # 4. Assertions
    assert len(results) == 1
    doc = results[0]
    assert isinstance(doc, ScoredDocument)
    assert doc.id == "test_1"
    assert doc.content == "Hello RAGtune"
    assert doc.metadata["extra"] == "data"
    assert doc.token_count == 5

def test_langchain_retriever_id_fallback():
    mock_doc = MagicMock()
    mock_doc.page_content = "No ID here"
    mock_doc.metadata = {} # No ID
    
    mock_lc_retriever = MagicMock()
    # Fallback to get_relevant_documents for older LC compatibility
    del mock_lc_retriever.invoke
    mock_lc_retriever.get_relevant_documents.return_value = [mock_doc]
    
    adapter = LangChainRetriever(mock_lc_retriever)
    tracker = CostTracker(CostBudget(), ControllerTrace())
    context = RAGtuneContext(query="query", tracker=tracker)
    results = adapter.retrieve(context)
    
    assert len(results) == 1
    assert results[0].id == "0" # Fallback to index
    assert results[0].token_count > 0 # Estimated correctly
