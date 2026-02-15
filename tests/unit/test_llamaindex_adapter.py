import pytest
from unittest.mock import MagicMock
from ragtune.adapters.llamaindex import LlamaIndexRetriever
from ragtune.core.types import ScoredDocument, RAGtuneContext
from ragtune.core.budget import CostTracker, CostBudget
from ragtune.core.types import ControllerTrace

class MockNode:
    def __init__(self, node_id, content, metadata=None):
        self.node_id = node_id
        self.content = content
        self.metadata = metadata or {}
    
    def get_content(self):
        return self.content

class MockNodeWithScore:
    def __init__(self, node, score):
        self.node = node
        self.score = score

def test_llamaindex_retriever_sync():
    mock_li = MagicMock()
    nodes = [
        MockNodeWithScore(MockNode("n1", "Llama works", {"author": "LI"}), 0.95),
        MockNodeWithScore(MockNode("n2", "Index works"), None)
    ]
    mock_li.retrieve.return_value = nodes
    
    adapter = LlamaIndexRetriever(mock_li)
    tracker = CostTracker(CostBudget(), ControllerTrace())
    context = RAGtuneContext(query="test", tracker=tracker)
    results = adapter.retrieve(context)
    
    assert len(results) == 2
    assert results[0].id == "n1"
    assert results[0].content == "Llama works"
    assert results[0].score == 0.95
    assert results[0].metadata["author"] == "LI"
    assert results[1].id == "n2"
    assert results[1].score == 0.5  # Reciprocal default

@pytest.mark.asyncio
async def test_llamaindex_retriever_async():
    mock_li = MagicMock()
    # Mock both retrieve and aretrieve to test the fallback logic
    nodes = [MockNodeWithScore(MockNode("n1", "Async Llama"), 0.8)]
    
    # Test with aretrieve available
    async def mock_aretrieve(q): return nodes
    mock_li.aretrieve = mock_aretrieve
    
    adapter = LlamaIndexRetriever(mock_li)
    tracker = CostTracker(CostBudget(), ControllerTrace())
    context = RAGtuneContext(query="test", tracker=tracker)
    results = await adapter.aretrieve(context)
    
    assert len(results) == 1
    assert results[0].id == "n1"
    assert results[0].content == "Async Llama"
