import pytest
from unittest.mock import MagicMock, patch
from ragtune.components.reformulators import LLMReformulator
from ragtune.core.types import RAGtuneContext
from ragtune.core.budget import CostTracker, CostBudget
from ragtune.core.types import ControllerTrace

class MockResponse:
    def __init__(self, content):
        self.choices = [MagicMock()]
        self.choices[0].message.content = content

@pytest.fixture
def mock_context():
    trace = ControllerTrace()
    budget = CostBudget(limits={"reformulations": 5})
    tracker = CostTracker(budget, trace)
    return RAGtuneContext(query="What is RAG?", tracker=tracker)

def test_llm_reformulator_parsing_clean_json(mock_context):
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MockResponse('["how does RAG work", "explain retrieval augmented generation"]')
        
        reformulator = LLMReformulator(model_name="test-model")
        results = reformulator.generate(mock_context)
        
        assert len(results) == 2
        assert "how does RAG work" in results

def test_llm_reformulator_strips_code_fences(mock_context):
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MockResponse('```json\n["how does RAG work", "explain retrieval augmented generation"]\n```')
        
        reformulator = LLMReformulator()
        results = reformulator.generate(mock_context)
        assert len(results) == 2

def test_llm_reformulator_handles_leading_trailing_text(mock_context):
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MockResponse('Sure, here you go: ["how does RAG work", "explain retrieval augmented generation"] hope this helps!')
        
        reformulator = LLMReformulator()
        results = reformulator.generate(mock_context)
        assert len(results) == 2

def test_llm_reformulator_drops_original_query(mock_context):
    with patch("litellm.completion") as mock_completion:
        # Original query is "What is RAG?"
        mock_completion.return_value = MockResponse('["What is RAG?", "how does RAG work"]')
        
        reformulator = LLMReformulator()
        results = reformulator.generate(mock_context)
        assert results == ["how does RAG work"]

def test_llm_reformulator_drops_empty_whitespace(mock_context):
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MockResponse('["", "   ", "how does RAG work"]')
        
        reformulator = LLMReformulator()
        results = reformulator.generate(mock_context)
        assert results == ["how does RAG work"]

def test_llm_reformulator_near_duplicate_filtering(mock_context):
    with patch("litellm.completion") as mock_completion:
        # Highly similar queries
        mock_completion.return_value = MockResponse('["What is RAG system?", "What is RAG systems?"]')
        
        reformulator = LLMReformulator()
        results = reformulator.generate(mock_context)
        assert len(results) == 1
        assert results == ["What is RAG system?"]

def test_llm_reformulator_non_json_output(mock_context):
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MockResponse('This is not JSON at all.')
        
        reformulator = LLMReformulator()
        results = reformulator.generate(mock_context)
        assert results == []

def test_llm_reformulator_malformed_json(mock_context):
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MockResponse('[ "unclosed quote ]')
        
        reformulator = LLMReformulator()
        results = reformulator.generate(mock_context)
        assert results == []

def test_llm_reformulator_respects_m(mock_context):
    with patch("litellm.completion") as mock_completion:
        # Defaults to m=2 in config
        mock_completion.return_value = MockResponse('["v1", "v2", "v3", "v4"]')
        
        reformulator = LLMReformulator()
        results = reformulator.generate(mock_context)
        assert len(results) == 2

def test_llm_reformulator_max_tokens_enforcement(mock_context):
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MockResponse('["variation 1"]')
        
        reformulator = LLMReformulator()
        reformulator.generate(mock_context)
        
        # Verify max_tokens was passed to litellm.completion
        _, kwargs = mock_completion.call_args
        assert "max_tokens" in kwargs
        assert kwargs["max_tokens"] == 1000

