import pytest
import pandas as pd
from unittest.mock import MagicMock
from ragtune.adapters.pyterrier import PyTerrierRetriever, RAGtuneTransformer
from ragtune.core.types import ScoredDocument, ControllerOutput, ControllerTrace, RAGtuneContext
from ragtune.core.budget import CostTracker, CostBudget

def test_pyterrier_retriever_conversion():
    # Mock PT transformer
    mock_pt = MagicMock()
    # Mock output DataFrame
    results_df = pd.DataFrame([
        {"qid": "q1", "docno": "d1", "text": "Doc 1 content", "score": 10.5, "extra": "info"},
        {"qid": "q1", "docno": "d2", "text": "Doc 2 content", "score": 8.2}
    ])
    mock_pt.transform.return_value = results_df
    
    adapter = PyTerrierRetriever(mock_pt)
    tracker = CostTracker(CostBudget(), ControllerTrace())
    context = RAGtuneContext(query="test query", tracker=tracker)
    results = adapter.retrieve(context)
    
    assert len(results) == 2
    assert results[0].id == "d1"
    assert results[0].content == "Doc 1 content"
    assert results[0].score == 10.5
    assert results[0].metadata["extra"] == "info"
    assert results[1].id == "d2"
    assert results[1].score == 8.2

def test_ragtune_transformer_conversion():
    # Mock Controller
    mock_controller = MagicMock()
    
    # Mock Controller output
    mock_output = ControllerOutput(
        query="test",
        documents=[
            ScoredDocument(id="d_out_1", content="Result 1", score=0.9, metadata={"meta": "val"}),
            ScoredDocument(id="d_out_2", content="Result 2", score=0.7)
        ],
        trace=ControllerTrace(),
        final_budget_state={}
    )
    mock_controller.run.return_value = mock_output
    
    transformer = RAGtuneTransformer(mock_controller)
    
    # Input queries DataFrame
    input_df = pd.DataFrame([{"qid": "q1", "query": "test query"}])
    
    res_df = transformer.transform(input_df)
    
    assert len(res_df) == 2
    assert res_df.iloc[0]["docno"] == "d_out_1"
    assert res_df.iloc[0]["score"] == 0.9
    assert res_df.iloc[0]["meta"] == "val"
    assert res_df.iloc[1]["docno"] == "d_out_2"
    assert "text" in res_df.columns
    assert res_df.iloc[0]["text"] == "Result 1"
