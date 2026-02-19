import pytest
from ragtune.core.pool import CandidatePool, PoolItem, ItemState
from ragtune.core.types import RAGtuneContext, ControllerTrace
from ragtune.core.budget import CostTracker, CostBudget
from ragtune.components.estimators import SimilarityEstimator
from unittest.mock import MagicMock, patch
import numpy as np

def test_similarity_estimator_boosting():
    # Mock SentenceTransformer globally
    from unittest.mock import MagicMock, patch
    import numpy as np
    
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        # Mock encoding: SimilarityEstimator expects tensors or objects util.cos_sim can handle
        import torch
        def mock_encode(texts, **kwargs):
            return torch.tensor([[1.0, 0.0] for _ in texts])
        
        mock_model.encode.side_effect = mock_encode
        
        estimator = SimilarityEstimator(model_name="mock")
        
        items = [
            PoolItem(doc_id="a", content="fox", sources={"ret": 0.5}),
            PoolItem(doc_id="b", content="dog", sources={"ret": 0.5})
        ]
        pool = CandidatePool(items)
        
        tracker = CostTracker(CostBudget(), ControllerTrace())
        context = RAGtuneContext(query="fox", tracker=tracker)
        
        # Reranking item 'a' manually
        pool.transition(["a"], ItemState.IN_FLIGHT)
        pool.update_scores({"a": 1.0}, strategy="test")
        
        # Estimates for 'b' based on similarity to winner 'a'
        estimates = estimator.value(pool, context)
        assert "b" in estimates
        # Mock returns identical vectors, so similarity is max
        assert estimates["b"].priority > 0.0
