import pytest
import sys
from unittest.mock import MagicMock
import numpy as np

# Mock sentence_transformers BEFORE importing anything that uses it
mock_st = MagicMock()
sys.modules["sentence_transformers"] = mock_st

from ragtune.core.types import ScoredDocument, RAGtuneContext
from ragtune.core.budget import CostTracker, CostBudget
from ragtune.core.types import ControllerTrace
from ragtune.components.estimators import SimilarityEstimator

def test_similarity_estimator_boosting():
    # 1. Setup Mock Model
    mock_model = MagicMock()
    mock_st.SentenceTransformer.return_value = mock_model
    
    # doc_1 (Winner), doc_2 (Similar to doc_1), doc_3 (Different)
    doc_1 = ScoredDocument(id="doc_1", content="RAG is great", reranker_score=0.9, score=0.5)
    doc_2 = ScoredDocument(id="doc_2", content="Retrieval Augmented Generation", score=0.4)
    doc_3 = ScoredDocument(id="doc_3", content="Pizza is tasty", score=0.45)
    
    pool = [doc_1, doc_2, doc_3]
    
    # Mock embeddings: doc_1 and doc_2 are close, doc_3 is far
    embeddings = np.array([
        [1.0, 0.0], # doc_1
        [0.9, 0.1], # doc_2 (similar)
        [0.0, 1.0]  # doc_3 (different)
    ])
    mock_model.encode.return_value = embeddings
    
    # 2. Initialize Estimator
    estimator = SimilarityEstimator()
    
    # 3. Estimate with doc_1 as winner
    tracker = CostTracker(CostBudget(), ControllerTrace())
    context = RAGtuneContext(query="test", tracker=tracker)
    estimates = estimator.estimate(pool, [0], context)
    
    # 4. Assertions
    assert estimates[1] > pool[1].score # doc_2 boosted
    assert estimates[1] > estimates[2]  # doc_2 wins due to similarity
    print(f"\nBoosted Scores: doc_2={estimates[1]:.4f}, doc_3={estimates[2]:.4f}")

def test_hybrid_scheduler_escalation():
    from ragtune.components.schedulers import ActiveLearningScheduler
    from ragtune.core.types import RerankStrategy
    from ragtune.core.budget import CostTracker, CostBudget
    from ragtune.core.types import ControllerTrace

    # Mock estimator that returns ambiguous results (close scores)
    mock_estimator = MagicMock()
    mock_estimator.estimate.return_value = [0.5, 0.48, 0.2] # Gap 0.02 < 0.05
    
    scheduler = ActiveLearningScheduler(
        batch_size=1, 
        initial_strategy=RerankStrategy.CROSS_ENCODER,
        estimator=mock_estimator
    )
    
    pool = [
        ScoredDocument(id="1", content="X", score=0.5),
        ScoredDocument(id="2", content="Y", score=0.48),
        ScoredDocument(id="3", content="Z", score=0.2)
    ]
    tracker = CostTracker(CostBudget(), ControllerTrace())
    context = RAGtuneContext(query="test", tracker=tracker)
    
    # Should escalate to LLM due to low gap (ambiguity)
    proposal = scheduler.propose_next_batch(pool, [], context)
    assert proposal.strategy == RerankStrategy.LLM
    print(f"Escalated strategy: {proposal.strategy}")

if __name__ == "__main__":
    test_similarity_estimator_boosting()
    test_hybrid_scheduler_escalation()
