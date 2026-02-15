import pytest
from ragtune.core.types import ScoredDocument, RAGtuneContext
from ragtune.core.budget import CostTracker, CostBudget
from ragtune.core.types import ControllerTrace
from ragtune.components.estimators import UtilityEstimator

def test_estimator_boost_by_source():
    estimator = UtilityEstimator()
    
    # Create docs: A and B share source "wiki"
    doc_a = ScoredDocument(id="a", content="A", metadata={"source": "wiki"}, score=0.5)
    doc_b = ScoredDocument(id="b", content="B", metadata={"source": "wiki"}, score=0.5)
    doc_c = ScoredDocument(id="c", content="C", metadata={"source": "news"}, score=0.5)
    
    pool = [doc_a, doc_b, doc_c]
    
    tracker = CostTracker(CostBudget(), ControllerTrace())
    context = RAGtuneContext(query="test", tracker=tracker)
    
    # Round 1: No ones ranked yet
    initial_estimates = estimator.estimate(pool, [], context)
    assert initial_estimates == [0.5, 0.5, 0.5]
    
    # Simulate doc_a being reranked with high score
    doc_a_reranked = doc_a.model_copy(update={"reranker_score": 0.9, "score": 0.9})
    pool[0] = doc_a_reranked
    
    # doc_b should be boosted because it shares "wiki" source with winner doc_a
    new_estimates = estimator.estimate(pool, [0], context)
    
    assert new_estimates[1] > 0.5 # Boosted
    assert new_estimates[2] == 0.5 # Not boosted (different source)
    assert new_estimates[0] == 0.9 # Keeps its score
    
    print(f"Estimates after feedback: {new_estimates}")

if __name__ == "__main__":
    test_estimator_boost_by_source()
