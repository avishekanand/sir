import pytest
import numpy as np
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.types import RAGtuneContext, ControllerTrace, ItemState
from ragtune.core.pool import CandidatePool, PoolItem
from ragtune.components.estimators import ReformIREstimator


def make_context():
    trace = ControllerTrace()
    budget = CostBudget(limits={"tokens": 4000})
    tracker = CostTracker(budget, trace)
    return RAGtuneContext(query="test query", tracker=tracker)


def make_item(doc_id, sources, reranker_score=None):
    item = PoolItem(doc_id=doc_id, content=f"content {doc_id}", sources=sources)
    if reranker_score is not None:
        item.reranker_score = reranker_score
        item.state = ItemState.RERANKED
    return item


def test_fallback_when_insufficient_reranked():
    """Before min_reranked items exist, priority = max source score."""
    est = ReformIREstimator(min_reranked_for_regression=3)
    pool = CandidatePool([
        make_item("d1", {"original": 0.8, "rewrite_0": 0.6}),
        make_item("d2", {"original": 0.3}),
    ])
    outputs = est.value(pool, make_context())
    assert outputs["d1"].priority == pytest.approx(0.8)
    assert outputs["d2"].priority == pytest.approx(0.3)


def test_fallback_returns_only_candidates():
    """Fallback path should not include RERANKED items."""
    est = ReformIREstimator(min_reranked_for_regression=5)
    pool = CandidatePool([
        make_item("r1", {"original": 0.9}, reranker_score=0.85),
        make_item("c1", {"original": 0.5}),
    ])
    outputs = est.value(pool, make_context())
    assert "r1" not in outputs
    assert "c1" in outputs


def test_weight_learning_applied_to_candidates():
    """With sufficient RERANKED items, learned weights are applied to CANDIDATEs."""
    est = ReformIREstimator(min_reranked_for_regression=2)
    pool = CandidatePool([
        make_item("r1", {"original": 1.0, "rewrite_0": 0.0}, reranker_score=0.9),
        make_item("r2", {"original": 0.0, "rewrite_0": 1.0}, reranker_score=0.2),
        make_item("c1", {"original": 0.8, "rewrite_0": 0.2}),
    ])
    outputs = est.value(pool, make_context())
    assert "c1" in outputs
    assert "r1" not in outputs
    assert "r2" not in outputs
    # Learned weights should favour "original" (higher reranker signal)
    # so c1 priority ~ 0.8 * alpha_orig + 0.2 * alpha_rw0
    assert outputs["c1"].priority > 0


def test_learned_weights_in_metadata():
    """Learned weights are stored in EstimatorOutput.metadata for feedback."""
    est = ReformIREstimator(min_reranked_for_regression=2)
    pool = CandidatePool([
        make_item("r1", {"original": 0.9}, reranker_score=0.85),
        make_item("r2", {"original": 0.4}, reranker_score=0.35),
        make_item("c1", {"original": 0.6}),
    ])
    outputs = est.value(pool, make_context())
    assert "reformir_weights" in outputs["c1"].metadata
    weights = outputs["c1"].metadata["reformir_weights"]
    assert "original" in weights
    assert all(0.0 <= v <= 1.0 for v in weights.values())


def test_weights_aggregatable_for_feedback():
    """Controller can collect reformir_weights from metadata across all outputs."""
    est = ReformIREstimator(min_reranked_for_regression=2)
    pool = CandidatePool([
        make_item("r1", {"original": 0.9, "rewrite_0": 0.1}, reranker_score=0.8),
        make_item("r2", {"original": 0.2, "rewrite_0": 0.8}, reranker_score=0.3),
        make_item("c1", {"original": 0.5, "rewrite_0": 0.5}),
        make_item("c2", {"original": 0.7, "rewrite_0": 0.3}),
    ])
    outputs = est.value(pool, make_context())

    estimates: dict = {}
    for out in outputs.values():
        estimates.update(out.metadata)

    assert "reformir_weights" in estimates
    assert isinstance(estimates["reformir_weights"], dict)


def test_empty_eligible_returns_empty():
    """No CANDIDATE items → empty dict returned."""
    est = ReformIREstimator(min_reranked_for_regression=2)
    pool = CandidatePool([
        make_item("r1", {"original": 0.9}, reranker_score=0.85),
        make_item("r2", {"original": 0.5}, reranker_score=0.45),
    ])
    outputs = est.value(pool, make_context())
    assert outputs == {}
