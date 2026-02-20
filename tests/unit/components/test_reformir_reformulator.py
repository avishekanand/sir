import sys
import pytest
from unittest.mock import MagicMock
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.types import RAGtuneContext, ControllerTrace
from ragtune.components.reformulators import ReformIRReformulator


def make_context(reformulation_limit=3):
    trace = ControllerTrace()
    budget = CostBudget(limits={"reformulations": reformulation_limit})
    tracker = CostTracker(budget, trace)
    return RAGtuneContext(query="explain neural networks", tracker=tracker)


def test_budget_exhausted_returns_empty():
    ctx = make_context(reformulation_limit=0)
    result = ReformIRReformulator(n_variants=3).generate(ctx)
    assert result == []


def test_querygym_variants_returned(monkeypatch):
    """When querygym works, returns its variant outputs filtered."""
    ctx = make_context()

    mock_result = MagicMock()
    mock_result.metadata = {
        "variant_outputs": {
            "variant_1": {"raw_output": "neural network tutorial"},
            "variant_2": {"raw_output": "deep learning basics"},
            "variant_3": {"raw_output": "AI network explanation"},
        }
    }
    mock_qg = MagicMock()
    mock_qg.create_reformulator.return_value.reformulate.return_value = mock_result
    mock_qg.QueryItem = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "querygym", mock_qg)

    result = ReformIRReformulator(n_variants=5).generate(ctx)

    assert len(result) > 0
    assert "explain neural networks" not in result  # original filtered out


def test_querygym_unavailable_falls_back_to_llm(monkeypatch):
    """When querygym is absent, falls back to LLM rewrite path."""
    ctx = make_context()

    # Simulate querygym not installed
    monkeypatch.setitem(sys.modules, "querygym", None)

    mock_response = MagicMock()
    mock_response.choices[0].message.content = '["neural nets explained", "deep learning intro"]'

    import litellm
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: mock_response)

    result = ReformIRReformulator(n_variants=2).generate(ctx)

    assert len(result) >= 1
    assert all(q != "explain neural networks" for q in result)


def test_both_querygym_and_llm_fail_returns_empty(monkeypatch):
    """If both querygym and LLM fail, return empty list (no crash)."""
    ctx = make_context()

    monkeypatch.setitem(sys.modules, "querygym", None)

    import litellm
    monkeypatch.setattr(litellm, "completion", MagicMock(side_effect=RuntimeError("api down")))

    result = ReformIRReformulator(n_variants=2).generate(ctx)
    assert result == []


def test_near_duplicates_filtered(monkeypatch):
    """Near-duplicate variants are de-duplicated before returning."""
    ctx = make_context()

    monkeypatch.setitem(sys.modules, "querygym", None)

    mock_response = MagicMock()
    # "neural nets" and "neural net" are near-duplicates (ratio > 0.8)
    mock_response.choices[0].message.content = (
        '["neural nets", "neural net", "deep learning overview", "AI fundamentals"]'
    )

    import litellm
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: mock_response)

    result = ReformIRReformulator(n_variants=5).generate(ctx)
    assert len(result) == len(set(result))  # no exact duplicates
    assert len(result) < 4  # near-duplicate removed
