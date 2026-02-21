import sys
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.types import RAGtuneContext, ControllerTrace, ItemState
from ragtune.core.pool import PoolItem


def make_context(query="what causes COVID"):
    trace = ControllerTrace()
    tracker = CostTracker(CostBudget(), trace)
    return RAGtuneContext(query=query, tracker=tracker)


def make_item(doc_id, content="doc content", score=0.5):
    item = PoolItem(doc_id=doc_id, content=content, sources={"original": score})
    return item


def mock_monot5_reranker_class(scores: dict):
    """Returns a mock MonoT5ReRanker class whose transform() returns given scores."""
    mock_instance = MagicMock()
    mock_instance.transform.side_effect = lambda df: pd.DataFrame({
        "docno": list(scores.keys()),
        "score": list(scores.values()),
        "qid": "q0",
        "query": df["query"].iloc[0],
        "rank": range(len(scores)),
    })
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls, mock_instance


@pytest.fixture(autouse=True)
def patch_pyterrier(monkeypatch):
    """Prevent pt.init() from running during unit tests."""
    mock_pt = MagicMock()
    mock_pt.started.return_value = True
    monkeypatch.setitem(sys.modules, "pyterrier", mock_pt)
    monkeypatch.setitem(sys.modules, "pyterrier_t5", MagicMock())


def test_rerank_returns_scores_for_all_documents(monkeypatch):
    scores = {"d1": 0.9, "d2": 0.4, "d3": 0.1}
    mock_cls, mock_instance = mock_monot5_reranker_class(scores)

    monkeypatch.setitem(sys.modules, "pyterrier_t5", MagicMock(MonoT5ReRanker=mock_cls))

    from ragtune.components.rerankers import MonoT5Reranker
    reranker = MonoT5Reranker.__new__(MonoT5Reranker)
    reranker._reranker = mock_instance

    docs = [make_item("d1"), make_item("d2"), make_item("d3")]
    result = reranker.rerank(docs, make_context())

    assert set(result.keys()) == {"d1", "d2", "d3"}
    assert result["d1"] == pytest.approx(0.9)
    assert result["d2"] == pytest.approx(0.4)


def test_rerank_empty_documents_returns_empty(monkeypatch):
    mock_cls, mock_instance = mock_monot5_reranker_class({})
    monkeypatch.setitem(sys.modules, "pyterrier_t5", MagicMock(MonoT5ReRanker=mock_cls))

    from ragtune.components.rerankers import MonoT5Reranker
    reranker = MonoT5Reranker.__new__(MonoT5Reranker)
    reranker._reranker = mock_instance

    result = reranker.rerank([], make_context())
    assert result == {}
    mock_instance.transform.assert_not_called()


def test_rerank_dataframe_contains_query(monkeypatch):
    """The DataFrame passed to MonoT5 must include the query string."""
    captured = {}
    mock_instance = MagicMock()

    def capture_transform(df):
        captured["df"] = df.copy()
        return pd.DataFrame({"docno": df["docno"], "score": [0.5] * len(df), "qid": "q0", "rank": range(len(df)), "query": df["query"]})

    mock_instance.transform.side_effect = capture_transform
    monkeypatch.setitem(sys.modules, "pyterrier_t5", MagicMock(MonoT5ReRanker=MagicMock(return_value=mock_instance)))

    from ragtune.components.rerankers import MonoT5Reranker
    reranker = MonoT5Reranker.__new__(MonoT5Reranker)
    reranker._reranker = mock_instance

    ctx = make_context(query="what is COVID-19")
    reranker.rerank([make_item("d1"), make_item("d2")], ctx)

    assert "query" in captured["df"].columns
    assert all(captured["df"]["query"] == "what is COVID-19")
    assert list(captured["df"]["docno"]) == ["d1", "d2"]


def test_rerank_uses_final_score_as_initial_score(monkeypatch):
    """PoolItem.final_score() is passed as the initial score column."""
    captured = {}
    mock_instance = MagicMock()

    def capture_transform(df):
        captured["scores"] = list(df["score"])
        return pd.DataFrame({"docno": df["docno"], "score": list(df["score"]), "qid": "q0", "rank": range(len(df)), "query": df["query"]})

    mock_instance.transform.side_effect = capture_transform
    monkeypatch.setitem(sys.modules, "pyterrier_t5", MagicMock(MonoT5ReRanker=MagicMock(return_value=mock_instance)))

    from ragtune.components.rerankers import MonoT5Reranker
    reranker = MonoT5Reranker.__new__(MonoT5Reranker)
    reranker._reranker = mock_instance

    item = make_item("d1", score=0.75)
    reranker.rerank([item], make_context())

    assert captured["scores"][0] == pytest.approx(0.75)
