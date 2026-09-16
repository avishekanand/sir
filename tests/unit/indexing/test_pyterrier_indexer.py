"""
Unit tests for ragtune.indexing.pyterrier_indexer.PyTerrierIndexer.

PyTerrier (JVM) is mocked throughout so tests run without a Java installation.
"""

import os
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

import ragtune.indexing.pyterrier_indexer as pt_module
from ragtune.indexing.pyterrier_indexer import PyTerrierIndexer


SMALL_CORPUS = {
    "d1": {"text": "BM25 is a ranking function",   "title": "BM25"},
    "d2": {"text": "TF-IDF is a baseline",          "title": "TF-IDF"},
    "d3": {"text": "Dense retrieval uses embeddings","title": "Dense"},
}


@pytest.fixture
def mock_pt():
    """Provide a mocked pyterrier module injected into the indexer module."""
    mock = MagicMock()
    mock.started.return_value = True  # skip pt.init()
    mock.IterDictIndexer.return_value = MagicMock()
    mock.IndexFactory.of.return_value = MagicMock(name="pt_index")
    with patch.object(pt_module, "pt", mock):
        yield mock


class TestBuildFromCorpus:

    def test_creates_index_directory(self, tmp_path, mock_pt):
        indexer = PyTerrierIndexer()
        indexer.build_from_corpus(SMALL_CORPUS, str(tmp_path / "idx"))
        assert (tmp_path / "idx").is_dir()

    def test_calls_iter_dict_indexer(self, tmp_path, mock_pt):
        indexer = PyTerrierIndexer()
        indexer.build_from_corpus(SMALL_CORPUS, str(tmp_path / "idx"))

        mock_pt.IterDictIndexer.assert_called_once()
        mock_pt.IterDictIndexer.return_value.index.assert_called_once()

    def test_indexed_docs_have_docno_text_title(self, tmp_path, mock_pt):
        indexer = PyTerrierIndexer()
        indexer.build_from_corpus(SMALL_CORPUS, str(tmp_path / "idx"))

        # Capture the iterable passed to .index()
        call_args = mock_pt.IterDictIndexer.return_value.index.call_args
        docs = list(call_args[0][0])  # first positional arg, materialise generator

        assert len(docs) == 3
        docnos = {d["docno"] for d in docs}
        assert docnos == {"d1", "d2", "d3"}
        for doc in docs:
            assert "docno" in doc
            assert "text" in doc
            assert "title" in doc

    def test_returns_true_on_success(self, tmp_path, mock_pt):
        indexer = PyTerrierIndexer()
        result = indexer.build_from_corpus(SMALL_CORPUS, str(tmp_path / "idx"))
        assert result is True

    def test_raises_if_pyterrier_missing(self, tmp_path):
        with patch.object(pt_module, "pt", None):
            indexer = PyTerrierIndexer()
            with pytest.raises(ImportError, match="python-terrier"):
                indexer.build_from_corpus(SMALL_CORPUS, str(tmp_path / "idx"))


class TestExists:

    def test_returns_false_for_nonexistent_path(self, tmp_path):
        indexer = PyTerrierIndexer()
        assert indexer.exists(str(tmp_path / "no_such_index")) is False

    def test_returns_false_if_no_data_properties(self, tmp_path):
        (tmp_path / "myindex").mkdir()
        assert PyTerrierIndexer().exists(str(tmp_path / "myindex")) is False

    def test_returns_true_when_data_properties_present(self, tmp_path):
        idx = tmp_path / "myindex"
        idx.mkdir()
        (idx / "data.properties").write_text("num.Docs=3\n")
        assert PyTerrierIndexer().exists(str(idx)) is True


class TestLoad:

    def test_calls_index_factory_of(self, tmp_path, mock_pt):
        indexer = PyTerrierIndexer()
        result = indexer.load(str(tmp_path / "idx"))

        mock_pt.IndexFactory.of.assert_called_once()
        assert result is mock_pt.IndexFactory.of.return_value

    def test_raises_if_pyterrier_missing(self, tmp_path):
        with patch.object(pt_module, "pt", None):
            with pytest.raises(ImportError, match="python-terrier"):
                PyTerrierIndexer().load(str(tmp_path / "idx"))


class TestSearch:

    def test_search_returns_sorted_top_k(self, tmp_path, mock_pt):
        results_df = pd.DataFrame([
            {"qid": "q1", "docno": "d1", "score": 5.0},
            {"qid": "q1", "docno": "d2", "score": 9.5},
            {"qid": "q1", "docno": "d3", "score": 7.0},
        ])
        mock_retriever = MagicMock()
        mock_retriever.transform.return_value = results_df
        mock_pt.terrier.Retriever.return_value = mock_retriever

        results = PyTerrierIndexer().search("bm25 ranking", top_k=2, index_path=str(tmp_path / "idx"))

        assert len(results) == 2
        assert results[0].doc_id == "d2"
        assert results[0].score == pytest.approx(9.5)
        assert results[1].doc_id == "d3"

    def test_search_uses_bm25_weighting_model(self, tmp_path, mock_pt):
        mock_retriever = MagicMock()
        mock_retriever.transform.return_value = pd.DataFrame([{"qid": "q1", "docno": "d1", "score": 1.0}])
        mock_pt.terrier.Retriever.return_value = mock_retriever

        PyTerrierIndexer().search("query", top_k=5, index_path=str(tmp_path / "idx"))

        _, kwargs = mock_pt.terrier.Retriever.call_args
        assert kwargs.get("wmodel") == "BM25"

    def test_raises_if_pyterrier_missing(self, tmp_path):
        with patch.object(pt_module, "pt", None):
            with pytest.raises(ImportError, match="python-terrier"):
                PyTerrierIndexer().search("q", top_k=5, index_path=str(tmp_path / "idx"))
