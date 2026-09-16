"""
Unit tests for ragtune.indexing.dense_indexer.

FakeDenseIndexer exercises the shared DenseIndexer template logic (encoding,
batching, normalization, build/search orchestration) via a trivial in-memory
storage backend, decoupled from FAISS/numpy specifics. FaissIndexer and
NumpyIndexer each get their own backend-specific tests below.

Heavy dependencies (faiss, sentence_transformers) are mocked so tests run
without GPU or large model downloads.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

from ragtune.indexing.base import SearchResult
from ragtune.indexing.dense_indexer import (
    DenseIndexer,
    FaissIndexer,
    FaissIndexData,
    NumpyIndexer,
    NumpyIndexData,
)

SMALL_CORPUS = {
    "d1": {"text": "Dense retrieval uses embeddings", "title": ""},
    "d2": {"text": "BM25 is a lexical baseline",      "title": ""},
    "d3": {"text": "ColBERT uses late interaction",   "title": ""},
}

DIM = 8  # tiny dimension for tests


# ---------------------------------------------------------------------------
# FakeDenseIndexer: trivial in-memory backend for testing shared template logic
# ---------------------------------------------------------------------------

class FakeDenseIndexer(DenseIndexer):
    """Minimal concrete DenseIndexer with in-memory storage (no real I/O)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.store = {}

    def encode_corpus(self, texts: List[str]) -> np.ndarray:
        return np.random.rand(len(texts), DIM).astype(np.float32)

    def _save_vectors(self, vectors, docnos, index_path):
        self.store[index_path] = (vectors, docnos)

    def exists(self, index_path: str) -> bool:
        return index_path in self.store

    def load(self, index_path: str) -> Any:
        return self.store[index_path]

    def _search_vectors(self, query_vec, data, top_k):
        vectors, docnos = data
        scores = vectors @ query_vec
        top_idx = np.argsort(-scores)[:top_k]
        return [SearchResult(doc_id=docnos[i], score=float(scores[i])) for i in top_idx]


class TestDenseIndexerBuildFromCorpus:

    def test_calls_save_vectors_with_docnos_and_matrix(self, tmp_path):
        indexer = FakeDenseIndexer("fake-model")
        path = str(tmp_path / "idx")
        indexer.build_from_corpus(SMALL_CORPUS, path)

        vectors, docnos = indexer.store[path]
        assert set(docnos) == {"d1", "d2", "d3"}
        assert vectors.shape == (3, DIM)

    def test_returns_true(self, tmp_path):
        indexer = FakeDenseIndexer("fake-model")
        assert indexer.build_from_corpus(SMALL_CORPUS, str(tmp_path / "idx")) is True

    def test_creates_index_directory(self, tmp_path):
        indexer = FakeDenseIndexer("fake-model")
        path = tmp_path / "idx"
        indexer.build_from_corpus(SMALL_CORPUS, str(path))
        assert path.is_dir()


class TestDenseIndexerNormalization:

    def test_normalized_vectors_have_unit_norm(self, tmp_path):
        path = str(tmp_path / "idx")
        indexer = FakeDenseIndexer("m", normalize=True)
        indexer.build_from_corpus(SMALL_CORPUS, path)
        vectors, _ = indexer.store[path]
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)

    def test_skip_normalization_when_disabled(self, tmp_path):
        class FixedNormIndexer(FakeDenseIndexer):
            def encode_corpus(self, texts):
                return np.full((len(texts), DIM), 0.5, dtype=np.float32)

        path = str(tmp_path / "idx")
        indexer = FixedNormIndexer("m", normalize=False)
        indexer.build_from_corpus(SMALL_CORPUS, path)
        vectors, _ = indexer.store[path]
        norms = np.linalg.norm(vectors, axis=1)
        assert not np.allclose(norms, 1.0, atol=1e-5)


class TestDenseIndexerSearchTemplate:

    def test_search_loads_index_and_delegates_to_search_vectors(self, tmp_path):
        path = str(tmp_path / "idx")
        indexer = FakeDenseIndexer("m")
        indexer.build_from_corpus(SMALL_CORPUS, path)

        results = indexer.search("some query", top_k=2, index_path=path)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.doc_id in {"d1", "d2", "d3"} for r in results)


# ---------------------------------------------------------------------------
# FaissIndexer (concrete: SentenceTransformer encoding + faiss.IndexFlatIP)
# ---------------------------------------------------------------------------

def _make_mock_faiss():
    mock_faiss = MagicMock()
    mock_index = MagicMock()
    mock_faiss.IndexFlatIP.return_value = mock_index
    mock_faiss.read_index.return_value = mock_index
    return mock_faiss, mock_index


class TestFaissIndexerBuildFromCorpus:

    def test_creates_three_files(self, tmp_path):
        mock_faiss, _ = _make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            indexer = FaissIndexer("fake-model")
            indexer.encode_corpus = lambda texts: np.random.rand(len(texts), DIM).astype(np.float32)
            indexer.build_from_corpus(SMALL_CORPUS, str(tmp_path / "idx"))

        mock_faiss.write_index.assert_called_once()
        assert (tmp_path / "idx" / "docnos.json").exists()
        assert (tmp_path / "idx" / "metadata.json").exists()

    def test_vectors_added_to_faiss_index(self, tmp_path):
        mock_faiss, mock_index = _make_mock_faiss()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            indexer = FaissIndexer("m")
            indexer.encode_corpus = lambda texts: np.random.rand(len(texts), DIM).astype(np.float32)
            indexer.build_from_corpus(SMALL_CORPUS, str(tmp_path / "idx"))

        mock_index.add.assert_called_once()
        assert mock_index.add.call_args[0][0].shape == (3, DIM)

    def test_raises_if_faiss_missing(self, tmp_path):
        with patch.dict("sys.modules", {"faiss": None}):
            indexer = FaissIndexer("m")
            indexer.encode_corpus = lambda texts: np.random.rand(len(texts), DIM).astype(np.float32)
            with pytest.raises(ImportError, match="faiss-cpu"):
                indexer.build_from_corpus(SMALL_CORPUS, str(tmp_path / "idx"))


class TestFaissIndexerExists:

    def test_false_when_missing(self, tmp_path):
        assert FaissIndexer("m").exists(str(tmp_path / "no_index")) is False

    def test_true_when_both_files_present(self, tmp_path):
        idx = tmp_path / "idx"
        idx.mkdir()
        (idx / "index.faiss").write_bytes(b"")
        (idx / "docnos.json").write_text("[]")
        assert FaissIndexer("m").exists(str(idx)) is True


class TestFaissIndexerLoad:

    def test_returns_faiss_index_data(self, tmp_path):
        idx = tmp_path / "idx"
        idx.mkdir()
        (idx / "docnos.json").write_text(json.dumps(["d1", "d2"]))
        (idx / "metadata.json").write_text(json.dumps({"model_name": "my-model", "dim": 8, "num_docs": 2}))

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.read_index.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            data = FaissIndexer("my-model").load(str(idx))

        assert isinstance(data, FaissIndexData)
        assert data.docnos == ["d1", "d2"]
        assert data.index is mock_index


class TestFaissIndexerSearch:

    def test_search_vectors_maps_indices_to_docnos(self):
        data = FaissIndexData(index=MagicMock(), docnos=["a", "b", "c"], model_name="m", index_path="/tmp/x")
        data.index.search.return_value = (
            np.array([[0.9, 0.5]]),
            np.array([[2, 0]]),
        )
        indexer = FaissIndexer("m")
        results = indexer._search_vectors(np.zeros(DIM, dtype=np.float32), data, top_k=2)

        assert results[0].doc_id == "c"
        assert results[0].score == pytest.approx(0.9)
        assert results[1].doc_id == "a"
        assert results[1].score == pytest.approx(0.5)

    def test_search_vectors_skips_missing_results(self):
        """FAISS returns -1 for indices when fewer than top_k results exist."""
        data = FaissIndexData(index=MagicMock(), docnos=["a"], model_name="m", index_path="/tmp/x")
        data.index.search.return_value = (np.array([[0.9, 0.0]]), np.array([[0, -1]]))
        indexer = FaissIndexer("m")
        results = indexer._search_vectors(np.zeros(DIM, dtype=np.float32), data, top_k=2)
        assert len(results) == 1
        assert results[0].doc_id == "a"


class TestFaissIndexerEncodeCorpus:

    def test_encode_corpus_uses_sentence_transformer(self, tmp_path):
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((2, DIM), dtype=np.float32)
        mock_st_module.SentenceTransformer.return_value = mock_model

        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP.return_value = MagicMock()

        corpus = {"a": {"text": "alpha"}, "b": {"text": "beta"}}

        with patch.dict("sys.modules", {"faiss": mock_faiss, "sentence_transformers": mock_st_module}):
            indexer = FaissIndexer("fake-model", batch_size=2)
            indexer.build_from_corpus(corpus, str(tmp_path / "idx"))

        mock_st_module.SentenceTransformer.assert_called_once_with("fake-model", device="cpu")
        mock_model.encode.assert_called()

    def test_model_loaded_lazily(self):
        indexer = FaissIndexer("fake-model")
        assert indexer._model is None

    def test_default_model_is_minilm(self):
        indexer = FaissIndexer()
        assert "MiniLM" in indexer.model_name_or_path


# ---------------------------------------------------------------------------
# NumpyIndexer (concrete: SentenceTransformer encoding + plain .npy storage)
# ---------------------------------------------------------------------------

class TestNumpyIndexerBuildFromCorpus:

    def test_creates_three_files(self, tmp_path):
        indexer = NumpyIndexer("m")
        indexer.encode_corpus = lambda texts: np.random.rand(len(texts), DIM).astype(np.float32)
        indexer.build_from_corpus(SMALL_CORPUS, str(tmp_path / "idx"))

        assert (tmp_path / "idx" / "vectors.npy").exists()
        assert (tmp_path / "idx" / "docnos.json").exists()
        assert (tmp_path / "idx" / "metadata.json").exists()

    def test_saved_matrix_shape_matches_corpus(self, tmp_path):
        indexer = NumpyIndexer("m")
        indexer.encode_corpus = lambda texts: np.random.rand(len(texts), DIM).astype(np.float32)
        indexer.build_from_corpus(SMALL_CORPUS, str(tmp_path / "idx"))

        vectors = np.load(tmp_path / "idx" / "vectors.npy")
        assert vectors.shape == (3, DIM)


class TestNumpyIndexerExists:

    def test_false_when_missing(self, tmp_path):
        assert NumpyIndexer("m").exists(str(tmp_path / "no_index")) is False

    def test_true_when_both_files_present(self, tmp_path):
        idx = tmp_path / "idx"
        idx.mkdir()
        np.save(idx / "vectors.npy", np.zeros((1, DIM), dtype=np.float32))
        (idx / "docnos.json").write_text("[]")
        assert NumpyIndexer("m").exists(str(idx)) is True


class TestNumpyIndexerLoadAndSearch:

    def test_load_returns_numpy_index_data(self, tmp_path):
        indexer = NumpyIndexer("m")
        indexer.encode_corpus = lambda texts: np.random.rand(len(texts), DIM).astype(np.float32)
        path = str(tmp_path / "idx")
        indexer.build_from_corpus(SMALL_CORPUS, path)

        data = indexer.load(path)
        assert isinstance(data, NumpyIndexData)
        assert set(data.docnos) == {"d1", "d2", "d3"}
        assert data.vectors.shape == (3, DIM)

    def test_search_returns_exact_top_k_by_inner_product(self, tmp_path):
        # Build a deterministic, already-orthonormal-ish corpus so exact
        # nearest neighbor is unambiguous.
        indexer = NumpyIndexer("m", normalize=False)
        fixed_vectors = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # d1
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # d2
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # d3 (close to d1)
        ], dtype=np.float32)
        indexer.encode_corpus = MagicMock(side_effect=lambda texts: fixed_vectors[: len(texts)])
        path = str(tmp_path / "idx")
        indexer.build_from_corpus(SMALL_CORPUS, path)

        # Query vector matches d1's direction exactly.
        indexer.encode_corpus = MagicMock(return_value=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
        results = indexer.search("doc one query", top_k=2, index_path=path)

        assert results[0].doc_id == "d1"
        assert results[1].doc_id == "d3"

    def test_search_skips_missing_files_gracefully_via_exists_check(self, tmp_path):
        indexer = NumpyIndexer("m")
        assert indexer.exists(str(tmp_path / "missing")) is False


class TestNumpyIndexerEncodeCorpus:

    def test_default_model_is_minilm(self):
        indexer = NumpyIndexer()
        assert "MiniLM" in indexer.model_name_or_path

    def test_encode_corpus_uses_sentence_transformer(self, tmp_path):
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((2, DIM), dtype=np.float32)
        mock_st_module.SentenceTransformer.return_value = mock_model

        corpus = {"a": {"text": "alpha"}, "b": {"text": "beta"}}

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            indexer = NumpyIndexer("fake-model", batch_size=2)
            indexer.build_from_corpus(corpus, str(tmp_path / "idx"))

        mock_st_module.SentenceTransformer.assert_called_once_with("fake-model", device="cpu")
        mock_model.encode.assert_called()
