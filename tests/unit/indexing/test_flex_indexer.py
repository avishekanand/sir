"""
Unit tests for ragtune.indexing.flex_indexer.FlexIndexer._get_encoder routing.

Encoder classes themselves are mocked — these tests only verify FlexIndexer
resolves the right class and merges kwargs with the correct precedence.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import ragtune.indexing.flex_indexer as flex_module
from ragtune.indexing.flex_indexer import FlexIndexer


@pytest.fixture
def mock_pyterrier_dr():
    with patch.object(flex_module, "_require_pyterrier_dr", return_value=MagicMock()):
        yield


class TestGetEncoderRouting:

    def test_resolved_model_name_is_forwarded(self, mock_pyterrier_dr):
        mock_cls = MagicMock()
        with patch(
            "ragtune.indexing.encoders.resolve_encoder",
            return_value=(mock_cls, {}, "Qwen/Qwen3-Embedding-4B"),
        ):
            indexer = FlexIndexer(model_name="Qwen/Qwen3-Embedding-4B", device="cpu", batch_size=16)
            indexer._get_encoder()

        _, kwargs = mock_cls.call_args
        assert kwargs["model_name"] == "Qwen/Qwen3-Embedding-4B"

    def test_bare_shorthand_does_not_force_model_name(self, mock_pyterrier_dr):
        mock_cls = MagicMock()
        with patch(
            "ragtune.indexing.encoders.resolve_encoder",
            return_value=(mock_cls, {}, None),
        ):
            indexer = FlexIndexer(model_name="qwen3", device="cpu")
            indexer._get_encoder()

        _, kwargs = mock_cls.call_args
        assert "model_name" not in kwargs

    def test_device_and_batch_size_always_win_over_encoder_params(self, mock_pyterrier_dr):
        mock_cls = MagicMock()
        with patch(
            "ragtune.indexing.encoders.resolve_encoder",
            return_value=(mock_cls, {}, None),
        ):
            indexer = FlexIndexer(
                model_name="qwen3",
                device="cuda",
                batch_size=32,
                # Stray device/batch_size inside encoder_params must NOT win.
                device_should_not_apply="ignored",
            )
            indexer.model_kwargs = {"device": "cpu", "batch_size": 999, "use_fp16": True}
            indexer._get_encoder()

        _, kwargs = mock_cls.call_args
        assert kwargs["device"] == "cuda"
        assert kwargs["batch_size"] == 32
        assert kwargs["use_fp16"] is True

    def test_encoder_params_can_override_model_name(self, mock_pyterrier_dr):
        mock_cls = MagicMock()
        with patch(
            "ragtune.indexing.encoders.resolve_encoder",
            return_value=(mock_cls, {}, None),  # bare "qwen3" shorthand resolution
        ):
            indexer = FlexIndexer(model_name="qwen3", device="cpu")
            indexer.model_kwargs = {"model_name": "Qwen/Qwen3-Embedding-8B"}
            indexer._get_encoder()

        _, kwargs = mock_cls.call_args
        assert kwargs["model_name"] == "Qwen/Qwen3-Embedding-8B"

    def test_default_kwargs_from_resolve_encoder_applied(self, mock_pyterrier_dr):
        mock_cls = MagicMock()
        with patch(
            "ragtune.indexing.encoders.resolve_encoder",
            return_value=(mock_cls, {"some_default": "value"}, None),
        ):
            indexer = FlexIndexer(model_name="bge-m3", device="cpu")
            indexer._get_encoder()

        _, kwargs = mock_cls.call_args
        assert kwargs["some_default"] == "value"


class TestSearch:

    def test_search_returns_sorted_top_k(self):
        results_df = pd.DataFrame([
            {"qid": "q1", "docno": "d1", "score": 0.5},
            {"qid": "q1", "docno": "d2", "score": 0.9},
            {"qid": "q1", "docno": "d3", "score": 0.1},
        ])
        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = results_df

        indexer = FlexIndexer(model_name="qwen3")
        with patch.object(indexer, "get_retriever", return_value=mock_pipeline) as mock_get_retriever:
            results = indexer.search("some query", top_k=2, index_path="/tmp/idx")

        mock_get_retriever.assert_called_once_with("/tmp/idx", backend="np")
        assert len(results) == 2
        assert results[0].doc_id == "d2"
        assert results[0].score == pytest.approx(0.9)
        assert results[1].doc_id == "d1"

    def test_search_forwards_custom_backend(self):
        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = pd.DataFrame([{"qid": "q1", "docno": "d1", "score": 1.0}])

        indexer = FlexIndexer(model_name="qwen3")
        with patch.object(indexer, "get_retriever", return_value=mock_pipeline) as mock_get_retriever:
            indexer.search("q", top_k=5, index_path="/tmp/idx", backend="faiss_flat")

        mock_get_retriever.assert_called_once_with("/tmp/idx", backend="faiss_flat")
