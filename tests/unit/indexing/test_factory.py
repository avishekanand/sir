"""
Unit tests for ragtune.indexing.factory.IndexFactory.
"""

import pytest
from unittest.mock import MagicMock

# Import the indexing package to trigger all @registry.indexer() decorators.
import ragtune.indexing  # noqa: F401

from ragtune.indexing.factory import IndexFactory
from ragtune.indexing.pyterrier_indexer import PyTerrierIndexer
from ragtune.indexing.dense_indexer import FaissIndexer, NumpyIndexer
from ragtune.indexing.flex_indexer import FlexIndexer
from ragtune.indexing.pyserini_indexer import PyseriniIndexer


class TestIndexFactoryCreate:

    def test_create_pyterrier(self):
        indexer = IndexFactory.create("pyterrier")
        assert isinstance(indexer, PyTerrierIndexer)

    def test_create_faiss_with_model_arg(self):
        indexer = IndexFactory.create("faiss", model_name_or_path="all-MiniLM-L6-v2")
        assert isinstance(indexer, FaissIndexer)
        assert indexer.model_name_or_path == "all-MiniLM-L6-v2"

    def test_create_flex_with_model_name(self):
        indexer = IndexFactory.create("flex", model_name="qwen3")
        assert isinstance(indexer, FlexIndexer)
        assert indexer.model_name == "qwen3"

    def test_create_pyserini(self):
        indexer = IndexFactory.create("pyserini")
        assert isinstance(indexer, PyseriniIndexer)

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown indexer type"):
            IndexFactory.create("does-not-exist")

    def test_error_message_lists_available_types(self):
        with pytest.raises(ValueError) as exc_info:
            IndexFactory.create("nope")
        msg = str(exc_info.value)
        assert "pyterrier" in msg
        assert "faiss" in msg


class TestIndexFactoryFromConfig:

    def test_sparse_type_returns_pyterrier(self):
        mock_config = MagicMock()
        mock_config.type = "sparse"

        indexer = IndexFactory.from_config(mock_config)
        assert isinstance(indexer, PyTerrierIndexer)

    def test_dense_faiss_passes_model_and_params(self):
        mock_config = MagicMock()
        mock_config.type = "dense"
        mock_config.backend = "faiss"
        mock_config.model.name = "my-model"
        mock_config.params = {"device": "cpu", "batch_size": 32}

        indexer = IndexFactory.from_config(mock_config)
        assert isinstance(indexer, FaissIndexer)
        assert indexer.model_name_or_path == "my-model"
        assert indexer.batch_size == 32

    def test_dense_numpy_passes_model_name_or_path(self):
        """Regression: numpy backend must use model_name_or_path, not model_name
        (NumpyIndexer is a DenseIndexer subclass, like FaissIndexer — not FlexIndexer)."""
        mock_config = MagicMock()
        mock_config.type = "dense"
        mock_config.backend = "numpy"
        mock_config.model.name = "my-model"
        mock_config.params = {"device": "cpu", "batch_size": 32}

        indexer = IndexFactory.from_config(mock_config)
        assert isinstance(indexer, NumpyIndexer)
        assert indexer.model_name_or_path == "my-model"
        assert indexer.batch_size == 32

    def test_dense_flex_passes_model_name(self):
        mock_config = MagicMock()
        mock_config.type = "dense"
        mock_config.backend = "flex"
        mock_config.model.name = "qwen3"
        mock_config.params = {"device": "cpu", "batch_size": 32}

        indexer = IndexFactory.from_config(mock_config)
        assert isinstance(indexer, FlexIndexer)
        assert indexer.model_name == "qwen3"

    def test_dense_forwards_arbitrary_params(self):
        mock_config = MagicMock()
        mock_config.type = "dense"
        mock_config.backend = "flex"
        mock_config.model.name = "qwen3"
        mock_config.params = {"device": "cuda", "batch_size": 16, "max_length": 8192, "use_fp16": True}

        indexer = IndexFactory.from_config(mock_config)
        assert indexer.device == "cuda"
        assert indexer.batch_size == 16
        assert indexer.model_kwargs == {"max_length": 8192, "use_fp16": True}

    def test_dense_missing_backend_raises(self):
        mock_config = MagicMock()
        mock_config.type = "dense"
        mock_config.backend = None

        with pytest.raises(ValueError, match="index.backend is required"):
            IndexFactory.from_config(mock_config)

    def test_dense_missing_model_raises(self):
        mock_config = MagicMock()
        mock_config.type = "dense"
        mock_config.backend = "faiss"
        mock_config.model = None

        with pytest.raises(ValueError, match="index.model.name is required"):
            IndexFactory.from_config(mock_config)

    def test_dense_missing_model_name_raises(self):
        mock_config = MagicMock()
        mock_config.type = "dense"
        mock_config.backend = "faiss"
        mock_config.model.name = None

        with pytest.raises(ValueError, match="index.model.name is required"):
            IndexFactory.from_config(mock_config)

    def test_unknown_type_raises(self):
        mock_config = MagicMock()
        mock_config.type = "hybrid"

        with pytest.raises(ValueError, match="Unknown index.type"):
            IndexFactory.from_config(mock_config)
