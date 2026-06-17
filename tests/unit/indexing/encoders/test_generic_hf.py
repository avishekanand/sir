"""
Unit tests for ragtune.indexing.encoders.generic_hf.GenericHFEncoder.

transformers model loading is mocked; pooling math uses real torch tensors.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from ragtune.indexing.encoders.generic_hf import GenericHFEncoder


def _make_mock_transformers(hidden_dim=4):
    def fake_tokenize(batch, **kwargs):
        n = len(batch)
        return {
            "input_ids": torch.ones((n, 3), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]] * n, dtype=torch.long),
        }

    mock_tokenizer = MagicMock(side_effect=fake_tokenize)
    mock_tokenizer_cls = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    def fake_forward(**inputs):
        n, seq_len = inputs["input_ids"].shape
        out = MagicMock()
        out.last_hidden_state = torch.randn(n, seq_len, hidden_dim)
        return out

    mock_model = MagicMock(side_effect=fake_forward)
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model.half.return_value = mock_model

    mock_model_cls = MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_model

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer = mock_tokenizer_cls
    mock_transformers.AutoModel = mock_model_cls
    return mock_transformers, mock_tokenizer, mock_model


class TestPoolingStrategies:

    def test_invalid_pooling_raises(self):
        mock_transformers, _, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            with pytest.raises(ValueError, match="pooling must be one of"):
                GenericHFEncoder("fake-model", pooling="bogus", device="cpu")

    def test_cls_pooling_takes_first_token(self):
        hidden = torch.tensor([[[1., 1.], [9., 9.], [9., 9.]]])
        mask = torch.tensor([[1, 1, 0]])
        mock_transformers, _, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = GenericHFEncoder("fake-model", pooling="cls", device="cpu")
        pooled = encoder._pool(hidden, mask)
        assert torch.allclose(pooled[0], torch.tensor([1., 1.]))

    def test_mean_pooling_ignores_padding(self):
        hidden = torch.tensor([[[2., 2.], [4., 4.], [100., 100.]]])
        mask = torch.tensor([[1, 1, 0]])
        mock_transformers, _, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = GenericHFEncoder("fake-model", pooling="mean", device="cpu")
        pooled = encoder._pool(hidden, mask)
        assert torch.allclose(pooled[0], torch.tensor([3., 3.]))  # mean of [2,2] and [4,4]

    def test_last_token_pooling_uses_attention_mask(self):
        hidden = torch.tensor([[[1., 1.], [2., 2.], [9., 9.]]])
        mask = torch.tensor([[1, 1, 0]])
        mock_transformers, _, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = GenericHFEncoder("fake-model", pooling="last_token", device="cpu")
        pooled = encoder._pool(hidden, mask)
        assert torch.allclose(pooled[0], torch.tensor([2., 2.]))


class TestEncodePrefixesAndOptions:

    def test_query_and_doc_prefixes_applied_separately(self):
        mock_transformers, mock_tokenizer, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = GenericHFEncoder(
                "fake-model", query_prefix="query: ", doc_prefix="passage: ", device="cpu"
            )
            encoder.encode_queries(["foo"])
            query_call = mock_tokenizer.call_args[0][0]
            encoder.encode_docs(["bar"])
            doc_call = mock_tokenizer.call_args[0][0]

        assert query_call[0] == "query: foo"
        assert doc_call[0] == "passage: bar"

    def test_max_length_forwarded(self):
        mock_transformers, mock_tokenizer, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = GenericHFEncoder("fake-model", max_length=64, device="cpu")
            encoder.encode_docs(["doc"])
        assert mock_tokenizer.call_args[1]["max_length"] == 64

    def test_normalize_true_yields_unit_norm(self):
        mock_transformers, _, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = GenericHFEncoder("fake-model", normalize=True, device="cpu")
            emb = encoder.encode_docs(["doc one", "doc two"])
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)

    def test_normalize_false_skips_normalization(self):
        mock_transformers, _, mock_model = _make_mock_transformers()

        def fake_forward(**inputs):
            n, seq_len = inputs["input_ids"].shape
            out = MagicMock()
            out.last_hidden_state = torch.full((n, seq_len, 4), 2.0)
            return out

        mock_model.side_effect = fake_forward
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = GenericHFEncoder("fake-model", normalize=False, pooling="mean", device="cpu")
            emb = encoder.encode_docs(["doc"])
        norms = np.linalg.norm(emb, axis=1)
        assert not np.allclose(norms, 1.0, atol=1e-5)

    def test_fp16_calls_model_half(self):
        mock_transformers, _, mock_model = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            GenericHFEncoder("fake-model", fp16=True, device="cpu")
        mock_model.half.assert_called_once()

    def test_empty_input_returns_empty_array(self):
        mock_transformers, _, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = GenericHFEncoder("fake-model", device="cpu")
            emb = encoder.encode_docs([])
        assert emb.shape == (0, 0)
