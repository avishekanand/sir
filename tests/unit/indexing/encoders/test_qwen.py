"""
Unit tests for ragtune.indexing.encoders.qwen.

transformers model loading is mocked; pooling math uses real torch tensors
(torch is already a hard dependency) so numeric checks are exact.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from ragtune.indexing.encoders.qwen import Qwen3Encoder, _detailed_instruct, _last_token_pool


class TestLastTokenPool:

    def test_right_padding_picks_last_real_token(self):
        hidden = torch.tensor([
            [[1., 1., 1.], [2., 2., 2.], [0., 0., 0.], [0., 0., 0.]],
            [[3., 3., 3.], [4., 4., 4.], [5., 5., 5.], [6., 6., 6.]],
        ])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        pooled = _last_token_pool(hidden, mask)
        assert torch.allclose(pooled[0], torch.tensor([2., 2., 2.]))
        assert torch.allclose(pooled[1], torch.tensor([6., 6., 6.]))

    def test_left_padding_picks_final_position(self):
        hidden = torch.tensor([
            [[0., 0., 0.], [0., 0., 0.], [1., 1., 1.], [2., 2., 2.]],
            [[0., 0., 0.], [0., 0., 0.], [3., 3., 3.], [4., 4., 4.]],
        ])
        mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        pooled = _last_token_pool(hidden, mask)
        assert torch.allclose(pooled[0], torch.tensor([2., 2., 2.]))
        assert torch.allclose(pooled[1], torch.tensor([4., 4., 4.]))


class TestDetailedInstruct:

    def test_format(self):
        assert _detailed_instruct("Find relevant docs", "what is bm25?") == (
            "Instruct: Find relevant docs\nQuery: what is bm25?"
        )


def _make_mock_transformers(hidden_dim=4):
    def fake_tokenize(batch, **kwargs):
        n = len(batch)
        return {
            "input_ids": torch.ones((n, 3), dtype=torch.long),
            "attention_mask": torch.ones((n, 3), dtype=torch.long),
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


class TestQwen3EncoderEncoding:

    def test_query_gets_instruction_prefix(self):
        mock_transformers, mock_tokenizer, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = Qwen3Encoder(device="cpu")
            encoder.encode_queries(["what is bm25?"])

        called_texts = mock_tokenizer.call_args[0][0]
        assert called_texts[0].startswith("Instruct: ")
        assert "what is bm25?" in called_texts[0]

    def test_doc_has_no_instruction_prefix(self):
        mock_transformers, mock_tokenizer, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = Qwen3Encoder(device="cpu")
            encoder.encode_docs(["bm25 is a ranking function"])

        called_texts = mock_tokenizer.call_args[0][0]
        assert called_texts[0] == "bm25 is a ranking function"

    def test_add_instruction_to_query_false_disables_prefix(self):
        mock_transformers, mock_tokenizer, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = Qwen3Encoder(device="cpu", add_instruction_to_query=False)
            encoder.encode_queries(["what is bm25?"])

        called_texts = mock_tokenizer.call_args[0][0]
        assert called_texts[0] == "what is bm25?"

    def test_output_is_l2_normalized(self):
        mock_transformers, _, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = Qwen3Encoder(device="cpu")
            emb = encoder.encode_docs(["doc one", "doc two"])

        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)

    def test_max_length_forwarded_to_tokenizer(self):
        mock_transformers, mock_tokenizer, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = Qwen3Encoder(device="cpu", max_length=128)
            encoder.encode_docs(["doc"])

        assert mock_tokenizer.call_args[1]["max_length"] == 128

    def test_fp16_calls_model_half(self):
        mock_transformers, _, mock_model = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            Qwen3Encoder(device="cpu", use_fp16=True)
        mock_model.half.assert_called_once()

    def test_empty_input_returns_empty_array(self):
        mock_transformers, _, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = Qwen3Encoder(device="cpu")
            emb = encoder.encode_docs([])
        assert emb.shape == (0, 0)

    def test_default_model_name(self):
        mock_transformers, _, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = Qwen3Encoder(device="cpu")
        assert encoder.model_name == "Qwen/Qwen3-Embedding-0.6B"

    def test_custom_model_name(self):
        mock_transformers, _, _ = _make_mock_transformers()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            encoder = Qwen3Encoder(model_name="Qwen/Qwen3-Embedding-4B", device="cpu")
        assert encoder.model_name == "Qwen/Qwen3-Embedding-4B"
