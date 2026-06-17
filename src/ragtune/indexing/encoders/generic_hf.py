"""
Generic HuggingFace bi-encoder — fallback for arbitrary HF model IDs.

pyterrier_dr's HgfBiEncoder.from_pretrained() is rigid: CLS-pooling only,
no max_length control, no instruction prefixes, and its constructor rejects
extra kwargs (e.g. passing max_length raises TypeError). GenericHFEncoder
loads model + tokenizer directly via transformers and exposes the knobs
that matter for embedding models without a dedicated family file: pooling
strategy, truncation length, normalization, fp16, and optional prefixes.

Prefer a dedicated encoder file (like qwen.py) over this one whenever a
model family needs nonstandard handling (asymmetric instructions, special
tokens, long-context defaults, multi-vector output, etc).
"""

from typing import List, Optional

import numpy as np
import pyterrier_dr

_POOLING_STRATEGIES = {"mean", "cls", "last_token"}


class GenericHFEncoder(pyterrier_dr.BiEncoder):
    """
    Generic HuggingFace bi-encoder with configurable pooling / truncation / prefixes.

    Parameters
    ----------
    model_name : str
        Any HuggingFace model ID loadable via AutoModel/AutoTokenizer.
    max_length : int
        Max token length for truncation.
    pooling : str
        "mean" (default) | "cls" | "last_token"
    normalize : bool
        L2-normalize output embeddings.
    query_prefix / doc_prefix : str
        Optional text prepended to queries / documents before encoding.
    fp16 : bool
        Load model weights in half precision (GPU only).

    Example
    -------
    >>> encoder = GenericHFEncoder("intfloat/e5-large-v2", max_length=256,
    ...                             query_prefix="query: ", doc_prefix="passage: ")
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        pooling: str = "mean",
        normalize: bool = True,
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 32,
        device: Optional[str] = None,
        fp16: bool = False,
        text_field: str = "text",
        verbose: bool = False,
    ):
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)

        if pooling not in _POOLING_STRATEGIES:
            raise ValueError(
                f"pooling must be one of {_POOLING_STRATEGIES}, got {pooling!r}"
            )

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModel.from_pretrained(model_name)
        if fp16:
            model = model.half()
        self.model = model.to(self.device).eval()

    def _pool(self, last_hidden_state, attention_mask):
        import torch

        if self.pooling == "cls":
            return last_hidden_state[:, 0]

        if self.pooling == "last_token":
            seq_lens = attention_mask.sum(dim=1) - 1
            return last_hidden_state[torch.arange(last_hidden_state.shape[0]), seq_lens]

        # mean pooling — masks out padding tokens
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode(self, texts: List[str], prefix: str, batch_size: Optional[int]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        bs = batch_size or self.batch_size
        results = []
        with torch.no_grad():
            for start in range(0, len(texts), bs):
                batch = [prefix + t for t in texts[start : start + bs]]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                out = self.model(**inputs)
                emb = self._pool(out.last_hidden_state, inputs["attention_mask"])
                if self.normalize:
                    emb = F.normalize(emb, p=2, dim=1)
                results.append(emb.cpu().numpy())
        if not results:
            return np.empty((0, 0))
        return np.concatenate(results, axis=0)

    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        return self._encode(list(texts), self.query_prefix, batch_size)

    def encode_docs(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        return self._encode(list(texts), self.doc_prefix, batch_size)

    def __repr__(self):
        return f"GenericHFEncoder({self.model_name!r}, pooling={self.pooling!r})"
