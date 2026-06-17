"""
Dense indexer via pyterrier_dr's FlexIndex.

FlexIndex stores document vectors in a flexible format that supports multiple
retrieval backends (NumPy exhaustive, FAISS HNSW, Torch, etc.).

Indexing pipeline:  encoder >> FlexIndex(index_path)
Retrieval pipeline: encoder >> flex_index.np_retriever()

Encoder resolution (see ragtune.indexing.encoders for the registry)
--------------------------------------------------------------------
  "bge-m3", "tasb", "ance", "tct"  → pyterrier_dr built-ins (dense + sparse +
                                      ColBERT / TasB / Ance / TctColBert)
  "qwen3"                          → ragtune.indexing.encoders.Qwen3Encoder
                                      (pyterrier-dr/PyPI has no Qwen3 support)
  <anything else>                  → treated as a raw HuggingFace model ID,
                                      handled by GenericHFEncoder (configurable
                                      pooling, max_length, prefixes, fp16)

Install
-------
    pip install pyterrier-dr python-terrier

Reference
---------
https://github.com/Mandeep-Rathee/pyterrier_dr
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragtune.indexing.base import BaseIndexer, SearchResult
from ragtune.registry import registry


_RETRIEVER_BACKENDS = {"np", "torch", "faiss_hnsw", "faiss_flat"}


def _require_pyterrier_dr():
    try:
        import pyterrier_dr
        return pyterrier_dr
    except ImportError:
        raise ImportError(
            "pyterrier-dr is required for FlexIndexer: pip install pyterrier-dr"
        )


def _require_pyterrier():
    try:
        import pyterrier as pt
        if not pt.started():
            pt.init()
        return pt
    except ImportError:
        raise ImportError(
            "python-terrier is required for FlexIndexer: pip install python-terrier"
        )


@registry.indexer("flex")
class FlexIndexer(BaseIndexer):
    """
    Dense indexer using pyterrier_dr's FlexIndex.

    Parameters
    ----------
    model_name : str
        Encoder key ("qwen3", "bge-m3", "tasb", "ance", "tct") or a raw
        HuggingFace model ID (routed to GenericHFEncoder).
    batch_size : int
        Document encoding batch size.
    device : str
        "cpu" or "cuda".
    model_kwargs : dict
        Extra kwargs forwarded to the resolved encoder's constructor — valid
        keys differ per family (e.g. Qwen3Encoder takes max_length/use_fp16/
        task_description; GenericHFEncoder takes max_length/pooling/
        query_prefix/doc_prefix/fp16).

    Example
    -------
    >>> indexer = FlexIndexer(model_name="qwen3", batch_size=32)
    >>> indexer.build_from_corpus(corpus, "indexes/biology-qwen3")
    >>> retriever = indexer.get_retriever("indexes/biology-qwen3", backend="np")
    >>> # retriever is a PyTerrier transformer: use in pt pipelines or .transform(df)
    """

    def __init__(
        self,
        model_name: str = "qwen3",
        batch_size: int = 32,
        device: str = "cpu",
        **model_kwargs,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model_kwargs = model_kwargs

    def _get_encoder(self):
        _require_pyterrier_dr()
        from ragtune.indexing.encoders import resolve_encoder

        cls, defaults, resolved_model_name = resolve_encoder(self.model_name)
        kwargs = dict(defaults)
        if resolved_model_name is not None:
            kwargs["model_name"] = resolved_model_name
        # encoder_params can add/override family-specific kwargs (max_length,
        # pooling, prefixes, fp16, or even model_name to pick a different
        # checkpoint size) — but never device/batch_size, which are explicit
        # top-level FlexIndexer settings and must stay authoritative.
        kwargs.update(self.model_kwargs)
        kwargs["batch_size"] = self.batch_size
        kwargs["device"] = self.device
        return cls(**kwargs)

    def build_from_corpus(
        self, corpus: Dict[str, Dict], index_path: str, **params
    ) -> bool:
        """
        Encode corpus with the configured encoder and store in a FlexIndex.

        Parameters
        ----------
        corpus : dict
            {doc_id: {"text": ..., "title": ...}}
        index_path : str
            Directory where the FlexIndex is written.
        """
        pt = _require_pyterrier()
        dr = _require_pyterrier_dr()

        index_path = os.path.abspath(index_path)
        # Do NOT pre-create index_path: pyterrier_dr's FlexIndexer.index()
        # treats an already-existing directory (even empty) as "index already
        # exists" and refuses to write, unless mode="overwrite". It creates
        # the directory itself once that check passes.
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # verbose=False disables FlexIndex's own progress bar, which wraps a
        # plain generator with no total — so it can only show count/rate
        # ("27101dvec [01:55, 411.75dvec/s]"), never %/ETA. We already know
        # len(corpus) up front, so we wrap the iterator ourselves with a real
        # total to get a standard tqdm bar (%, elapsed, ETA, rate).
        flex_index = dr.FlexIndex(index_path, verbose=False)
        encoder = self._get_encoder()
        pipeline = encoder >> flex_index

        def _iter_docs():
            for doc_id, doc in corpus.items():
                yield {"docno": str(doc_id), "text": str(doc.get("text", ""))}

        docs = pt.tqdm(_iter_docs(), total=len(corpus), desc="Encoding & indexing", unit="doc")
        pipeline.index(docs)

        with open(os.path.join(index_path, "flex_metadata.json"), "w") as f:
            json.dump(
                {"model_name": self.model_name, "num_docs": len(corpus)}, f
            )

        return True

    def exists(self, index_path: str) -> bool:
        p = Path(index_path)
        return p.is_dir() and any(p.iterdir())

    def load(self, index_path: str) -> Any:
        """Return a FlexIndex object (call .np_retriever() etc. on it)."""
        dr = _require_pyterrier_dr()
        return dr.FlexIndex(os.path.abspath(index_path))

    def search(
        self, query: str, top_k: int, index_path: str, backend: str = "np", **params
    ) -> List[SearchResult]:
        """
        Run a single query against a built FlexIndex and return the top_k hits.

        backend : str
            Retrieval backend — "np" and "faiss_flat" are exact (brute-force);
            "torch" is exact but GPU-oriented; "faiss_hnsw" is approximate.
            Defaults to "np" (exact, no extra deps beyond pyterrier_dr).
        """
        import pandas as pd

        pipeline = self.get_retriever(index_path, backend=backend)
        queries_df = pd.DataFrame([{"qid": "q1", "query": query}])
        res = pipeline.transform(queries_df).sort_values("score", ascending=False).head(top_k)
        return [
            SearchResult(doc_id=str(row["docno"]), score=float(row["score"]))
            for _, row in res.iterrows()
        ]

    def get_retriever(self, index_path: str, backend: str = "np") -> Any:
        """
        Return an end-to-end retrieval pipeline: encoder >> flex_index.<backend>_retriever().

        Parameters
        ----------
        index_path : str
            Path to the built FlexIndex directory.
        backend : str
            One of "np", "torch", "faiss_hnsw", "faiss_flat".
            "np" is the default and requires no extra deps.

        Returns
        -------
        A PyTerrier transformer pipeline ready for .transform(queries_df).
        """
        if backend not in _RETRIEVER_BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}. Choose from: {sorted(_RETRIEVER_BACKENDS)}"
            )
        flex = self.load(index_path)
        encoder = self._get_encoder()
        retriever_fn = getattr(flex, f"{backend}_retriever")
        return encoder >> retriever_fn()
