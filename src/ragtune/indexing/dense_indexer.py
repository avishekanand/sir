"""
Dense (vector) indexers: abstract base plus two concrete storage backends.

DenseIndexer is a template: it owns query/document encoding (sentence-
transformers by default), batching, optional L2 normalisation, and the
build_from_corpus()/search() orchestration. Concrete subclasses only
implement how vectors are persisted/loaded/searched:

  FaissIndexer  → faiss.IndexFlatIP on disk (exact, requires faiss-cpu)
  NumpyIndexer  → plain .npy matrix on disk (exact, zero extra dependencies)

Both are exact (brute-force) search — no approximate-NN tradeoffs.

Install
-------
    pip install sentence-transformers
    pip install faiss-cpu   # only needed for FaissIndexer
"""

import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ragtune.indexing.base import BaseIndexer, SearchResult
from ragtune.registry import registry


@dataclass
class FaissIndexData:
    """Wrapper returned by FaissIndexer.load()."""
    index: Any            # faiss.Index
    docnos: List[str]     # positional match to FAISS rows
    model_name: str
    index_path: str


@dataclass
class NumpyIndexData:
    """Wrapper returned by NumpyIndexer.load()."""
    vectors: np.ndarray   # [N, dim] float32
    docnos: List[str]     # positional match to vectors rows
    model_name: str
    index_path: str


class DenseIndexer(BaseIndexer):
    """
    Template base for embedding-based indexers.

    Handles encoding (sentence-transformers by default — override
    encode_corpus() for a different encoder), batching, normalisation, and
    the build/search orchestration. Subclasses only implement the storage
    hooks: _save_vectors(), exists(), load(), _search_vectors().

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path used by the encoder.
    batch_size : int
        Number of documents encoded per forward pass.
    device : str
        "cpu" or "cuda".
    normalize : bool
        L2-normalise vectors before storing (makes inner product ≡ cosine sim).
    max_length : int | None
        Forwarded to the sentence-transformers model's max_seq_length.
    """

    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str = "cpu",
        normalize: bool = True,
        max_length: Optional[int] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self.max_length = max_length
        self._model = None  # lazy-loaded on first encode

    def encode_corpus(self, texts: List[str]) -> np.ndarray:
        """Default encoder: sentence-transformers. Override for a different one."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name_or_path, device=self.device)
            if self.max_length is not None:
                self._model.max_seq_length = self.max_length
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def _encode_in_batches(self, texts: List[str]) -> np.ndarray:
        from tqdm import tqdm

        chunks = []
        with tqdm(total=len(texts), desc="Encoding", unit="doc") as pbar:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                chunks.append(self.encode_corpus(batch))
                pbar.update(len(batch))
        vectors = np.vstack(chunks).astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors /= norms
        return vectors

    def build_from_corpus(
        self, corpus: Dict[str, Dict], index_path: str, **params
    ) -> bool:
        """Encode corpus and persist via the backend-specific _save_vectors()."""
        index_path = os.path.abspath(index_path)
        os.makedirs(index_path, exist_ok=True)

        docnos = list(corpus.keys())
        texts = [corpus[d].get("text", "") for d in docnos]
        vectors = self._encode_in_batches(texts)

        self._save_vectors(vectors, docnos, index_path)
        return True

    @abstractmethod
    def _save_vectors(self, vectors: np.ndarray, docnos: List[str], index_path: str) -> None:
        """Persist vectors + docnos to index_path. Backend-specific."""

    @abstractmethod
    def exists(self, index_path: str) -> bool:
        """Return True if a built index already exists at index_path."""

    @abstractmethod
    def load(self, index_path: str) -> Any:
        """Load and return a backend-specific index data object."""

    def search(self, query: str, top_k: int, index_path: str, **params) -> List[SearchResult]:
        """Encode query, load the index, and run exact top_k search."""
        data = self.load(index_path)
        query_vec = self._encode_in_batches([query])[0]
        return self._search_vectors(query_vec, data, top_k)

    @abstractmethod
    def _search_vectors(self, query_vec: np.ndarray, data: Any, top_k: int) -> List[SearchResult]:
        """Exact nearest-neighbor search against the loaded index data. Backend-specific."""


@registry.indexer("faiss")
class FaissIndexer(DenseIndexer):
    """
    Dense indexer: sentence-transformers encoding + FAISS flat inner-product index.

    Storage layout written to index_path/
        index.faiss      FAISS IndexFlatIP (vectors are L2-normalised)
        docnos.json      ordered list of doc IDs (positional match to FAISS rows)
        metadata.json    model_name, dim, num_docs

    Example
    -------
    >>> indexer = FaissIndexer("sentence-transformers/all-MiniLM-L6-v2")
    >>> indexer.build_from_corpus(corpus, "indexes/biology-dense")
    >>> indexer.search("what is bm25?", top_k=10, index_path="indexes/biology-dense")
    """

    def _save_vectors(self, vectors: np.ndarray, docnos: List[str], index_path: str) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FaissIndexer: pip install faiss-cpu"
            )
        dim = vectors.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(vectors)

        faiss.write_index(faiss_index, os.path.join(index_path, "index.faiss"))
        with open(os.path.join(index_path, "docnos.json"), "w") as f:
            json.dump(docnos, f)
        with open(os.path.join(index_path, "metadata.json"), "w") as f:
            json.dump(
                {"model_name": self.model_name_or_path, "dim": dim, "num_docs": len(docnos)},
                f,
            )

    def exists(self, index_path: str) -> bool:
        p = Path(index_path)
        return (p / "index.faiss").exists() and (p / "docnos.json").exists()

    def load(self, index_path: str) -> FaissIndexData:
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FaissIndexer: pip install faiss-cpu"
            )
        index_path = os.path.abspath(index_path)
        faiss_index = faiss.read_index(os.path.join(index_path, "index.faiss"))
        with open(os.path.join(index_path, "docnos.json")) as f:
            docnos = json.load(f)
        with open(os.path.join(index_path, "metadata.json")) as f:
            meta = json.load(f)
        return FaissIndexData(
            index=faiss_index,
            docnos=docnos,
            model_name=meta.get("model_name", self.model_name_or_path),
            index_path=index_path,
        )

    def _search_vectors(
        self, query_vec: np.ndarray, data: FaissIndexData, top_k: int
    ) -> List[SearchResult]:
        scores, indices = data.index.search(query_vec.reshape(1, -1).astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(doc_id=data.docnos[idx], score=float(score)))
        return results


@registry.indexer("numpy")
class NumpyIndexer(DenseIndexer):
    """
    Dense indexer: sentence-transformers encoding + plain numpy matrix storage.

    No FAISS dependency — exact brute-force search via matrix multiplication.
    Good for small/medium corpora or environments without faiss-cpu.

    Storage layout written to index_path/
        vectors.npy      float32 matrix [N, dim] (rows are L2-normalised)
        docnos.json       ordered list of doc IDs (positional match to rows)
        metadata.json     model_name, dim, num_docs

    Example
    -------
    >>> indexer = NumpyIndexer("sentence-transformers/all-MiniLM-L6-v2")
    >>> indexer.build_from_corpus(corpus, "indexes/biology-numpy")
    >>> indexer.search("what is bm25?", top_k=10, index_path="indexes/biology-numpy")
    """

    def _save_vectors(self, vectors: np.ndarray, docnos: List[str], index_path: str) -> None:
        np.save(os.path.join(index_path, "vectors.npy"), vectors)
        with open(os.path.join(index_path, "docnos.json"), "w") as f:
            json.dump(docnos, f)
        with open(os.path.join(index_path, "metadata.json"), "w") as f:
            json.dump(
                {
                    "model_name": self.model_name_or_path,
                    "dim": vectors.shape[1],
                    "num_docs": len(docnos),
                },
                f,
            )

    def exists(self, index_path: str) -> bool:
        p = Path(index_path)
        return (p / "vectors.npy").exists() and (p / "docnos.json").exists()

    def load(self, index_path: str) -> NumpyIndexData:
        index_path = os.path.abspath(index_path)
        vectors = np.load(os.path.join(index_path, "vectors.npy"))
        with open(os.path.join(index_path, "docnos.json")) as f:
            docnos = json.load(f)
        with open(os.path.join(index_path, "metadata.json")) as f:
            meta = json.load(f)
        return NumpyIndexData(
            vectors=vectors,
            docnos=docnos,
            model_name=meta.get("model_name", self.model_name_or_path),
            index_path=index_path,
        )

    def _search_vectors(
        self, query_vec: np.ndarray, data: NumpyIndexData, top_k: int
    ) -> List[SearchResult]:
        scores = data.vectors @ query_vec
        top_indices = np.argsort(-scores)[:top_k]
        return [SearchResult(doc_id=data.docnos[i], score=float(scores[i])) for i in top_indices]
