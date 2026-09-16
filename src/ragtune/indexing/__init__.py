"""
Indexing subpackage for SIR / RAGtune.

Importing this package triggers @registry.indexer(...) decorators for all
built-in indexers, making them available via IndexFactory.create() and
the ragtune CLI.

Registered keys
---------------
  "pyterrier"  → PyTerrierIndexer   (sparse BM25, python-terrier)
  "faiss"      → FaissIndexer       (dense, sentence-transformers + faiss-cpu)
  "numpy"      → NumpyIndexer       (dense, sentence-transformers + plain .npy, no faiss)
  "flex"       → FlexIndexer        (dense, pyterrier_dr: Qwen3, BGE-M3, ...)
  "pyserini"   → PyseriniIndexer    (stub, not yet implemented)

All indexers expose search(query, top_k, index_path) for exact top_k retrieval
(see ragtune.indexing.base.SearchResult).
"""

from ragtune.indexing.base import BaseIndexer, SearchResult
from ragtune.indexing.factory import IndexFactory

# Sparse
from ragtune.indexing import pyterrier_indexer   # noqa: F401  registers "pyterrier"
from ragtune.indexing import pyserini_indexer    # noqa: F401  registers "pyserini"

# Dense
from ragtune.indexing import dense_indexer       # noqa: F401  registers "faiss", "numpy"
from ragtune.indexing import flex_indexer        # noqa: F401  registers "flex"

from ragtune.indexing.pyterrier_indexer import PyTerrierIndexer
from ragtune.indexing.dense_indexer import (
    DenseIndexer,
    FaissIndexer,
    FaissIndexData,
    NumpyIndexer,
    NumpyIndexData,
)
from ragtune.indexing.flex_indexer import FlexIndexer
from ragtune.indexing.pyserini_indexer import PyseriniIndexer

__all__ = [
    "BaseIndexer",
    "SearchResult",
    "IndexFactory",
    "PyTerrierIndexer",
    "DenseIndexer",
    "FaissIndexer",
    "FaissIndexData",
    "NumpyIndexer",
    "NumpyIndexData",
    "FlexIndexer",
    "PyseriniIndexer",
]
