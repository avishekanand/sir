"""
Pyserini indexer — stub for future implementation.

Pyserini wraps Anserini (Java) for BM25/sparse indexing and also provides
FAISS-based dense retrieval with DPR, ANCE, and other bi-encoder models.

Install
-------
    pip install pyserini
    (requires JDK 11+)

Reference
---------
https://github.com/castorini/pyserini
"""

from typing import Any, Dict

from ragtune.indexing.base import BaseIndexer
from ragtune.registry import registry


@registry.indexer("pyserini")
class PyseriniIndexer(BaseIndexer):
    """
    BM25 / sparse indexer via Pyserini (Anserini-backed).

    Not yet implemented — placeholder registered in the registry so that
    config validation can surface it as a known type.
    """

    def build_from_corpus(
        self, corpus: Dict[str, Dict], index_path: str, **params
    ) -> bool:
        raise NotImplementedError(
            "PyseriniIndexer is not yet implemented. "
            "Track progress on branch: feat/pyserini-indexer"
        )

    def exists(self, index_path: str) -> bool:
        raise NotImplementedError("PyseriniIndexer is not yet implemented.")

    def load(self, index_path: str) -> Any:
        raise NotImplementedError("PyseriniIndexer is not yet implemented.")
