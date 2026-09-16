"""
Sparse BM25 indexer via PyTerrier's IterDictIndexer.

Takes a BEIR-style corpus dict (from any DataLoader) and builds a standard
Terrier inverted index on disk.  The resulting index is compatible with
PyTerrierRetriever in ragtune.adapters.pyterrier.

Install
-------
    pip install python-terrier
"""

import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import pyterrier as pt
except ImportError:
    pt = None

from ragtune.indexing.base import BaseIndexer, SearchResult
from ragtune.registry import registry


@registry.indexer("pyterrier")
class PyTerrierIndexer(BaseIndexer):
    """
    Builds a PyTerrier inverted index from a BEIR-style corpus dict.

    Example
    -------
    >>> indexer = PyTerrierIndexer()
    >>> indexer.build_from_corpus(corpus, index_path="indexes/biology-bm25")
    >>> pt_index = indexer.load("indexes/biology-bm25")
    >>> # Use with PyTerrierRetriever:
    >>> retriever = PyTerrierRetriever(index_path="indexes/biology-bm25")
    """

    def build_from_corpus(
        self, corpus: Dict[str, Dict], index_path: str, **params
    ) -> bool:
        """
        Index a BEIR-style corpus dict with IterDictIndexer.

        Parameters
        ----------
        corpus : dict
            {doc_id: {"text": ..., "title": ...}}
        index_path : str
            Directory where the Terrier index is written.
        """
        if pt is None:
            raise ImportError(
                "python-terrier is required for PyTerrierIndexer: "
                "pip install python-terrier"
            )
        if not pt.started():
            pt.init()

        index_path = os.path.abspath(index_path)
        os.makedirs(index_path, exist_ok=True)

        def _iter_docs():
            for doc_id, doc in corpus.items():
                yield {
                    "docno": str(doc_id),
                    "text": str(doc.get("text", "")),
                    "title": str(doc.get("title", "")),
                }

        indexer = pt.IterDictIndexer(
            index_path,
            overwrite=True,
            meta={"docno": 5000, "text": 50000},
        )
        indexer.index(_iter_docs())
        return True

    def exists(self, index_path: str) -> bool:
        p = Path(index_path)
        return p.is_dir() and (p / "data.properties").exists()

    def load(self, index_path: str) -> Any:
        """Return a pt.IndexBase ready for BatchRetrieve / terrier.Retriever."""
        if pt is None:
            raise ImportError(
                "python-terrier is required for PyTerrierIndexer: "
                "pip install python-terrier"
            )
        if not pt.started():
            pt.init()
        return pt.IndexFactory.of(os.path.abspath(index_path))

    def search(self, query: str, top_k: int, index_path: str, **params) -> List[SearchResult]:
        """Run BM25 retrieval for a single query and return the top_k hits."""
        if pt is None:
            raise ImportError(
                "python-terrier is required for PyTerrierIndexer: "
                "pip install python-terrier"
            )
        if not pt.started():
            pt.init()
        import pandas as pd

        bm25 = pt.terrier.Retriever(os.path.abspath(index_path), wmodel="BM25")
        queries_df = pd.DataFrame([{"qid": "q1", "query": query}])
        res = bm25.transform(queries_df).sort_values("score", ascending=False).head(top_k)
        return [
            SearchResult(doc_id=str(row["docno"]), score=float(row["score"]))
            for _, row in res.iterrows()
        ]
