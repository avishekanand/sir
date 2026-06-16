"""
Abstract base class for all data loaders in the SIR pipeline.

Every loader exposes:
  - raw_data          : List[Sample]           (query + relevant doc pairs)
  - get_corpus()      : Dict[str, Dict]        BEIR-style {doc_id: {text, title}}
  - get_queries()     : Dict[str, str]         {query_id: query_text}
  - get_qrels()       : Dict[str, Dict[str,int]] {query_id: {doc_id: relevance}}
  - get_query_objects(): List[Query]
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from src.ragtune.data.datastructures import Query, Context, Sample


class BaseDataLoader(ABC):
    """
    Abstract base class all dataset loaders must inherit from.

    Subclasses implement `_load_data()` which populates `self.raw_data`
    as a list of (query, document, relevance) triples encoded as Sample objects.
    """

    def __init__(self, dataset: str, split: str, **kwargs):
        self.dataset = dataset
        self.split = split
        self.raw_data: List[Sample] = []
        self._corpus: Dict[str, Dict] = {}      # {doc_id: {text, title}}
        self._queries: Dict[str, str] = {}      # {query_id: text}
        self._qrels: Dict[str, Dict[str, int]] = {}  # {qid: {doc_id: rel}}
        self._loaded = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_data(self) -> None:
        """Load and populate self.raw_data, self._corpus, self._queries,
        and self._qrels.  Called lazily on first access."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._loaded:
            self._load_data()
            self._loaded = True

    # ------------------------------------------------------------------
    # Public accessors (BEIR-compatible interface)
    # ------------------------------------------------------------------

    def get_corpus(self) -> Dict[str, Dict]:
        """Return corpus as {doc_id: {'text': ..., 'title': ...}}."""
        self._ensure_loaded()
        return self._corpus

    def get_queries(self) -> Dict[str, str]:
        """Return queries as {query_id: query_text}."""
        self._ensure_loaded()
        return self._queries

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """Return qrels as {query_id: {doc_id: relevance_score}}."""
        self._ensure_loaded()
        return self._qrels

    def get_query_objects(self) -> List[Query]:
        """Return list of Query objects (with IDs)."""
        self._ensure_loaded()
        seen = {}
        for sample in self.raw_data:
            qid = sample.query.id()
            if qid not in seen:
                seen[qid] = sample.query
        return list(seen.values())

    def load(self) -> Tuple[Dict, Dict, Dict]:
        """
        Convenience method: returns (corpus, queries, qrels) matching
        the BEIR GenericDataLoader.load() signature.
        """
        self._ensure_loaded()
        return self._corpus, self._queries, self._qrels

    def __len__(self):
        self._ensure_loaded()
        return len(self._queries)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dataset={self.dataset!r}, split={self.split!r}, "
            f"queries={len(self._queries)}, docs={len(self._corpus)})"
        )
