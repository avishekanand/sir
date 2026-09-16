"""
Base abstraction for all indexers in the SIR pipeline.

Concrete indexers implement build_from_corpus() (takes a BEIR-style corpus dict
directly from any DataLoader) and load() (returns a live index object for retrieval).

The file-based build() is a default implementation that parses the file into a
corpus dict and delegates to build_from_corpus(), so subclasses only need to
override build_from_corpus().
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List


@dataclass
class SearchResult:
    """One scored hit returned by BaseIndexer.search()."""
    doc_id: str
    score: float


class BaseIndexer(ABC):

    # ------------------------------------------------------------------
    # Primary API — speaks directly to DataLoader.get_corpus() output
    # ------------------------------------------------------------------

    @abstractmethod
    def build_from_corpus(
        self,
        corpus: Dict[str, Dict],  # {doc_id: {"text": ..., "title": ...}}
        index_path: str,
        **params,
    ) -> bool:
        """Build an index from a BEIR-style in-memory corpus dict."""

    # ------------------------------------------------------------------
    # Secondary API — file-based (parses file, then calls build_from_corpus)
    # ------------------------------------------------------------------

    def build(
        self,
        collection_path: str,
        format: str,
        fields: Dict[str, str],
        **params,
    ) -> bool:
        """Build from a JSON/JSONL file; delegates to build_from_corpus."""
        corpus = self._load_file_to_corpus(collection_path, format, fields)
        return self.build_from_corpus(corpus, **params)

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def exists(self, index_path: str) -> bool:
        """Return True if a built index already exists at index_path."""

    @abstractmethod
    def load(self, index_path: str) -> Any:
        """Load and return a live index object ready for retrieval."""

    def search(self, query: str, top_k: int, index_path: str, **params) -> List[SearchResult]:
        """Run a single query against a built index and return top_k hits."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement search().")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_file_to_corpus(
        self, collection_path: str, format: str, fields: Dict[str, str]
    ) -> Dict[str, Dict]:
        """Parse a JSON/JSONL file into a BEIR-style corpus dict."""
        id_col = fields.get("id_field", "doc_id")
        text_col = fields.get("text_field", "text")
        title_col = fields.get("title_field", "title")

        corpus: Dict[str, Dict] = {}
        for raw in self._iter_file(collection_path, format):
            doc_id = str(raw.get(id_col, ""))
            corpus[doc_id] = {
                "text": str(raw.get(text_col, "")),
                "title": str(raw.get(title_col, "")),
            }
        return corpus

    @staticmethod
    def _iter_file(path: str, format: str) -> Iterator[Dict]:
        with open(path) as f:
            if format == "jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            elif format == "json":
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                yield from items
            else:
                raise NotImplementedError(f"Unsupported format: {format!r}")
