"""
``autoindex`` indexer: build a BM25 index through a learned representation program.

The program runs once at build time; no LLM calls. The index directory holds
the program source, the chunks, and the BM25 settings. ``load()`` rebuilds BM25
from the stored chunks (the BM25 matrix itself isn't persisted).

    indexer = IndexFactory.create("autoindex", program_path="runs/tr/best_program.py")
    indexer.build_from_corpus(corpus, index_path="indexes/tr-autoindex")
    hits = indexer.search("query", top_k=10, index_path="indexes/tr-autoindex")
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragtune.indexing.autoindex.bm25 import ChunkBM25Index
from ragtune.indexing.autoindex.program import (
    IDENTITY_PROGRAM_CODE,
    Chunk,
    corpus_to_documents,
    load_program,
    run_program,
)
from ragtune.indexing.base import BaseIndexer, SearchResult
from ragtune.registry import registry

PROGRAM_FILE = "program.py"
CHUNKS_FILE = "chunks.jsonl"
CONFIG_FILE = "autoindex.json"


@registry.indexer("autoindex")
class AutoIndexBM25Indexer(BaseIndexer):
    """BM25 (MaxP) over chunks produced by a representation program.

    Pass the program as ``program_code`` or ``program_path``. Without either it
    uses the identity program, i.e. plain full-document BM25.
    """

    def __init__(
        self,
        program_code: Optional[str] = None,
        program_path: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
        candidate_k: int = 1000,
        backend: str = "auto",
        preprocess_timeout_s: Optional[float] = None,
    ):
        if program_code and program_path:
            raise ValueError("pass program_code or program_path, not both")
        if program_path:
            program_code = Path(program_path).read_text(encoding="utf-8")
        self.program_code = program_code or IDENTITY_PROGRAM_CODE
        self.k1, self.b, self.candidate_k, self.backend = k1, b, candidate_k, backend
        self.preprocess_timeout_s = preprocess_timeout_s
        self._cache: Dict[str, ChunkBM25Index] = {}

    def build_from_corpus(self, corpus: Dict[str, Dict], index_path: str, **params) -> bool:
        program = load_program(self.program_code)
        chunks = run_program(program, corpus_to_documents(corpus), timeout_s=self.preprocess_timeout_s)

        out = Path(index_path)
        out.mkdir(parents=True, exist_ok=True)
        (out / PROGRAM_FILE).write_text(self.program_code, encoding="utf-8")
        with (out / CHUNKS_FILE).open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        (out / CONFIG_FILE).write_text(json.dumps({
            "k1": self.k1, "b": self.b, "candidate_k": self.candidate_k,
            "n_docs": len(corpus), "n_chunks": len(chunks),
        }, indent=2), encoding="utf-8")
        self._cache[str(out)] = self._make_index(chunks)
        return True

    def _make_index(self, chunks: List[Chunk], settings: Optional[dict] = None) -> ChunkBM25Index:
        s = settings or {}
        return ChunkBM25Index(chunks, k1=s.get("k1", self.k1), b=s.get("b", self.b),
                              candidate_k=s.get("candidate_k", self.candidate_k), backend=self.backend)

    def exists(self, index_path: str) -> bool:
        p = Path(index_path)
        return (p / CHUNKS_FILE).exists() and (p / CONFIG_FILE).exists()

    def load(self, index_path: str) -> Any:
        key = str(Path(index_path))
        if key not in self._cache:
            if not self.exists(index_path):
                raise FileNotFoundError(f"no autoindex index at {index_path}")
            p = Path(index_path)
            settings = json.loads((p / CONFIG_FILE).read_text(encoding="utf-8"))
            with (p / CHUNKS_FILE).open(encoding="utf-8") as f:
                chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
            self._cache[key] = self._make_index(chunks, settings)
        return self._cache[key]

    def search(self, query: str, top_k: int, index_path: str, **params) -> List[SearchResult]:
        return [SearchResult(doc_id=h.doc_id, score=h.score) for h in self.load(index_path).search(query, top_k)]
