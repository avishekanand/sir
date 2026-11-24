"""
Chunk-level BM25 with MaxP document aggregation: AutoIndex's fixed retriever.

Scoring is Lucene BM25 (k1=1.5, b=0.75 by default):

    idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
    s(q, c)  = Σ_{t∈q} idf(t) · tf / (tf + k1 · (1 - b + b · |c| / avgdl))

A query scores the top ``candidate_k`` chunks, then each document takes the
maximum score over its chunks (MaxP). Uses ``bm25s`` when it's installed and a
pure-Python inverted index otherwise; both give the same scores.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ragtune.indexing.autoindex.program import Chunk

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


@dataclass
class DocHit:
    doc_id: str
    score: float
    rank: int


class Tokenizer:
    """Lowercase + regex tokenization, with optional stopwords and stemmer.

    The default (no stopwords, no stemmer) matches the reference implementation's
    ``bm25s.tokenize`` call.
    """

    def __init__(
        self,
        stopwords: Optional[Iterable[str]] = None,
        stemmer: Optional[Callable[[str], str]] = None,
    ):
        self.stopwords = frozenset(stopwords or ())
        self.stemmer = stemmer

    def __call__(self, text: str) -> List[str]:
        toks = [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in self.stopwords]
        if self.stemmer is not None:
            toks = [self.stemmer(t) for t in toks]
        return toks

    def describe(self) -> str:
        parts = ["lowercased", "regex tokens of 2+ word characters"]
        parts.append("stemmed" if self.stemmer else "NOT stemmed")
        parts.append("stopwords removed" if self.stopwords else "stopwords NOT removed")
        return ", ".join(parts)


def _has_bm25s() -> bool:
    try:
        import bm25s  # noqa: F401
        return True
    except ImportError:
        return False


class ChunkBM25Index:
    """BM25 over chunks, searched at document level with MaxP."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Tokenizer] = None,
        candidate_k: int = 1000,
        backend: str = "auto",
    ):
        if not chunks:
            raise ValueError("cannot build a BM25 index over zero chunks")
        if backend not in ("auto", "bm25s", "python"):
            raise ValueError(f"unknown backend {backend!r}")
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer or Tokenizer()
        self.candidate_k = candidate_k
        self.chunk_doc_ids = [c.doc_id for c in chunks]
        self.n_chunks = len(chunks)
        self.n_docs = len(set(self.chunk_doc_ids))

        corpus_tokens = [self.tokenizer(c.text) for c in chunks]
        use_bm25s = backend == "bm25s" or (backend == "auto" and _has_bm25s())
        self.backend = "bm25s" if use_bm25s else "python"
        if use_bm25s:
            self._build_bm25s(corpus_tokens)
        else:
            self._build_python(corpus_tokens)

    # -- pure-Python backend ------------------------------------------------

    def _build_python(self, corpus_tokens: List[List[str]]) -> None:
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._lengths = [len(t) for t in corpus_tokens]
        for idx, toks in enumerate(corpus_tokens):
            for term, tf in Counter(toks).items():
                self._postings[term].append((idx, tf))
        n = self.n_chunks
        self._avgdl = (sum(self._lengths) / n) or 1.0
        self._idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()
        }

    def _chunk_scores_python(self, q_tokens: List[str], k: int) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self._avgdl
        for term in q_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for idx, tf in self._postings[term]:
                norm = k1 * (1 - b + b * self._lengths[idx] / avgdl)
                scores[idx] += idf * tf / (tf + norm)
        return heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])

    # -- bm25s backend --------------------------------------------------------

    def _build_bm25s(self, corpus_tokens: List[List[str]]) -> None:
        import bm25s

        vocab: Dict[str, int] = {}
        ids = [[vocab.setdefault(t, len(vocab)) for t in toks] for toks in corpus_tokens]
        self._vocab = vocab
        self._bm25s = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        self._bm25s.index(bm25s.tokenization.Tokenized(ids=ids, vocab=vocab), show_progress=False)

    def _chunk_scores_bm25s(self, q_tokens: List[str], k: int) -> List[Tuple[int, float]]:
        q_ids = [self._vocab[t] for t in q_tokens if t in self._vocab]
        if not q_ids:
            return []
        scores = self._bm25s.get_scores(q_ids)
        k = min(k, len(scores))
        top = scores.argpartition(-k)[-k:]
        return sorted(((int(i), float(scores[i])) for i in top if scores[i] > 0),
                      key=lambda kv: kv[1], reverse=True)

    # -- search ---------------------------------------------------------------

    def score_chunks(self, query: str, k: Optional[int] = None) -> List[Tuple[int, float]]:
        """Top-k (chunk_index, score) pairs with score > 0."""
        k = min(k or self.candidate_k, self.n_chunks)
        q_tokens = self.tokenizer(query)
        if self.backend == "bm25s":
            return self._chunk_scores_bm25s(q_tokens, k)
        return self._chunk_scores_python(q_tokens, k)

    def search(self, query: str, top_k: int = 100) -> List[DocHit]:
        """Document-level results: MaxP over the top ``candidate_k`` chunks."""
        best: Dict[str, float] = {}
        for idx, score in self.score_chunks(query):
            doc_id = self.chunk_doc_ids[idx]
            if score > best.get(doc_id, float("-inf")):
                best[doc_id] = score
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [DocHit(doc_id=d, score=s, rank=i + 1) for i, (d, s) in enumerate(ranked)]
