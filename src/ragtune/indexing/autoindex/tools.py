"""
Read-only tools for the AutoIndex Analysis Agent.

The paper's three tools are exposed over in-memory data, presented as two virtual
JSONL files:

  documents.jsonl           {"doc_id", "text", "title"?}
  validation_queries.jsonl  {"query_id", "query_text", "relevant_doc_ids"}

Held-out queries are never given to the toolbox, so the agent can't reach them.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, Iterator, List, Optional

from ragtune.indexing.autoindex.bm25 import ChunkBM25Index

DOCUMENTS_FILE = "documents.jsonl"
VALIDATION_FILE = "validation_queries.jsonl"
_FORBIDDEN_FILES = {"queries.jsonl", "evaluation_queries.jsonl", "test_queries.jsonl"}
_MAX_TOP_K = 50
_MAX_CHARS = 4000
_MAX_GREP_RESULTS = 25
_GREP_LINE_CHARS = 300


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bm25_retrieve",
            "description": "Query the current BM25 index. Returns doc_id, score and rank for each result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query text to search"},
                    "top_k": {"type": "integer", "description": "Number of results (default 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                f"Read '{DOCUMENTS_FILE}' or '{VALIDATION_FILE}'. Use filter_id to return only the "
                "entry whose doc_id or query_id matches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": f"'{DOCUMENTS_FILE}' or '{VALIDATION_FILE}'"},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return (default 800)"},
                    "filter_id": {"type": "string", "description": "doc_id or query_id to look up"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Regex search over a file's JSONL lines. Returns matching lines (truncated) with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regex"},
                    "file_path": {"type": "string", "description": f"'{DOCUMENTS_FILE}' or '{VALIDATION_FILE}'"},
                    "max_results": {"type": "integer", "description": "Maximum matching lines (default 10)"},
                },
                "required": ["pattern", "file_path"],
            },
        },
    },
]


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "... [truncated]"


class AnalysisToolbox:
    """Dispatches Analysis Agent tool calls against the current index and validation data."""

    def __init__(
        self,
        index: ChunkBM25Index,
        corpus: Dict[str, Dict],
        val_queries: Dict[str, str],
        val_qrels: Dict[str, Dict[str, int]],
    ):
        self.index = index
        self.corpus = corpus
        self.val_queries = val_queries
        self.val_qrels = val_qrels
        self.calls: List[str] = []

    # -- virtual files ------------------------------------------------------

    def _doc_record(self, doc_id: str) -> dict:
        entry = self.corpus[doc_id]
        rec = {"doc_id": doc_id, "text": entry.get("text", "")}
        if entry.get("title"):
            rec["title"] = entry["title"]
        return rec

    def _query_record(self, qid: str) -> dict:
        rel = [d for d, r in self.val_qrels.get(qid, {}).items() if r > 0]
        return {"query_id": qid, "query_text": self.val_queries[qid], "relevant_doc_ids": rel}

    def _lines(self, file_path: str) -> Iterator[str]:
        if file_path == DOCUMENTS_FILE:
            for doc_id in self.corpus:
                yield json.dumps(self._doc_record(doc_id), ensure_ascii=False)
        else:
            for qid in self.val_queries:
                yield json.dumps(self._query_record(qid), ensure_ascii=False)

    def _check_file(self, file_path: str) -> Optional[str]:
        if file_path in _FORBIDDEN_FILES:
            return f"Error: access to held-out queries is forbidden. Use '{VALIDATION_FILE}'."
        if file_path not in (DOCUMENTS_FILE, VALIDATION_FILE):
            return f"Error: unknown file '{file_path}'. Available: '{DOCUMENTS_FILE}', '{VALIDATION_FILE}'."
        return None

    # -- tools ----------------------------------------------------------------

    def bm25_retrieve(self, query: str, top_k: int = 10) -> str:
        top_k = max(1, min(int(top_k), _MAX_TOP_K))
        hits = self.index.search(query, top_k=top_k)
        return json.dumps([{"doc_id": h.doc_id, "score": round(h.score, 4), "rank": h.rank} for h in hits])

    def read_file(self, file_path: str, max_chars: int = 800, filter_id: Optional[str] = None) -> str:
        err = self._check_file(file_path)
        if err:
            return err
        max_chars = max(1, min(int(max_chars), _MAX_CHARS))
        if filter_id is not None:
            if file_path == DOCUMENTS_FILE and filter_id in self.corpus:
                return _truncate(json.dumps(self._doc_record(filter_id), ensure_ascii=False), max_chars)
            if file_path == VALIDATION_FILE and filter_id in self.val_queries:
                return _truncate(json.dumps(self._query_record(filter_id), ensure_ascii=False), max_chars)
            return f"Error: no entry with id '{filter_id}' in {file_path}"
        out, used = [], 0
        for line in self._lines(file_path):
            out.append(line)
            used += len(line) + 1
            if used >= max_chars:
                break
        return _truncate("\n".join(out), max_chars)

    def grep_search(self, pattern: str, file_path: str, max_results: int = 10) -> str:
        err = self._check_file(file_path)
        if err:
            return err
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"
        max_results = max(1, min(int(max_results), _MAX_GREP_RESULTS))
        matches = []
        for line_no, line in enumerate(self._lines(file_path), start=1):
            m = regex.search(line)
            if m:
                start = max(0, m.start() - _GREP_LINE_CHARS // 2)
                matches.append(f"line {line_no}: {_truncate(line[start:start + _GREP_LINE_CHARS], _GREP_LINE_CHARS)}")
                if len(matches) >= max_results:
                    break
        return "\n".join(matches) if matches else "No matches found."

    def dispatch(self, name: str, args: dict) -> str:
        fns: Dict[str, Callable[..., str]] = {
            "bm25_retrieve": self.bm25_retrieve,
            "read_file": self.read_file,
            "grep_search": self.grep_search,
        }
        self.calls.append(name)
        if name not in fns:
            return f"Error: unknown tool '{name}'"
        try:
            return fns[name](**args)
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:  # tool errors go back to the model, not up the stack
            return f"Error: {type(e).__name__}: {e}"
