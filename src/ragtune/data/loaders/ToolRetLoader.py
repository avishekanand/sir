"""ToolRet Data Loader — cross-corpus matching across web/code/customized.

ToolRet (Shi et al., ACL 2025) stores queries in per-subset parquet files
and tools across three corpora (web, code, customized). Queries reference
tools from mixed corpora, so we load all three and match.

HuggingFace layout (mangopy/ToolRet-Queries, mangopy/ToolRet-Tools):
  Queries : {subset}/queries-00000-of-00001.parquet
      Columns: id, query, instruction, labels (JSON), category
  Tools   : {corpus}/tools-00000-of-00001.parquet  (corpus = web|code|customized)
      Columns: id, documentation (JSON or str)

Reference: https://arxiv.org/abs/2503.01763
"""

import json as _json
import logging
from typing import Dict, List, Optional

from ragtune.data.datastructures.query import Query
from ragtune.data.datastructures.context import Context
from ragtune.data.datastructures.sample import Sample
from ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from ragtune.data.constants import HFDatasets

logger = logging.getLogger(__name__)

TOOLRET_CORPORA = ("web", "code", "customized")


def _parse_json_labels(labels) -> list:
    if isinstance(labels, str):
        try:
            return _json.loads(labels)
        except _json.JSONDecodeError:
            return []
    return list(labels) if isinstance(labels, (list, tuple)) else []


def _flatten_tool_doc(doc) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        parts = [
            f"{k}: {v}"
            for k in ("name", "description", "expressions", "parameters", "path")
            if (v := doc.get(k))
        ]
        return " | ".join(parts)
    return str(doc)


class ToolRetLoader(BaseDataLoader):
    """Load a single ToolRet sub-dataset with cross-corpus matching.

    Parameters
    ----------
    dataset : str
        Sub-dataset name (e.g., 'apibank', 'gorilla-tensor', 'metatool').
    split : str
        Ignored (ToolRet has no splits). Kept for interface compatibility.
    n_queries : int
        Max queries to load (0 = all).
    max_corpus_docs : int | None
        Cap corpus tools. None = all tools.
    cache_dir : str | None
        Optional HuggingFace cache directory.
    """

    def __init__(
        self,
        dataset: str,
        split: str = "test",
        n_queries: int = 0,
        max_corpus_docs: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(dataset=dataset, split=split)
        self.n_queries = n_queries
        self.max_corpus_docs = max_corpus_docs
        self.cache_dir = cache_dir

    def _load_data(self) -> None:
        from huggingface_hub import hf_hub_download
        import pandas as pd

        logger.info(f"[ToolRetLoader] dataset={self.dataset!r}")

        # ---- Queries ----
        q_path = hf_hub_download(
            HFDatasets.TOOLRET_QUERIES,
            f"{self.dataset}/queries-00000-of-00001.parquet",
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        qdf = pd.read_parquet(q_path)

        # ---- Tools (cross-corpus) ----
        tool_map: Dict[str, str] = {}
        for corpus_name in TOOLRET_CORPORA:
            try:
                t_path = hf_hub_download(
                    HFDatasets.TOOLRET_TOOLS,
                    f"{corpus_name}/tools-00000-of-00001.parquet",
                    repo_type="dataset",
                    cache_dir=self.cache_dir,
                )
                tdf = pd.read_parquet(t_path)
                for _, row in tdf.iterrows():
                    doc = row.get("documentation")
                    if isinstance(doc, str):
                        try:
                            doc = _json.loads(doc)
                        except (_json.JSONDecodeError, ValueError):
                            pass
                    tool_map[row["id"]] = _flatten_tool_doc(doc)
            except Exception as e:
                logger.warning(
                    f"[ToolRetLoader] Failed to load corpus {corpus_name}: {e}"
                )
                continue

        # ---- Match queries to tools ----
        seen_docnos = set()
        for _, row in qdf.iterrows():
            qid = row["id"]
            labels = row.get("labels")
            if labels is None or (isinstance(labels, float) and pd.isna(labels)):
                continue
            for lbl in _parse_json_labels(labels):
                if isinstance(lbl, dict):
                    doc_id = lbl.get("id")
                    if doc_id and doc_id in tool_map:
                        if qid not in self._qrels:
                            self._qrels[qid] = {}
                        self._qrels[qid][doc_id] = int(lbl.get("relevance", 1))
                        seen_docnos.add(doc_id)
            if qid in self._qrels:
                self._queries[qid] = str(row.get("query", ""))
            if self.n_queries > 0 and len(self._queries) >= self.n_queries:
                break

        # ---- Build corpus (only referenced tools) ----
        tool_count = 0
        for doc_id in seen_docnos:
            if self.max_corpus_docs and tool_count >= self.max_corpus_docs:
                break
            self._corpus[doc_id] = {"text": tool_map[doc_id], "title": ""}
            tool_count += 1

        # ---- Build raw_data ----
        from ragtune.data.loaders.HuggingFaceLoader import build_raw_data

        query_objs: Dict[str, Query] = {}
        for qid, text in self._queries.items():
            query_objs[qid] = Query(text=text, idx=qid)
        build_raw_data(self.raw_data, query_objs, self._qrels, self._corpus)

        logger.info(
            f"[ToolRetLoader] {self.dataset}: "
            f"{len(self._queries)} queries, "
            f"{len(self._corpus)} tools, "
            f"{sum(len(v) for v in self._qrels.values())} qrel pairs"
        )
