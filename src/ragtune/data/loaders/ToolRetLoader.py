"""ToolRet Data Loader — cross-corpus matching across web/code/customized."""

import json as _json
import logging
from src.ragtune.data.datastructures import Query, Context, Sample
from src.ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from src.ragtune.data.constants import HFDatasets

logger = logging.getLogger(__name__)


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
    """Load a single ToolRet sub-dataset with cross-corpus matching."""

    def __init__(self, dataset: str, split: str = "test", n_queries: int = 50):
        super().__init__(dataset=dataset, split=split)
        self.n_queries = n_queries

    def _load_data(self):
        from huggingface_hub import hf_hub_download
        import pandas as pd

        q_path = hf_hub_download(
            HFDatasets.TOOLRET_QUERIES,
            f"{self.dataset}/queries-00000-of-00001.parquet",
            repo_type="dataset",
        )
        qdf = pd.read_parquet(q_path)
        tool_map = {}
        for corpus_name in ("web", "code", "customized"):
            try:
                t_path = hf_hub_download(
                    HFDatasets.TOOLRET_TOOLS,
                    f"{corpus_name}/tools-00000-of-00001.parquet",
                    repo_type="dataset",
                )
                tdf = pd.read_parquet(t_path)
                for _, row in tdf.iterrows():
                    doc = row.get("documentation")
                    if isinstance(doc, str):
                        try:
                            doc = _json.loads(doc)
                        except:
                            pass
                    tool_map[row["id"]] = _flatten_tool_doc(doc)
            except:
                continue
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
        for doc_id in seen_docnos:
            self._corpus[doc_id] = {"text": tool_map[doc_id], "title": ""}
        query_objs = {
            qid: Query(text=text, idx=qid) for qid, text in self._queries.items()
        }
        for qid, rels in self._qrels.items():
            if qid not in query_objs:
                continue
            for doc_id in rels:
                if doc_id in self._corpus:
                    ctx = Context(text=self._corpus[doc_id]["text"], idx=doc_id)
                    self.raw_data.append(
                        Sample(
                            idx=qid, query=query_objs[qid], evidences=ctx, answer=None
                        )
                    )
        logger.info(
            "ToolRet %s: %d queries, %d docs",
            self.dataset,
            len(self._queries),
            len(self._corpus),
        )
