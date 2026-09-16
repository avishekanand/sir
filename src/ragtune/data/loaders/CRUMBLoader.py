"""
CRUMB Passage-Retrieval Data Loader
====================================
Loads CRUMB (Complex Retrieval Unified Multi-task Benchmark) datasets.

CRUMB stores passage qrels inline within the query rows (``passage_qrels``
field), so there is no separate qrels file. Corpus is streamed to handle
large tasks (code_retrieval, legal_qa).

HuggingFace layout (jfkback/crumb):
  Queries : config="evaluation_queries", split=<task>
      Columns: query_id, query_content, passage_qrels [{id, label}]
  Corpus  : config="passage_corpus", split=<task>  (streamed)
      Columns: document_id, document_content

Passage-retrieval mode only; document_qrels are intentionally ignored.

Reference: https://huggingface.co/datasets/jfkback/crumb
"""

import logging
from typing import Dict, List, Optional

from ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from ragtune.data.loaders.HuggingFaceLoader import fetch_hf_split, build_raw_data
from ragtune.data.datastructures.query import Query
from ragtune.data.datastructures.context import Context
from ragtune.data.datastructures.sample import Sample

logger = logging.getLogger(__name__)

DATASET_ID = "jfkback/crumb"

CRUMB_TASKS: List[str] = [
    "paper_retrieval",
    "theorem_retrieval",
    "tip_of_the_tongue",
    "stack_exchange",
    "clinical_trial",
    "set_operation_entity_retrieval",
    "code_retrieval",
    "legal_qa",
]


class CRUMBLoader(BaseDataLoader):
    """
    Loads a single CRUMB passage-retrieval task.

    Passage qrels are read inline from query rows (``passage_qrels`` field).
    Only queries with at least one positive passage qrel are loaded.
    Corpus is streamed; non-gold passages are capped at ``max_corpus_docs``.

    Parameters
    ----------
    task : str
        One of the eight CRUMB task names.
    split : str
        Evaluation split (default ``'test'``; used as the HF split name).
    max_queries : int | None
        Cap number of queries loaded.
    max_corpus_docs : int | None
        Cap non-gold corpus passages. Gold passages are always included.
    cache_dir : str | None
        Optional HuggingFace cache directory.
    """

    def __init__(
        self,
        task: str,
        split: str = "test",
        max_queries: Optional[int] = None,
        max_corpus_docs: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        if task not in CRUMB_TASKS:
            raise ValueError(
                f"Unknown CRUMB task: {task!r}. Valid: {CRUMB_TASKS}"
            )
        super().__init__(dataset=DATASET_ID, split=split)
        self.task = task
        self.max_queries = max_queries
        self.max_corpus_docs = max_corpus_docs
        self.cache_dir = cache_dir

    def _load_data(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Required package missing. Install with:\n"
                "  pip install datasets\n"
                "Or: pip install -e '.[benchmarks]'"
            )

        logger.info(f"[CRUMBLoader] task={self.task!r}")

        # ---- Queries + inline passage qrels ----
        hf_kwargs: Dict = {}
        if self.cache_dir:
            hf_kwargs["cache_dir"] = self.cache_dir

        query_rows = fetch_hf_split(
            DATASET_ID, "evaluation_queries", self.task, self.cache_dir
        )
        query_objs: Dict[str, Query] = {}

        for row in query_rows:
            qid = str(row["query_id"])
            text = str(row["query_content"])
            pqrels = row.get("passage_qrels") or []
            rels = {
                str(entry["id"]): int(entry["label"])
                for entry in pqrels
                if int(entry["label"]) > 0
            }
            if not rels:
                continue
            self._queries[qid] = text
            query_objs[qid] = Query(text=text, idx=qid)
            self._qrels[qid] = rels
            if self.max_queries and len(self._queries) >= self.max_queries:
                break

        # ---- Corpus (streamed, gold-aware cap) ----
        gold_ids = {pid for rels in self._qrels.values() for pid in rels}
        corpus_ds = load_dataset(
            DATASET_ID, "passage_corpus", split=self.task, streaming=True, **hf_kwargs
        )
        non_gold_count = 0
        for row in corpus_ds:
            pid = str(row["document_id"])
            is_gold = pid in gold_ids
            if not is_gold and self.max_corpus_docs and non_gold_count >= self.max_corpus_docs:
                if gold_ids.issubset(self._corpus):
                    break
                continue
            self._corpus[pid] = {
                "text":  str(row["document_content"]),
                "title": "",
            }
            if not is_gold:
                non_gold_count += 1

        # Build Sample objects
        for qid, query_obj in query_objs.items():
            gold_docs = [pid for pid in self._qrels.get(qid, {}) if pid in self._corpus]
            if gold_docs:
                for pid in gold_docs:
                    doc = self._corpus[pid]
                    self.raw_data.append(Sample(
                        idx=qid,
                        query=query_obj,
                        evidences=Context(text=doc["text"], idx=pid, title=""),
                    ))
            else:
                self.raw_data.append(Sample(idx=qid, query=query_obj, evidences=None))

        logger.info(
            f"[CRUMBLoader] {len(self._queries)} queries, "
            f"{len(self._corpus)} passages, "
            f"{sum(len(v) for v in self._qrels.values())} qrel pairs"
        )
