"""
OBLIQ-Bench Data Loader
=======================
Loads OBLIQ-Bench datasets from HuggingFace.

OBLIQ-Bench tests oblique-query retrieval across three query types:
  tip-of-tongue (congress), analogue (math, writing), descriptive (twitter, wildchat).

Data layout on HuggingFace (dianetc/OBLIQ-Bench):
  Queries : config=<task>, split="queries"  — HF dataset split
  Corpus  : config=<task>, split="corpus"   — streamed for large tasks
  Qrels   : per-task TSV file in the repo   — downloaded via hf_hub_download
  Excluded IDs : per_query_excluded_ids.json (math/writing only)

Reference: https://huggingface.co/datasets/dianetc/OBLIQ-Bench
"""

import csv
import json
import logging
from typing import Dict, List, Optional, Tuple

from ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from ragtune.data.datastructures.query import Query
from ragtune.data.datastructures.context import Context
from ragtune.data.datastructures.sample import Sample

logger = logging.getLogger(__name__)

DATASET_ID = "dianetc/OBLIQ-Bench"

OBLIQ_TASKS: List[str] = ["congress", "math", "writing", "twitter", "wildchat"]

_TASK_META: Dict[str, Dict] = {
    "congress": {
        "qrels_path": "tip-of-tongue/congress/queries+qrels/qrels.tsv",
    },
    "math": {
        "qrels_path": "analogues/math/queries+qrels/qrels.tsv",
        "excluded_ids_path": "analogues/math/queries+qrels/per_query_excluded_ids.json",
    },
    "writing": {
        "qrels_path": "analogues/writing/queries+qrels/qrels.tsv",
        "excluded_ids_path": "analogues/writing/queries+qrels/per_query_excluded_ids.json",
    },
    "twitter": {
        "qrels_path": "descriptive/twitter/queries+qrels/qrels.tsv",
    },
    "wildchat": {
        "qrels_path": "descriptive/wildchat/queries+qrels/qrels.tsv",
    },
}


class OBLIQLoader(BaseDataLoader):
    """
    Loads a single OBLIQ-Bench task.

    Corpus is streamed to handle large tasks (wildchat: 508k docs).
    Non-gold documents are capped at ``max_corpus_docs``; gold documents
    from the qrels are always preserved.

    Parameters
    ----------
    task : str
        One of ``'congress'``, ``'math'``, ``'writing'``, ``'twitter'``,
        ``'wildchat'``.
    split : str
        Evaluation split (default ``'test'``; unused for corpus/qrels but
        stored for consistency).
    max_queries : int | None
        Cap number of queries loaded.
    max_corpus_docs : int | None
        Cap non-gold corpus documents. Gold documents are always included.
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
        if task not in OBLIQ_TASKS:
            raise ValueError(
                f"Unknown OBLIQ task: {task!r}. Valid: {OBLIQ_TASKS}"
            )
        super().__init__(dataset=DATASET_ID, split=split)
        self.task = task
        self.max_queries = max_queries
        self.max_corpus_docs = max_corpus_docs
        self.cache_dir = cache_dir
        self._excluded_ids: Dict[str, List[str]] = {}

    def _load_data(self) -> None:
        try:
            from datasets import load_dataset
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "Required packages missing. Install with:\n"
                "  pip install datasets huggingface_hub\n"
                "Or: pip install -e '.[benchmarks]'"
            )

        meta = _TASK_META[self.task]
        logger.info(f"[OBLIQLoader] task={self.task!r}")

        # ---- Queries ----
        hf_kwargs: Dict = {}
        if self.cache_dir:
            hf_kwargs["cache_dir"] = self.cache_dir
        query_rows = list(
            load_dataset(DATASET_ID, self.task, split="queries", **hf_kwargs)
        )
        if self.max_queries:
            query_rows = query_rows[: self.max_queries]

        query_objs: Dict[str, Query] = {}
        for row in query_rows:
            qid = str(row["_id"])
            text = str(row["text"])
            self._queries[qid] = text
            query_objs[qid] = Query(text=text, idx=qid)

        # ---- Qrels (TSV file via hf_hub_download) ----
        qrels_file = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=meta["qrels_path"],
        )
        query_filter = set(self._queries.keys())
        with open(qrels_file) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 3 or row[0] == "query-id":
                    continue
                qid, did, score = row[0], row[1], int(row[2])
                if qid in query_filter and score > 0:
                    self._qrels.setdefault(qid, {})[did] = score

        # ---- Excluded IDs (math / writing only) ----
        if "excluded_ids_path" in meta:
            excl_file = hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=meta["excluded_ids_path"],
            )
            with open(excl_file) as f:
                self._excluded_ids = json.load(f)

        # ---- Corpus (streamed, gold-aware cap) ----
        gold_ids = {did for rels in self._qrels.values() for did in rels}
        corpus_ds = load_dataset(
            DATASET_ID, self.task, split="corpus", streaming=True, **hf_kwargs
        )
        non_gold_count = 0
        for row in corpus_ds:
            did = str(row["_id"])
            is_gold = did in gold_ids
            if not is_gold and self.max_corpus_docs and non_gold_count >= self.max_corpus_docs:
                if gold_ids.issubset(self._corpus):
                    break
                continue
            self._corpus[did] = {
                "text":  str(row.get("text", "")),
                "title": str(row.get("title", "")),
            }
            if not is_gold:
                non_gold_count += 1

        # Build Sample objects
        for qid, query_obj in query_objs.items():
            gold_docs = [did for did in self._qrels.get(qid, {}) if did in self._corpus]
            if gold_docs:
                for did in gold_docs:
                    doc = self._corpus[did]
                    self.raw_data.append(Sample(
                        idx=qid,
                        query=query_obj,
                        evidences=Context(text=doc["text"], idx=did, title=doc["title"]),
                    ))
            else:
                self.raw_data.append(Sample(idx=qid, query=query_obj, evidences=None))

        logger.info(
            f"[OBLIQLoader] {len(self._queries)} queries, "
            f"{len(self._corpus)} docs, "
            f"{sum(len(v) for v in self._qrels.values())} qrel pairs"
        )

    def get_excluded_ids(self) -> Dict[str, List[str]]:
        """
        Return per-query excluded document IDs (math/writing tasks only).

        These are source documents that must be masked from evaluation because
        they contain the answer verbatim, making retrieval trivial.
        """
        self._ensure_loaded()
        return self._excluded_ids
