"""
CoIR (Code Information Retrieval) Data Loader
==============================================
Loads CoIR-Retrieval datasets from HuggingFace.

CoIR datasets follow the BEIR format but with a non-standard config layout:
  - corpus  : config="corpus",  split="corpus"
  - queries : config="queries", split="queries" (or "test" for some datasets)
  - qrels   : config=None (default config), split="test" (or "train")

Supported datasets:
    stackoverflow-qa       — Stack Overflow question → answer
    codefeedback-st        — Single-turn code feedback
    apps                   — Algorithmic problem → solution
    cosqa                  — Natural-language code search
    synthetic-text2sql     — Natural language → SQL

Reference: https://huggingface.co/CoIR-Retrieval
"""

import logging
from typing import Dict, List, Optional

from ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from ragtune.data.loaders.HuggingFaceLoader import (
    fetch_hf_split,
    populate_corpus,
    populate_qrels,
    build_raw_data,
)
from ragtune.data.datastructures.query import Query

logger = logging.getLogger(__name__)

COIR_DATASETS: List[str] = [
    "stackoverflow-qa",
    "codefeedback-st",
    "apps",
    "cosqa",
    "synthetic-text2sql",
]

_ORG = "CoIR-Retrieval"


class CoIRLoader(BaseDataLoader):
    """
    Loads a single CoIR-Retrieval dataset.

    Handles the non-standard BEIR config layout where qrels live in the
    default HuggingFace config (``config=None``) rather than a ``qrels`` config.

    Load order: qrels → queries → corpus. This allows gold-aware corpus capping
    without buffering all corpus rows in memory.

    Parameters
    ----------
    dataset : str
        One of the five CoIR dataset names (e.g. ``'stackoverflow-qa'``).
    split : str
        Evaluation split — determines which qrels split to try first
        (default ``'test'``; falls back to ``'train'`` automatically).
    max_queries : int | None
        Cap number of queries loaded (keeps only those with qrels).
    max_corpus_docs : int | None
        Cap non-gold corpus documents. Gold documents (from qrels) are always
        included regardless of this limit.
    cache_dir : str | None
        Optional HuggingFace cache directory.
    """

    def __init__(
        self,
        dataset: str,
        split: str = "test",
        max_queries: Optional[int] = None,
        max_corpus_docs: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        if dataset not in COIR_DATASETS:
            raise ValueError(
                f"Unknown CoIR dataset: {dataset!r}. Valid: {COIR_DATASETS}"
            )
        super().__init__(dataset=f"{_ORG}/{dataset}", split=split)
        self._coir_dataset = dataset
        self.max_queries = max_queries
        self.max_corpus_docs = max_corpus_docs
        self.cache_dir = cache_dir

    def _load_data(self) -> None:
        dataset_id = self.dataset
        logger.info(f"[CoIRLoader] Loading {dataset_id!r}")

        # ---- Qrels first (need gold IDs before streaming corpus) ----
        qrels_rows = None
        for qrels_split in [self.split, "train"]:
            try:
                qrels_rows = fetch_hf_split(
                    dataset_id, config=None, split=qrels_split, cache_dir=self.cache_dir
                )
                break
            except RuntimeError:
                continue
        if qrels_rows is None:
            raise RuntimeError(
                f"[CoIRLoader] Could not load qrels for {dataset_id!r} "
                f"(tried splits: {self.split!r}, 'train')"
            )
        populate_qrels(
            self._qrels, qrels_rows,
            qid_col="query-id", did_col="corpus-id", score_col="score",
        )
        self._qrels = {
            qid: {did: s for did, s in rels.items() if s > 0}
            for qid, rels in self._qrels.items()
        }
        self._qrels = {qid: rels for qid, rels in self._qrels.items() if rels}

        # ---- Queries (filter to those with qrels, cap to max_queries) ----
        try:
            queries_rows = fetch_hf_split(
                dataset_id, config="queries", split="queries", cache_dir=self.cache_dir
            )
        except RuntimeError:
            queries_rows = fetch_hf_split(
                dataset_id, config="queries", split="test", cache_dir=self.cache_dir
            )

        query_objs: Dict[str, Query] = {}
        for row in queries_rows:
            qid = str(row["_id"])
            if qid not in self._qrels:
                continue
            text = str(row.get("text", ""))
            self._queries[qid] = text
            query_objs[qid] = Query(text=text, idx=qid)
            if self.max_queries and len(self._queries) >= self.max_queries:
                break

        self._qrels = {qid: self._qrels[qid] for qid in self._queries if qid in self._qrels}

        # ---- Corpus (gold-aware cap via streaming) ----
        gold_ids = {did for rels in self._qrels.values() for did in rels}
        corpus_rows = fetch_hf_split(
            dataset_id, config="corpus", split="corpus", cache_dir=self.cache_dir
        )
        non_gold_count = 0
        for row in corpus_rows:
            doc_id = str(row.get("_id", row.get("id", "")))
            if not doc_id:
                continue
            is_gold = doc_id in gold_ids
            if not is_gold and self.max_corpus_docs and non_gold_count >= self.max_corpus_docs:
                continue
            self._corpus[doc_id] = {
                "text":  row.get("text", ""),
                "title": row.get("title", ""),
            }
            if not is_gold:
                non_gold_count += 1

        build_raw_data(self.raw_data, query_objs, self._qrels, self._corpus)
        logger.info(
            f"[CoIRLoader] {len(self._queries)} queries, "
            f"{len(self._corpus)} docs, "
            f"{sum(len(v) for v in self._qrels.values())} qrel pairs"
        )
