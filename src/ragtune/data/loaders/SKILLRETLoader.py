"""
SKILLRET Data Loader
====================
Loads the SKILLRET skill-retrieval benchmark from HuggingFace.

SKILLRET maps natural-language user requests to agent skills (tools, APIs,
functions) sourced from GitHub. The corpus is 6 660 skills; queries and qrels
use a separate split to avoid schema-mismatch issues across train/test.

HuggingFace layout (ThakiCloud/SKILLRET):
  Skills  : config="skills",  split="test"
      Columns: id, name, description, skill_md  (full Markdown body)
  Queries : config="queries", split="test"  (loaded via streaming)
      Columns: id, query
      Note: train split has an extra ``original_query`` column; non-streaming
      load unifies schemas and may raise a cast error. Streaming avoids this.
  Qrels   : config="qrels",   split="test"
      Columns: query_id, skill_id, relevance

Document representation: ``"{name}\\n\\n{description}"``.
``skill_md`` is intentionally omitted — it exceeds the 256-token limit of
``all-MiniLM-L6-v2`` and degrades embedding quality.

Reference: https://huggingface.co/datasets/ThakiCloud/SKILLRET
"""

import logging
from typing import Dict, List, Optional

from ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from ragtune.data.loaders.HuggingFaceLoader import fetch_hf_split, build_raw_data
from ragtune.data.datastructures.query import Query
from ragtune.data.datastructures.context import Context
from ragtune.data.datastructures.sample import Sample

logger = logging.getLogger(__name__)

DATASET_ID = "ThakiCloud/SKILLRET"


class SKILLRETLoader(BaseDataLoader):
    """
    Loads the SKILLRET benchmark (single corpus, no task axis).

    Parameters
    ----------
    split : str
        Evaluation split (default ``'test'``).
    max_queries : int | None
        Cap number of queries loaded.
    cache_dir : str | None
        Optional HuggingFace cache directory.
    """

    def __init__(
        self,
        split: str = "test",
        max_queries: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(dataset=DATASET_ID, split=split)
        self.max_queries = max_queries
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

        hf_kwargs: Dict = {}
        if self.cache_dir:
            hf_kwargs["cache_dir"] = self.cache_dir

        logger.info("[SKILLRETLoader] Loading skills corpus...")

        # ---- Skills corpus ----
        # Full corpus (6 660 skills) fits comfortably in memory.
        skills_ds = fetch_hf_split(DATASET_ID, "skills", self.split, self.cache_dir)
        for row in skills_ds:
            sid = str(row["id"])
            self._corpus[sid] = {
                "text":  f"{row['name']}\n\n{row['description']}",
                "title": row["name"],
            }

        # ---- Queries (streamed to avoid train/test schema mismatch) ----
        query_objs: Dict[str, Query] = {}
        for row in load_dataset(
            DATASET_ID, "queries", split=self.split, streaming=True, **hf_kwargs
        ):
            qid = str(row["id"])
            text = str(row["query"])
            self._queries[qid] = text
            query_objs[qid] = Query(text=text, idx=qid)
            if self.max_queries and len(self._queries) >= self.max_queries:
                break

        # ---- Qrels ----
        query_ids = set(self._queries.keys())
        qrels_ds = fetch_hf_split(DATASET_ID, "qrels", self.split, self.cache_dir)
        for row in qrels_ds:
            qid = str(row["query_id"])
            if qid in query_ids and int(row["relevance"]) > 0:
                self._qrels.setdefault(qid, {})[str(row["skill_id"])] = int(row["relevance"])

        build_raw_data(self.raw_data, query_objs, self._qrels, self._corpus)
        logger.info(
            f"[SKILLRETLoader] {len(self._corpus)} skills, "
            f"{len(self._queries)} queries, "
            f"{sum(len(v) for v in self._qrels.values())} qrel pairs"
        )
