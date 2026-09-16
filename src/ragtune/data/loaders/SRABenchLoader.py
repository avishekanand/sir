"""SRA-Bench Data Loader — skill retrieval augmentation benchmark.

SRA-Bench (Su et al., 2026) tests whether agents can retrieve, incorporate,
and apply external skills to solve capability-intensive tasks.

HuggingFace layout (WeihangSu/SRA-Bench):
  Corpus  : corpus/corpus.json
      List of {skill_id, name, description, content, ...}
  Instances: instances/{sub_dataset}.json
      List of {instance_id, dataset, question, skill_annotations, eval_data}

Sub-datasets: toolqa, theoremqa, bigcodebench, champ, logicbench, medcalcbench

Reference: https://arxiv.org/abs/2604.24594
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

SRA_SUBSETS = [
    "toolqa",
    "theoremqa",
    "bigcodebench",
    "champ",
    "logicbench",
    "medcalcbench",
]


class SRABenchLoader(BaseDataLoader):
    """Load SRA-Bench skill retrieval augmentation benchmark.

    Parameters
    ----------
    dataset : str
        Sub-dataset: 'toolqa', 'theoremqa', 'bigcodebench',
        'champ', 'logicbench', 'medcalcbench'.
    split : str
        Ignored. Kept for interface compatibility.
    n_queries : int
        Max queries to load (0 = all).
    max_corpus_docs : int | None
        Cap skills loaded. None = all.
    cache_dir : str | None
        Optional HuggingFace cache directory.
    """

    def __init__(
        self,
        dataset: str = "toolqa",
        split: str = "test",
        n_queries: int = 0,
        max_corpus_docs: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        if dataset not in SRA_SUBSETS:
            raise ValueError(
                f"Unknown SRA-Bench sub-dataset: {dataset!r}. Valid: {SRA_SUBSETS}"
            )
        super().__init__(dataset=dataset, split=split)
        self.n_queries = n_queries
        self.max_corpus_docs = max_corpus_docs
        self.cache_dir = cache_dir

    def _load_data(self) -> None:
        from huggingface_hub import hf_hub_download

        logger.info(f"[SRABenchLoader] dataset={self.dataset!r}")

        # ---- Corpus ----
        c_path = hf_hub_download(
            HFDatasets.SRA_BENCH_REPO,
            "corpus/corpus.json",
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        with open(c_path) as f:
            corpus_raw = _json.load(f)

        corpus_ids: set = set()
        skill_count = 0
        for entry in corpus_raw:
            if self.max_corpus_docs and skill_count >= self.max_corpus_docs:
                break
            parts = [entry.get("name", ""), entry.get("description", "")]
            content = entry.get("content", "")
            if content:
                parts.append(content)
            self._corpus[entry["skill_id"]] = {
                "text": "\n".join(parts),
                "title": entry.get("name", ""),
            }
            corpus_ids.add(entry["skill_id"])
            skill_count += 1

        # ---- Instances (queries + qrels) ----
        i_path = hf_hub_download(
            HFDatasets.SRA_BENCH_REPO,
            f"instances/{self.dataset}.json",
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        with open(i_path) as f:
            instances = _json.load(f)

        for inst in instances:
            qid = inst["instance_id"]
            for skill_id in inst.get("skill_annotations", []):
                if skill_id in corpus_ids:
                    if qid not in self._qrels:
                        self._qrels[qid] = {}
                    self._qrels[qid][skill_id] = 1
            if qid in self._qrels:
                self._queries[qid] = str(inst.get("question", ""))
            if self.n_queries > 0 and len(self._queries) >= self.n_queries:
                break

        # ---- Build raw_data ----
        from ragtune.data.loaders.HuggingFaceLoader import build_raw_data

        query_objs: Dict[str, Query] = {}
        for qid, text in self._queries.items():
            query_objs[qid] = Query(text=text, idx=qid)
        build_raw_data(self.raw_data, query_objs, self._qrels, self._corpus)

        logger.info(
            f"[SRABenchLoader] {self.dataset}: "
            f"{len(self._queries)} queries, "
            f"{len(self._corpus)} skills, "
            f"{sum(len(v) for v in self._qrels.values())} qrel pairs"
        )
