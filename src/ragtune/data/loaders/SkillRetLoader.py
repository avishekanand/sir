"""SkillRet Data Loader — skill retrieval for LLM agents.

SkillRet (Cho et al., 2026) provides structured skills from open-source
repositories with semantic tags and a two-level taxonomy.

HuggingFace layout (ThakiCloud/SKILLRET):
  Skills : data/skills/{split}.jsonl
      Columns: id, name, namespace, description, content, ...
  Queries: data/queries/{split}.jsonl
      Columns: id, query, ...
  Qrels  : data/qrels/{split}.jsonl
      Columns: query_id, skill_id, relevance

Reference: https://arxiv.org/abs/2605.05726
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


class SkillRetLoader(BaseDataLoader):
    """Load SkillRet skill retrieval benchmark.

    Parameters
    ----------
    dataset : str
        Split name: 'test' (4,997 queries) or 'train' (63,259 queries).
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
        dataset: str = "test",
        split: str = "test",
        n_queries: int = 0,
        max_corpus_docs: Optional[int] = None,
        cache_dir: Optional[str] = None,
        corpus_fields: Optional[List[str]] = None,
        corpus_sep: str = "\n",
        min_relevance: int = 1,
    ):
        super().__init__(dataset=dataset, split=split)
        self.n_queries = n_queries
        self.max_corpus_docs = max_corpus_docs
        self.cache_dir = cache_dir
        # Which skill fields to include in the corpus text.
        # Default: rich representation (name + namespace + description + content).
        # To reproduce Rahul's PR #20 exactly, set corpus_fields=["name", "description"]
        # and corpus_sep="\n\n" (Rahul joins with a blank line: "{name}\n\n{description}").
        self.corpus_fields = corpus_fields or [
            "name",
            "namespace",
            "description",
            "content",
        ]
        self.corpus_sep = corpus_sep
        # Minimum relevance score for a qrel to be considered "relevant".
        # Default 1 = keep only positive qrels (standard IR, matches Rahul's PR #20).
        # Set 0 to include relevance=0 entries (all judged docs).
        self.min_relevance = min_relevance

    def _load_data(self) -> None:
        from huggingface_hub import hf_hub_download

        logger.info(f"[SkillRetLoader] dataset={self.dataset!r}")

        # ---- Skills ----
        s_path = hf_hub_download(
            HFDatasets.SKILLRET_REPO,
            f"data/skills/{self.dataset}.jsonl",
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        skill_count = 0
        with open(s_path) as f:
            for line in f:
                if self.max_corpus_docs and skill_count >= self.max_corpus_docs:
                    break
                s = _json.loads(line)
                # Join only the configured fields, skipping empty values.
                # Rahul's exact format: "{name}\n\n{description}" — set
                # corpus_fields=["name","description"], corpus_sep="\n\n".
                parts = []
                for field in self.corpus_fields:
                    val = s.get(field, "")
                    if val:
                        parts.append(val)
                self._corpus[s["id"]] = {
                    "text": self.corpus_sep.join(parts),
                    "title": s.get("name", ""),
                }
                skill_count += 1

        # ---- Qrels ----
        r_path = hf_hub_download(
            HFDatasets.SKILLRET_REPO,
            f"data/qrels/{self.dataset}.jsonl",
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        qrels_raw = []
        with open(r_path) as f:
            for line in f:
                qrels_raw.append(_json.loads(line))

        # Only queries with at least one qrel >= min_relevance are relevant.
        # min_relevance is config-driven (default 1 = positive qrels only,
        # matching Rahul's PR #20). Set min_relevance=0 to include all judged docs.
        relevant_qids = {
            q["query_id"]
            for q in qrels_raw
            if int(q.get("relevance", 0)) >= self.min_relevance
        }
        for r in qrels_raw:
            rel = int(r.get("relevance", 0))
            if rel < self.min_relevance:
                continue
            qid = r["query_id"]
            if qid not in self._qrels:
                self._qrels[qid] = {}
            self._qrels[qid][r["skill_id"]] = rel

        # ---- Queries ----
        q_path = hf_hub_download(
            HFDatasets.SKILLRET_REPO,
            f"data/queries/{self.dataset}.jsonl",
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        with open(q_path) as f:
            for line in f:
                q = _json.loads(line)
                if q["id"] in relevant_qids:
                    self._queries[q["id"]] = q.get("query", "")
                if self.n_queries > 0 and len(self._queries) >= self.n_queries:
                    break

        # ---- Filter qrels to loaded queries only ----
        loaded_qids = set(self._queries.keys())
        self._qrels = {
            qid: rels for qid, rels in self._qrels.items() if qid in loaded_qids
        }

        # ---- Build raw_data ----
        from ragtune.data.loaders.HuggingFaceLoader import build_raw_data

        query_objs: Dict[str, Query] = {}
        for qid, text in self._queries.items():
            query_objs[qid] = Query(text=text, idx=qid)
        build_raw_data(self.raw_data, query_objs, self._qrels, self._corpus)

        logger.info(
            f"[SkillRetLoader] {self.dataset}: "
            f"{len(self._queries)} queries, "
            f"{len(self._corpus)} skills, "
            f"{sum(len(v) for v in self._qrels.values())} qrel pairs"
        )
