"""SRA-Bench Data Loader — skill retrieval augmentation benchmark."""

import json as _json
import logging
from src.ragtune.data.datastructures import Query, Context, Sample
from src.ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from src.ragtune.data.constants import HFDatasets

logger = logging.getLogger(__name__)


class SRABenchLoader(BaseDataLoader):
    """Load SRA-Bench skill retrieval augmentation benchmark."""

    def __init__(
        self, dataset: str = "toolqa", split: str = "test", n_queries: int = 50
    ):
        super().__init__(dataset=dataset, split=split)
        self.n_queries = n_queries

    def _load_data(self):
        from huggingface_hub import hf_hub_download

        c_path = hf_hub_download(
            HFDatasets.SRA_BENCH_REPO, "corpus/corpus.json", repo_type="dataset"
        )
        i_path = hf_hub_download(
            HFDatasets.SRA_BENCH_REPO,
            f"instances/{self.dataset}.json",
            repo_type="dataset",
        )
        with open(c_path) as f:
            corpus_raw = _json.load(f)
        with open(i_path) as f:
            instances = _json.load(f)
        corpus_ids = set()
        for entry in corpus_raw:
            parts = [entry.get("name", ""), entry.get("description", "")]
            content = entry.get("content", "")
            if content:
                parts.append(content)
            self._corpus[entry["skill_id"]] = {
                "text": "\n".join(parts),
                "title": entry.get("name", ""),
            }
            corpus_ids.add(entry["skill_id"])
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
            "SRA-Bench %s: %d queries, %d skills",
            self.dataset,
            len(self._queries),
            len(self._corpus),
        )
