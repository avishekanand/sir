"""SkillRet Data Loader — skill retrieval for LLM agents."""

import json as _json
import logging
from src.ragtune.data.datastructures import Query, Context, Sample
from src.ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from src.ragtune.data.constants import HFDatasets

logger = logging.getLogger(__name__)


class SkillRetLoader(BaseDataLoader):
    """Load SkillRet skill retrieval benchmark."""

    def __init__(self, dataset: str = "test", split: str = "test", n_queries: int = 50):
        super().__init__(dataset=dataset, split=split)
        self.n_queries = n_queries

    def _load_data(self):
        from huggingface_hub import hf_hub_download

        s_path = hf_hub_download(
            HFDatasets.SKILLRET_REPO,
            f"data/skills/{self.dataset}.jsonl",
            repo_type="dataset",
        )
        q_path = hf_hub_download(
            HFDatasets.SKILLRET_REPO,
            f"data/queries/{self.dataset}.jsonl",
            repo_type="dataset",
        )
        r_path = hf_hub_download(
            HFDatasets.SKILLRET_REPO,
            f"data/qrels/{self.dataset}.jsonl",
            repo_type="dataset",
        )
        with open(s_path) as f:
            for line in f:
                s = _json.loads(line)
                parts = [s.get("name", ""), s.get("namespace", "")]
                desc = s.get("description", "")
                if desc:
                    parts.append(desc)
                content = s.get("content", "")
                if content:
                    parts.append(content)
                self._corpus[s["id"]] = {
                    "text": "\n".join(parts),
                    "title": s.get("name", ""),
                }
        qrels_raw = []
        with open(r_path) as f:
            for line in f:
                qrels_raw.append(_json.loads(line))
        relevant_qids = {q["query_id"] for q in qrels_raw}
        for r in qrels_raw:
            qid = r["query_id"]
            if qid not in self._qrels:
                self._qrels[qid] = {}
            self._qrels[qid][r["skill_id"]] = int(r["relevance"])
        with open(q_path) as f:
            for line in f:
                q = _json.loads(line)
                if q["id"] in relevant_qids:
                    self._queries[q["id"]] = q.get("query", "")
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
            "SkillRet %s: %d queries, %d skills",
            self.dataset,
            len(self._queries),
            len(self._corpus),
        )
