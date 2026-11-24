"""
Validation metrics for AutoIndex, matching the paper's harness.

Recall@k is fractional (|relevant ∩ top-k| / |relevant|), nDCG@10 uses binary
gains, and both are macro-averaged over queries. Per-query hit/rank data feeds
the Analysis Agent and the search history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ragtune.indexing.autoindex.bm25 import ChunkBM25Index


@dataclass
class QueryOutcome:
    query_id: str
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float
    ranks: List[int]                 # 1-based ranks of relevant docs found in the top 100
    retrieved_doc_ids: List[str]     # top 10, for analysis

    @property
    def hit(self) -> bool:
        return self.recall_at_100 > 0

    @property
    def rank(self) -> Optional[int]:
        return min(self.ranks) if self.ranks else None


@dataclass
class EvalSummary:
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float
    per_query: List[QueryOutcome] = field(default_factory=list)

    def by_qid(self) -> Dict[str, QueryOutcome]:
        return {o.query_id: o for o in self.per_query}

    def as_dict(self) -> Dict[str, float]:
        return {"recall_at_10": self.recall_at_10, "recall_at_100": self.recall_at_100,
                "ndcg_at_10": self.ndcg_at_10}


def score_ranking(query_id: str, ranked_doc_ids: List[str], relevant: Dict[str, int]) -> QueryOutcome:
    rel = {d for d, r in relevant.items() if r > 0}
    n_rel = len(rel) or 1
    ranks = [i + 1 for i, d in enumerate(ranked_doc_ids[:100]) if d in rel]
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= 10)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), 10) + 1))
    return QueryOutcome(
        query_id=query_id,
        recall_at_10=sum(r <= 10 for r in ranks) / n_rel,
        recall_at_100=len(ranks) / n_rel,
        ndcg_at_10=dcg / idcg if idcg else 0.0,
        ranks=ranks,
        retrieved_doc_ids=ranked_doc_ids[:10],
    )


def evaluate_index(
    index: ChunkBM25Index,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    top_k: int = 100,
) -> EvalSummary:
    """Run every query in ``queries`` that has qrels; return macro-averaged metrics."""
    outcomes = []
    for qid, text in queries.items():
        if qid not in qrels:
            continue
        ranked = [h.doc_id for h in index.search(text, top_k=top_k)]
        outcomes.append(score_ranking(qid, ranked, qrels[qid]))
    n = len(outcomes) or 1
    return EvalSummary(
        recall_at_10=sum(o.recall_at_10 for o in outcomes) / n,
        recall_at_100=sum(o.recall_at_100 for o in outcomes) / n,
        ndcg_at_10=sum(o.ndcg_at_10 for o in outcomes) / n,
        per_query=outcomes,
    )


def diff_outcomes(new: EvalSummary, old: EvalSummary) -> Dict[str, List[str]]:
    """Query IDs that became hits (``improved``) or stopped being hits (``regressed``) at @100."""
    old_by = old.by_qid()
    improved, regressed = [], []
    for o in new.per_query:
        prev = old_by.get(o.query_id)
        if prev is None:
            continue
        if o.hit and not prev.hit:
            improved.append(o.query_id)
        elif prev.hit and not o.hit:
            regressed.append(o.query_id)
    return {"improved": improved, "regressed": regressed}
