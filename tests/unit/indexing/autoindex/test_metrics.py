import math

import pytest

from ragtune.indexing.autoindex.bm25 import ChunkBM25Index
from ragtune.indexing.autoindex.metrics import (
    EvalSummary,
    QueryOutcome,
    diff_outcomes,
    evaluate_index,
    score_ranking,
)
from ragtune.indexing.autoindex.program import Chunk


def test_score_ranking_fractional_recall_and_ndcg():
    ranked = ["x", "g1", "y"] + [f"n{i}" for i in range(20)] + ["g2"]
    o = score_ranking("q", ranked, {"g1": 1, "g2": 1, "g3": 1, "neg": 0})
    assert o.ranks == [2, 24]
    assert o.recall_at_10 == pytest.approx(1 / 3)
    assert o.recall_at_100 == pytest.approx(2 / 3)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert o.ndcg_at_10 == pytest.approx((1 / math.log2(3)) / idcg)
    assert o.hit and o.rank == 2
    assert o.retrieved_doc_ids == ranked[:10]


def test_score_ranking_miss():
    o = score_ranking("q", ["a", "b"], {"g": 1})
    assert not o.hit and o.rank is None and o.ndcg_at_10 == 0


def test_evaluate_index_skips_queries_without_qrels():
    idx = ChunkBM25Index([Chunk("a_0", "a", "apple"), Chunk("b_0", "b", "banana")], backend="python")
    s = evaluate_index(idx, {"q1": "apple", "q2": "banana", "q3": "x"}, {"q1": {"a": 1}, "q2": {"a": 1}})
    assert [o.query_id for o in s.per_query] == ["q1", "q2"]
    assert s.recall_at_100 == pytest.approx(0.5)


def _summary(hits):
    return EvalSummary(0, 0, 0, [QueryOutcome(q, 0, 1.0 if h else 0.0, 0, [1] if h else [], []) for q, h in hits])


def test_diff_outcomes():
    new = _summary([("q1", True), ("q2", False), ("q3", True)])
    old = _summary([("q1", False), ("q2", True), ("q3", True)])
    assert diff_outcomes(new, old) == {"improved": ["q1"], "regressed": ["q2"]}
