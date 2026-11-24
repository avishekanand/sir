"""Unit tests for the outcome-reflective arm. Fake LLM, no network."""

import json
import random
from unittest.mock import MagicMock, patch

import pytest

from ragtune.tuning.reflect_optimizer import ReflectConfig, ReflectOptimizer

RESTRICTED = {
    "reranker_types": ["cross-encoder"], "reformulator_types": ["identity"],
    "estimator_types": ["baseline", "utility"], "feedback_types": ["none"],
    "ce_models": ["cross-encoder/ms-marco-MiniLM-L-2-v2",
                  "cross-encoder/ms-marco-MiniLM-L-6-v2"],
    "budget_reformulations_range": (0, 0),
}


def make(**kw) -> ReflectOptimizer:
    cfg = ReflectConfig(name="t", n_iterations=2, n_eval_queries=4,
                        search_space_overrides=dict(RESTRICTED),
                        baseline_ndcg=0.33, verbose=False, **kw)
    o = ReflectOptimizer(cfg)
    o.relevance_map = {
        "q1": {"query": "statin side effects",
               "relevant": [("D_far", 2, 140), ("D_pool", 1, 40), ("D_none", 1, None)]},
        "q2": {"query": "aspirin dosage", "relevant": [("D_ok", 2, 3)]},
    }
    return o


def fake(payload) -> MagicMock:
    m = MagicMock(); m.content = payload if isinstance(payload, str) else json.dumps(payload)
    ch = MagicMock(); ch.message = m
    r = MagicMock(); r.choices = [ch]
    return r


# ── evidence rendering: the actual point of this arm ─────────────────────────

def test_evidence_flags_relevant_doc_beyond_depth():
    o = make()
    ev = o.render_evidence(["q1"], {"q1": []}, {"q1": 0.0}, depth=100)
    assert "first-stage rank 140" in ev
    assert "BEYOND original_query_depth=100" in ev


def test_evidence_distinguishes_the_three_failure_modes():
    o = make()
    per_q = {"q1": [{"rank": 1, "doc_id": "X", "rel": 0, "initial_rank": 2,
                     "reranked": True}]}
    ev = o.render_evidence(["q1"], per_q, {"q1": 0.0}, depth=100)
    assert "never entered the pool" in ev          # rank 140 > depth 100
    assert "entered the pool but lost downstream" in ev   # rank 40 <= depth
    assert "never retrieved by the first stage" in ev     # rank None


def test_evidence_marks_hits_and_shows_rerank_status():
    o = make()
    per_q = {"q2": [{"rank": 1, "doc_id": "D_ok", "rel": 2, "initial_rank": 3,
                     "reranked": True}]}
    ev = o.render_evidence(["q2"], per_q, {"q2": 1.0}, depth=100)
    assert "HIT " in ev
    assert "reranked=y" in ev
    assert "MISSED" not in ev


def test_evidence_includes_the_query_text():
    assert "statin side effects" in make().render_evidence(["q1"], {}, {"q1": 0.0}, 100)


# ── evidence query selection ─────────────────────────────────────────────────

def test_worst_queries_are_shown_with_a_contrast_sample():
    o = make(n_evidence_queries=4, n_contrast_queries=1)
    qids = o.choose_evidence_qids({"a": 0.9, "b": 0.1, "c": 0.2, "d": 0.3, "e": 0.8})
    assert qids[-1] == "a"                 # best, for contrast
    assert set(qids[:3]) == {"b", "c", "d"}  # three worst


def test_evidence_selection_before_any_trial_falls_back_to_the_map():
    o = make()
    assert set(o.choose_evidence_qids({})) <= set(o.relevance_map)


# ── parsing ──────────────────────────────────────────────────────────────────

def test_parse_merges_diagnosis_and_reasoning():
    params, why = ReflectOptimizer._parse(json.dumps(
        {"diagnosis": "doc beyond depth", "reasoning": "raise depth",
         "params": {"original_query_depth": 180}}))
    assert params["original_query_depth"] == 180
    assert "doc beyond depth" in why and "raise depth" in why


@pytest.mark.parametrize("raw", ["not json", "", None])
def test_parse_degrades_rather_than_raising(raw):
    assert ReflectOptimizer._parse(raw) == ({}, "")


def test_out_of_range_proposal_is_clamped():
    o = make()
    with patch("litellm.completion", return_value=fake(
            {"reasoning": "r", "params": {"original_query_depth": 99999}})):
        p, _, fb = o.propose([], set(), random.Random(0), "ev")
    assert not fb
    assert p["original_query_depth"] <= o.search_space.original_query_depth_range[1]


def test_api_failure_falls_back_to_random():
    o = make()
    with patch("litellm.completion", side_effect=RuntimeError("503")):
        p, why, fb = o.propose([], set(), random.Random(0), "ev")
    assert fb and "proposal call failed" in why
    assert p["reranker_type"] == "cross-encoder"


def test_duplicate_proposal_is_reasked_then_falls_back():
    from ragtune.tuning.llm_optimizer import validate_and_fix_params
    o = make()
    dup = {"reasoning": "d", "params": {"max_pool_size": 50}}
    seen = {o._sig(validate_and_fix_params(o.search_space, dup["params"]))}
    with patch("litellm.completion", side_effect=[fake(dup), fake(dup)]) as m:
        p, why, fb = o.propose([], seen, random.Random(0), "ev")
    assert m.call_count == 2 and fb and "duplicate" in why


# ── relevance map is built once from the fixed first stage ───────────────────

def test_relevance_map_records_first_stage_ranks():
    import pandas as pd
    o = make()
    o.relevance_map = {}
    eq = MagicMock(); eq.query_id="q9"; eq.query="q text"
    eq.qrels = {"A": 1, "B": 2, "MISSING": 1}
    ds = MagicMock(); ds.iter_queries.return_value=[eq]
    rt = MagicMock()
    rt.pt_transformer.transform.return_value = pd.DataFrame(
        {"docno": ["Z", "A", "Y", "B"], "score": [9.0, 8.0, 7.0, 6.0]})
    o.build_relevance_map(rt, ds)
    ranks = {d: r for d, _, r in o.relevance_map["q9"]["relevant"]}
    assert ranks["A"] == 2 and ranks["B"] == 4 and ranks["MISSING"] is None


def test_relevance_map_survives_a_broken_retriever():
    o = make(); o.relevance_map = {}
    eq = MagicMock(); eq.query_id="q9"; eq.query="t"; eq.qrels={"A":1}
    ds = MagicMock(); ds.iter_queries.return_value=[eq]
    rt = MagicMock(); rt.pt_transformer.transform.side_effect=RuntimeError("boom")
    o.build_relevance_map(rt, ds)      # must not raise
    assert o.relevance_map["q9"]["relevant"][0][2] is None


# ── loop ─────────────────────────────────────────────────────────────────────

def test_run_uses_exactly_the_budget_and_captures_evidence_qids():
    from ragtune.tuning.evaluator import TrialObjectives
    from ragtune.tuning.llm_optimizer import EvalResult, TraceAggregate, validate_and_fix_params
    o = make(); o.config.n_iterations = 3
    captured = []

    def _eval(controller, ds, n, ov, cost_model=None, capture_qids=None):
        captured.append(capture_qids)
        return EvalResult(
            objectives=TrialObjectives(ndcg_at_10=0.5, cost_usd=0.001,
                                       rerank_docs=20.0, latency_ms=10.0,
                                       queries_evaluated=4),
            trace=TraceAggregate(30.0, 0.6, 0.1, 0.0, 0.2, 0.0),
            query_ndcg={"q1": 0.1, "q2": 0.9},
            per_query={"q1": []},
        )

    with (
        patch("ragtune.tuning.reflect_optimizer.evaluate_controller_full", side_effect=_eval),
        patch.object(o, "build_relevance_map"),
        patch.object(type(o.search_space), "build_controller", return_value=MagicMock()),
        patch.object(o, "propose", side_effect=lambda h, s, r, e: (
            validate_and_fix_params(o.search_space, {"max_pool_size": 20 + len(h)}),
            "why", False)),
    ):
        hist = o.run(MagicMock(), MagicMock())
    assert len(hist) == 3
    assert all(c.ndcg_at_10 == 0.5 for c in hist)
    assert all(q is not None for q in captured), "evidence qids were not requested"


def test_build_failure_is_recorded_and_the_cell_continues():
    from ragtune.tuning.llm_optimizer import validate_and_fix_params
    o = make(); o.config.n_iterations = 2
    with (
        patch.object(o, "build_relevance_map"),
        patch.object(type(o.search_space), "build_controller",
                     side_effect=RuntimeError("component missing")),
        patch.object(o, "propose", side_effect=lambda h, s, r, e: (
            validate_and_fix_params(o.search_space, {}), "w", False)),
    ):
        hist = o.run(MagicMock(), MagicMock())
    assert len(hist) == 2
    assert all(c.error is not None for c in hist)
    assert all(c.mean_rerank_docs == float("inf") for c in hist)
