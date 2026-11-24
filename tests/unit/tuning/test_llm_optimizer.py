"""Unit tests for ragtune.tuning.llm_optimizer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from ragtune.tuning.llm_optimizer import (
    LLMAgentOptimizer,
    LLMCandidate,
    LLMOptimizerConfig,
    TraceAggregate,
    EvalResult,
    compute_pareto_front,
    evaluate_controller,
    _compute_param_correlations,
    _extract_query_trace,
    _aggregate_traces,
)
from ragtune.tuning.search_space import RAGtuneSearchSpace


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candidate(iteration=1, ndcg=0.5, cost=30.0, error=None, **params) -> LLMCandidate:
    base_params = {
        "reranker_type": "noop",
        "scheduler_type": "graceful-degradation",
        "reformulator_type": "identity",
        "estimator_type": "baseline",
        "feedback_type": "none",
        "original_query_depth": 10,
        "depth_per_reformulation": 5,
        "max_pool_size": 50,
        "near_duplicate_threshold": 0.8,
        "scheduler_batch_size": 5,
        "assembler_max_docs": 10,
        "budget_rerank_docs": 50,
        "budget_reformulations": 1,
        "ce_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "monot5_model": "castorini/monot5-base-msmarco",
        "monot5_batch_size": "16",
        "reformulator_model": "gpt-4o-mini",
        "reformulator_n_variants": 3,
        "similarity_model": "all-MiniLM-L6-v2",
        "min_reranked_for_regression": 3,
        "gd_llm_limit": 3,
        "gd_ce_limit": 10,
        "budget_stop_token_threshold": 0.9,
    }
    base_params.update(params)
    return LLMCandidate(
        iteration=iteration,
        params=base_params,
        ndcg_at_10=ndcg,
        mean_rerank_docs=cost,
        rationale="test",
        error=error,
    )


def _make_optimizer(overrides=None) -> LLMAgentOptimizer:
    cfg = LLMOptimizerConfig(
        llm_model="gpt-4o-mini",
        n_iterations=3,
        n_eval_queries=5,
        search_space_overrides=overrides or {
            "reranker_types": ["noop"],
            "reformulator_types": ["identity"],
            "estimator_types": ["baseline"],
            "scheduler_types": ["graceful-degradation"],
            "feedback_types": ["none"],
        },
    )
    return LLMAgentOptimizer(config=cfg)


# ── LLMCandidate.dominates ────────────────────────────────────────────────────

def test_dominates_strictly_better():
    a = _make_candidate(ndcg=0.8, cost=20.0)
    b = _make_candidate(ndcg=0.6, cost=30.0)
    assert a.dominates(b)
    assert not b.dominates(a)


def test_dominates_equal_not_dominating():
    a = _make_candidate(ndcg=0.5, cost=30.0)
    b = _make_candidate(ndcg=0.5, cost=30.0)
    assert not a.dominates(b)
    assert not b.dominates(a)


def test_dominates_trade_off_not_dominating():
    a = _make_candidate(ndcg=0.8, cost=50.0)
    b = _make_candidate(ndcg=0.4, cost=10.0)
    assert not a.dominates(b)
    assert not b.dominates(a)


def test_dominates_errored_never_dominates():
    a = _make_candidate(ndcg=1.0, cost=0.0, error="crash")
    b = _make_candidate(ndcg=0.1, cost=100.0)
    assert not a.dominates(b)


# ── compute_pareto_front ──────────────────────────────────────────────────────

def test_pareto_front_single():
    c = _make_candidate(ndcg=0.5, cost=30.0)
    assert compute_pareto_front([c]) == [c]


def test_pareto_front_dominated_excluded():
    a = _make_candidate(iteration=1, ndcg=0.8, cost=20.0)
    b = _make_candidate(iteration=2, ndcg=0.4, cost=40.0)  # dominated by a
    front = compute_pareto_front([a, b])
    assert a in front
    assert b not in front


def test_pareto_front_trade_off_both_kept():
    a = _make_candidate(iteration=1, ndcg=0.8, cost=50.0)
    b = _make_candidate(iteration=2, ndcg=0.4, cost=10.0)
    front = compute_pareto_front([a, b])
    assert len(front) == 2


def test_pareto_front_excludes_errored():
    a = _make_candidate(iteration=1, ndcg=0.0, cost=0.0, error="fail")
    b = _make_candidate(iteration=2, ndcg=0.5, cost=30.0)
    front = compute_pareto_front([a, b])
    assert a not in front
    assert b in front


def test_pareto_front_empty_input():
    assert compute_pareto_front([]) == []


# ── LLMOptimizerConfig ────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = LLMOptimizerConfig(name="test")
    assert cfg.n_iterations == 30
    assert cfg.llm_model == "gpt-4o-mini"
    assert cfg.temperature == 0.7


def test_config_from_yaml(tmp_path):
    import yaml
    data = {
        "name": "my-study",
        "llm_model": "claude-3-haiku",
        "n_iterations": 5,
        "n_eval_queries": 10,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    cfg = LLMOptimizerConfig.from_yaml(str(p))
    assert cfg.name == "my-study"
    assert cfg.llm_model == "claude-3-haiku"
    assert cfg.n_iterations == 5


# ── _validate_and_fix_params ──────────────────────────────────────────────────

def test_validate_clamps_ints():
    opt = _make_optimizer()
    sp = opt.search_space
    fixed = opt._validate_and_fix_params({
        "original_query_depth": 9999,   # way above max
        "budget_rerank_docs": -5,       # below min
    })
    assert fixed["original_query_depth"] <= sp.original_query_depth_range[1]
    assert fixed["budget_rerank_docs"] >= sp.budget_rerank_docs_range[0]


def test_validate_fixes_invalid_categorical():
    opt = _make_optimizer()
    fixed = opt._validate_and_fix_params({"reranker_type": "totally-invalid"})
    assert fixed["reranker_type"] in opt.search_space.reranker_types


def test_validate_fills_missing_params():
    opt = _make_optimizer()
    fixed = opt._validate_and_fix_params({})
    required_keys = [
        "reranker_type", "reformulator_type", "estimator_type", "scheduler_type",
        "feedback_type", "original_query_depth", "depth_per_reformulation",
        "max_pool_size", "near_duplicate_threshold", "scheduler_batch_size",
        "assembler_max_docs", "budget_rerank_docs", "budget_reformulations",
    ]
    for k in required_keys:
        assert k in fixed, f"Missing key: {k}"


def test_validate_monot5_batch_size_int_to_str():
    opt = _make_optimizer()
    fixed = opt._validate_and_fix_params({"monot5_batch_size": 16})
    assert fixed["monot5_batch_size"] == "16"


def test_validate_monot5_batch_size_invalid_clamps_to_nearest():
    opt = _make_optimizer()
    # 15 is closest to 16
    fixed = opt._validate_and_fix_params({"monot5_batch_size": 15})
    assert fixed["monot5_batch_size"] in ["4", "8", "16", "32"]


# ── _parse_response ───────────────────────────────────────────────────────────

def test_parse_response_valid_json():
    opt = _make_optimizer()
    raw = json.dumps({
        "rationale": "test reason",
        "params": {"reranker_type": "noop", "original_query_depth": 10},
    })
    params, rationale = opt._parse_response(raw, None)
    assert rationale == "test reason"
    assert "reranker_type" in params


def test_parse_response_with_markdown_fences():
    opt = _make_optimizer()
    raw = '```json\n{"rationale": "ok", "params": {}}\n```'
    params, rationale = opt._parse_response(raw, None)
    assert rationale == "ok"
    assert isinstance(params, dict)


def test_parse_response_malformed_returns_defaults():
    opt = _make_optimizer()
    params, rationale = opt._parse_response("not json at all", None)
    assert isinstance(params, dict)
    assert "reranker_type" in params  # filled by validate


# ── _build_system_prompt ──────────────────────────────────────────────────────

def test_system_prompt_contains_objectives():
    opt = _make_optimizer()
    prompt = opt._build_system_prompt("reranker")
    assert "ndcg_at_10" in prompt
    assert "mean_rerank_docs" in prompt


def test_system_prompt_lists_reranker_types():
    opt = _make_optimizer()
    prompt = opt._build_system_prompt("reranker")
    for rt in opt.search_space.reranker_types:
        assert rt in prompt


# ── LLMAgentOptimizer.run (mocked LLM + fake components) ─────────────────────

def _make_fake_litellm_response(params_dict: Dict[str, Any], rationale: str = "test") -> MagicMock:
    content = json.dumps({"rationale": rationale, "params": params_dict})
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _good_params() -> Dict[str, Any]:
    return {k: v for k, v in _make_candidate().params.items()}


def _make_eval_result(ndcg=0.5, cost=20.0, n_queries=5) -> EvalResult:
    from ragtune.tuning.evaluator import TrialObjectives
    return EvalResult(
        objectives=TrialObjectives(ndcg_at_10=ndcg, cost_usd=0.0, rerank_docs=cost,
                                   latency_ms=100.0, queries_evaluated=n_queries),
        trace=TraceAggregate(
            avg_pool_size=30.0,
            avg_pct_pool_reranked=0.6,
            feedback_stop_rate=0.1,
            retrieval_skip_rate=0.0,
            avg_rewrite_utility=0.2,
            avg_rerank_errors=0.0,
        ),
        # GEPA tracks per-query NDCG to build an instance-wise Pareto pool.
        query_ndcg={f"q{i}": ndcg for i in range(n_queries)},
    )


def test_run_returns_history_of_correct_length():
    opt = _make_optimizer()

    with (
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_full") as mock_eval,
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_minibatch") as mock_mini,
        patch.object(opt, "_propose", return_value=(_good_params(), "test rationale")),
    ):
        mock_eval.return_value = _make_eval_result()
        fake_retriever = MagicMock()
        fake_dataset = MagicMock()
        fake_dataset.queries = [MagicMock() for _ in range(5)]

        history = opt.run(fake_retriever, fake_dataset)

    assert mock_eval.called, "evaluation was not intercepted — patch target is stale"
    assert len(history) == opt.config.n_iterations


def test_run_stores_trace_in_candidate():
    opt = _make_optimizer()

    with (
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_full") as mock_eval,
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_minibatch") as mock_mini,
        patch.object(opt, "_propose", return_value=(_good_params(), "test")),
    ):
        mock_eval.return_value = _make_eval_result(ndcg=0.7, cost=15.0)
        fake_retriever = MagicMock()
        fake_dataset = MagicMock()
        fake_dataset.queries = [MagicMock()]
        history = opt.run(fake_retriever, fake_dataset)

    assert mock_eval.called, "evaluation was not intercepted — patch target is stale"
    assert all(c.trace is not None for c in history if not c.error)
    assert history[0].trace.avg_pool_size == 30.0


def test_run_records_error_on_build_failure():
    opt = _make_optimizer()

    with (
        patch("ragtune.tuning.search_space.RAGtuneSearchSpace.build_controller") as mock_build,
        patch.object(opt, "_propose", return_value=(_good_params(), "test")),
    ):
        mock_build.side_effect = RuntimeError("component not found")
        fake_retriever = MagicMock()
        fake_dataset = MagicMock()
        fake_dataset.queries = [MagicMock() for _ in range(5)]

        history = opt.run(fake_retriever, fake_dataset)

    assert all(c.error is not None for c in history)


def test_run_writes_pareto_files(tmp_path):
    cfg = LLMOptimizerConfig(
        name="test",
        n_iterations=2,
        n_eval_queries=2,
        output_dir=str(tmp_path / "out"),
        search_space_overrides={
            "reranker_types": ["noop"],
            "reformulator_types": ["identity"],
            "estimator_types": ["baseline"],
            "scheduler_types": ["graceful-degradation"],
            "feedback_types": ["none"],
        },
    )
    opt = LLMAgentOptimizer(config=cfg)

    with (
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_full") as mock_eval,
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_minibatch") as mock_mini,
        patch.object(opt, "_propose", return_value=(_good_params(), "rationale")),
    ):
        mock_eval.return_value = _make_eval_result(ndcg=0.6, cost=25.0)
        fake_retriever = MagicMock()
        fake_dataset = MagicMock()
        fake_dataset.queries = [MagicMock(), MagicMock()]
        opt.run(fake_retriever, fake_dataset)

    yaml_files = list((tmp_path / "out").glob("*.yaml"))
    assert len(yaml_files) >= 1


# ── TraceAggregate extraction ─────────────────────────────────────────────────

def _make_mock_output(events_spec: List[Dict]) -> MagicMock:
    """Build a fake ControllerOutput with specified trace events."""
    from ragtune.core.types import TraceEvent
    events = []
    for spec in events_spec:
        ev = MagicMock()
        ev.action = spec["action"]
        ev.details = spec.get("details", {})
        events.append(ev)
    output = MagicMock()
    output.trace.events = events
    output.final_budget_state = {"rerank_docs": 20.0}
    output.documents = []
    return output


def test_extract_query_trace_pool_init():
    output = _make_mock_output([
        {"action": "pool_init", "details": {"count": 40, "metrics": {"rewrite_utility_ratio": 0.25}}},
        {"action": "rerank_batch", "details": {"count": 10}},
        {"action": "rerank_batch", "details": {"count": 10}},
    ])
    sig = _extract_query_trace(output)
    assert sig["pool_size"] == 40
    assert sig["pct_pool_reranked"] == pytest.approx(0.5)
    assert sig["rewrite_utility"] == pytest.approx(0.25)
    assert sig["feedback_stop"] is False


def test_extract_query_trace_feedback_stop():
    output = _make_mock_output([
        {"action": "pool_init", "details": {"count": 20, "metrics": {}}},
        {"action": "feedback_stop"},
    ])
    sig = _extract_query_trace(output)
    assert sig["feedback_stop"] is True


def test_extract_query_trace_retrieval_skip():
    output = _make_mock_output([
        {"action": "pool_init", "details": {"count": 10, "metrics": {}}},
        {"action": "retrieval_skipped"},
    ])
    sig = _extract_query_trace(output)
    assert sig["retrieval_skip"] is True


def test_aggregate_traces_averages():
    per_query = [
        {"pool_size": 30, "pct_pool_reranked": 0.5, "feedback_stop": True,
         "retrieval_skip": False, "rewrite_utility": 0.2, "rerank_errors": 0},
        {"pool_size": 50, "pct_pool_reranked": 0.8, "feedback_stop": False,
         "retrieval_skip": True, "rewrite_utility": 0.4, "rerank_errors": 1},
    ]
    agg = _aggregate_traces(per_query)
    assert agg.avg_pool_size == pytest.approx(40.0)
    assert agg.avg_pct_pool_reranked == pytest.approx(0.65)
    assert agg.feedback_stop_rate == pytest.approx(0.5)
    assert agg.retrieval_skip_rate == pytest.approx(0.5)
    assert agg.avg_rewrite_utility == pytest.approx(0.3)
    assert agg.avg_rerank_errors == pytest.approx(0.5)


def test_aggregate_traces_empty():
    agg = _aggregate_traces([])
    assert agg.avg_pool_size == 0


# ── Parameter correlation ─────────────────────────────────────────────────────

def test_param_correlations_basic():
    history = [
        _make_candidate(iteration=1, ndcg=0.7, cost=20.0, reranker_type="cross-encoder"),
        _make_candidate(iteration=2, ndcg=0.6, cost=25.0, reranker_type="cross-encoder"),
        _make_candidate(iteration=3, ndcg=0.3, cost=5.0, reranker_type="noop"),
        _make_candidate(iteration=4, ndcg=0.35, cost=6.0, reranker_type="noop"),
    ]
    corr = _compute_param_correlations(history)
    assert "reranker_type" in corr
    assert "cross-encoder" in corr["reranker_type"]
    assert "noop" in corr["reranker_type"]
    ce_ndcg = corr["reranker_type"]["cross-encoder"][0]
    noop_ndcg = corr["reranker_type"]["noop"][0]
    assert ce_ndcg > noop_ndcg


def test_param_correlations_excludes_single_obs():
    history = [
        _make_candidate(iteration=1, ndcg=0.7, cost=20.0, reranker_type="cross-encoder"),
        _make_candidate(iteration=2, ndcg=0.3, cost=5.0, reranker_type="noop"),
        _make_candidate(iteration=3, ndcg=0.35, cost=6.0, reranker_type="noop"),
    ]
    corr = _compute_param_correlations(history)
    # cross-encoder has only 1 obs — should be excluded
    assert "cross-encoder" not in corr.get("reranker_type", {})
    assert "noop" in corr["reranker_type"]


def test_param_correlations_excludes_errored():
    history = [
        _make_candidate(iteration=1, ndcg=0.0, cost=0.0, error="crash", reranker_type="cross-encoder"),
        _make_candidate(iteration=2, ndcg=0.0, cost=0.0, error="crash", reranker_type="cross-encoder"),
        _make_candidate(iteration=3, ndcg=0.4, cost=10.0, reranker_type="noop"),
        _make_candidate(iteration=4, ndcg=0.45, cost=12.0, reranker_type="noop"),
    ]
    corr = _compute_param_correlations(history)
    assert "cross-encoder" not in corr.get("reranker_type", {})


# ── _build_user_message enrichment ────────────────────────────────────────────

def test_user_message_includes_trace_diagnostics():
    opt = _make_optimizer()
    trace = TraceAggregate(
        avg_pool_size=40.0, avg_pct_pool_reranked=0.45,
        feedback_stop_rate=0.2, retrieval_skip_rate=0.1,
        avg_rewrite_utility=0.05, avg_rerank_errors=0.0,
    )
    c = _make_candidate(iteration=1, ndcg=0.6, cost=20.0)
    c.trace = trace
    history = [c]
    pareto = compute_pareto_front(history)
    msg = opt._build_user_message(history, pareto, None, "reranker")
    assert "pool=" in msg
    assert "reranked=" in msg or "rnk%" in msg


def test_user_message_includes_rationale_replay():
    opt = _make_optimizer()
    c = _make_candidate(iteration=1, ndcg=0.6, cost=20.0)
    c.rationale = "I chose cross-encoder because it has higher accuracy"
    history = [c]
    pareto = compute_pareto_front(history)
    msg = opt._build_user_message(history, pareto, None, "reranker")
    assert "I chose cross-encoder" in msg


def test_user_message_includes_param_trends_when_enough_history():
    opt = _make_optimizer()
    history = [
        _make_candidate(iteration=i, ndcg=0.5 + i * 0.01, cost=20.0, reranker_type="noop")
        for i in range(5)
    ]
    pareto = compute_pareto_front(history)
    msg = opt._build_user_message(history, pareto, None, "reranker")
    assert "Parameter trends" in msg


def test_user_message_includes_diagnosis_hints():
    opt = _make_optimizer()
    history = [_make_candidate(iteration=1, ndcg=0.5, cost=20.0)]
    pareto = compute_pareto_front(history)
    msg = opt._build_user_message(history, pareto, None, "reranker")
    # Hints are scoped to the module being mutated. Assert the contract (a
    # module-scoped hints section mentioning the pool-utilisation signal) rather
    # than the exact threshold, which is a tuning constant free to change.
    assert "Diagnostic hints for [RERANKER]" in msg
    assert "pct_pool_reranked" in msg


# ── cost axis (regression: arms must share objective 2) ───────────────────────

def test_gepa_reports_cost_usd_when_a_cost_model_is_supplied():
    """
    Every arm compared against another must minimise the same objective 2, or
    the Pareto fronts differ in units and are not comparable. run_random once
    hardcoded rerank_docs while bayes/agenthpo used USD; this pins GEPA's half.
    """
    opt = _make_optimizer()
    opt.cost_model = MagicMock()

    with (
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_full") as mock_eval,
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_minibatch") as mock_mini,
        patch.object(opt, "_propose", return_value=(_good_params(), "r")),
    ):
        from ragtune.tuning.evaluator import TrialObjectives
        mock_eval.return_value = EvalResult(
            objectives=TrialObjectives(
                ndcg_at_10=0.5, cost_usd=0.000123, rerank_docs=42.0,
                latency_ms=100.0, queries_evaluated=2,
            ),
            trace=TraceAggregate(30.0, 0.6, 0.1, 0.0, 0.2, 0.0),
            query_ndcg={"q0": 0.5},
        )
        fake_ds = MagicMock(); fake_ds.queries = [MagicMock()]
        history = opt.run(MagicMock(), fake_ds)

    assert mock_eval.called
    # USD, not reranked documents
    assert all(c.mean_rerank_docs == 0.000123 for c in history if not c.error)


def test_gepa_reports_rerank_docs_when_no_cost_model():
    opt = _make_optimizer()
    assert opt.cost_model is None

    with (
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_full") as mock_eval,
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_minibatch") as mock_mini,
        patch.object(opt, "_propose", return_value=(_good_params(), "r")),
    ):
        mock_eval.return_value = _make_eval_result(ndcg=0.5, cost=42.0)
        fake_ds = MagicMock(); fake_ds.queries = [MagicMock()]
        history = opt.run(MagicMock(), fake_ds)

    assert all(c.mean_rerank_docs == 42.0 for c in history if not c.error)


# ── free-mutation ablation (gepafree arm) ─────────────────────────────────────

def test_free_mutation_lets_the_proposer_change_any_knob():
    """
    The gepafree ablation keeps GEPA's reflective context but drops the
    one-module-per-iteration restriction. Isolates per-module mutation as the
    single difference from the agenthpo arm.
    """
    from ragtune.tuning.llm_optimizer import FREE_MODULE, _MODULE_PARAM_KEYS
    opt = _make_optimizer()
    opt.config.free_mutation = True
    sysp = opt._build_system_prompt(FREE_MODULE)
    assert "change ANY parameter" in sysp
    assert "module only" not in sysp
    # the union covers strictly more knobs than any single module
    assert len(_MODULE_PARAM_KEYS[FREE_MODULE]) > max(
        len(v) for k, v in _MODULE_PARAM_KEYS.items() if k != FREE_MODULE)


def test_per_module_restriction_is_the_default():
    from ragtune.tuning.llm_optimizer import LLMOptimizerConfig
    assert LLMOptimizerConfig(name="t").free_mutation is False
    opt = _make_optimizer()
    sysp = opt._build_system_prompt("reranker")
    assert "module only" in sysp
    assert "change ANY parameter" not in sysp


def test_free_mutation_shows_hints_for_every_module():
    from ragtune.tuning.llm_optimizer import FREE_MODULE, _MODULE_CYCLE
    opt = _make_optimizer()
    opt.config.free_mutation = True
    c = _make_candidate(iteration=1, ndcg=0.6, cost=20.0)
    msg = opt._build_user_message([c], compute_pareto_front([c]), c, FREE_MODULE)
    for mod in _MODULE_CYCLE:
        assert f"[{mod}]" in msg
    assert "coordinated changes" in msg


def test_per_module_shows_only_that_module_hints():
    opt = _make_optimizer()
    c = _make_candidate(iteration=1, ndcg=0.6, cost=20.0)
    msg = opt._build_user_message([c], compute_pareto_front([c]), c, "reranker")
    assert "Diagnostic hints for [RERANKER]" in msg
    assert "mutate ONLY the [RERANKER]" in msg


def test_free_mutation_run_uses_the_free_target():
    """run() must stop cycling modules when free_mutation is set."""
    from ragtune.tuning.llm_optimizer import FREE_MODULE
    opt = _make_optimizer()
    opt.config.free_mutation = True
    opt.config.n_startup = 0
    seen = []

    def spy(history, pool, parent, target_module):
        seen.append(target_module)
        return _good_params(), "r"

    with (
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_full") as mock_eval,
        patch("ragtune.tuning.llm_optimizer.evaluate_controller_minibatch", return_value=0.9),
        patch.object(opt, "_propose", side_effect=spy),
    ):
        mock_eval.return_value = _make_eval_result()
        fake_ds = MagicMock(); fake_ds.queries = [MagicMock()]
        opt.run(MagicMock(), fake_ds)

    assert seen, "proposer was never called"
    assert set(seen) == {FREE_MODULE}


# ── calibrated minibatch screen ───────────────────────────────────────────────

def test_screen_gate_scales_with_the_parent_standard_error():
    """
    The old gate was a hardcoded 0.02 NDCG delta on 10 queries, ~4x inside the
    sampling noise, so it discarded candidates on coin flips. It is now
    k * SE(parent on the minibatch).
    """
    import statistics as st
    opt = _make_optimizer()
    opt.config.screen_k_sigma = 2.0
    qids = {f"q{i}" for i in range(10)}
    spread = {f"q{i}": v for i, v in enumerate([0.0, 1.0] * 5)}   # high variance
    tight = {f"q{i}": 0.5 for i in range(10)}                      # zero variance

    parent_hi = _make_candidate(ndcg=0.5, cost=20.0)
    parent_hi.query_ndcg = spread
    parent_lo = _make_candidate(ndcg=0.5, cost=20.0)
    parent_lo.query_ndcg = tight

    def gate_for(parent):
        # child scores exactly at the parent mean: must never be rejected
        with patch("ragtune.tuning.llm_optimizer.evaluate_controller_minibatch",
                   return_value=0.5), \
             patch.object(type(opt.search_space), "build_controller",
                          return_value=MagicMock()), \
             patch("ragtune.tuning.llm_optimizer.evaluate_controller_full") as mf:
            mf.return_value = _make_eval_result()
            c = opt._minibatch_screen_then_eval(
                1, _good_params(), MagicMock(), MagicMock(), qids, parent,
                "r", "reranker")
        return c

    assert gate_for(parent_hi).error is None, "tie rejected under high variance"
    assert gate_for(parent_lo).error is None, "tie rejected under zero variance"
    # a genuinely worse child is still rejected
    with patch("ragtune.tuning.llm_optimizer.evaluate_controller_minibatch",
               return_value=0.0), \
         patch.object(type(opt.search_space), "build_controller",
                      return_value=MagicMock()):
        c = opt._minibatch_screen_then_eval(
            1, _good_params(), MagicMock(), MagicMock(), qids, parent_lo,
            "r", "reranker")
    assert c.error is not None and "screen" in c.error


def test_screen_gate_has_a_floor_for_zero_variance_parents():
    opt = _make_optimizer()
    assert opt.config.screen_min_gate > 0


def test_gepa_history_budget_matches_the_other_llm_arms():
    """agenthpo and reflect show the full history; GEPA must not show less."""
    from ragtune.tuning.llm_optimizer import LLMOptimizerConfig
    assert LLMOptimizerConfig(name="t").max_history_in_prompt >= 30
