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
    compute_pareto_front,
    evaluate_controller,
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
    params, rationale = opt._parse_response(raw)
    assert rationale == "test reason"
    assert "reranker_type" in params


def test_parse_response_with_markdown_fences():
    opt = _make_optimizer()
    raw = '```json\n{"rationale": "ok", "params": {}}\n```'
    params, rationale = opt._parse_response(raw)
    assert rationale == "ok"
    assert isinstance(params, dict)


def test_parse_response_malformed_returns_defaults():
    opt = _make_optimizer()
    params, rationale = opt._parse_response("not json at all")
    assert isinstance(params, dict)
    assert "reranker_type" in params  # filled by validate


# ── _build_system_prompt ──────────────────────────────────────────────────────

def test_system_prompt_contains_objectives():
    opt = _make_optimizer()
    prompt = opt._build_system_prompt()
    assert "ndcg_at_10" in prompt
    assert "mean_rerank_docs" in prompt


def test_system_prompt_lists_reranker_types():
    opt = _make_optimizer()
    prompt = opt._build_system_prompt()
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


def test_run_returns_history_of_correct_length():
    opt = _make_optimizer()

    with (
        patch("ragtune.tuning.llm_optimizer.evaluate_controller") as mock_eval,
        patch.object(opt, "_propose", return_value=(_good_params(), "test rationale")),
    ):
        from ragtune.tuning.evaluator import TrialObjectives
        mock_eval.return_value = TrialObjectives(
            ndcg_at_10=0.5, rerank_docs=20.0, latency_ms=100.0, queries_evaluated=5
        )

        fake_retriever = MagicMock()
        fake_dataset = MagicMock()
        fake_dataset.queries = [MagicMock() for _ in range(5)]

        history = opt.run(fake_retriever, fake_dataset)

    assert len(history) == opt.config.n_iterations


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
        patch("ragtune.tuning.llm_optimizer.evaluate_controller") as mock_eval,
        patch.object(opt, "_propose", return_value=(_good_params(), "rationale")),
    ):
        from ragtune.tuning.evaluator import TrialObjectives
        mock_eval.return_value = TrialObjectives(
            ndcg_at_10=0.6, rerank_docs=25.0, latency_ms=50.0, queries_evaluated=2
        )

        fake_retriever = MagicMock()
        fake_dataset = MagicMock()
        fake_dataset.queries = [MagicMock(), MagicMock()]

        opt.run(fake_retriever, fake_dataset)

    yaml_files = list((tmp_path / "out").glob("*.yaml"))
    assert len(yaml_files) >= 1
