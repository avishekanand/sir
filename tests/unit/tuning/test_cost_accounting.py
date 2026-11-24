"""
Cost accounting end-to-end: CostTracker recording -> snapshot -> QueryUsage ->
dollars. Covers the budget-exhaustion edge cases Rule 4 requires for cost logic.
No network, no real models.
"""

import pytest

from ragtune.core.budget import CostBudget, CostTracker
from ragtune.core.types import ControllerTrace
from ragtune.tuning.cost_model import (
    SBERT_BATCH_CAP,
    CostModel,
    CostModelConfig,
    QueryUsage,
)

CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tracker(**limits) -> CostTracker:
    return CostTracker(CostBudget(limits=limits or {"rerank_docs": 50}), ControllerTrace())


def model() -> CostModel:
    return CostModel(CostModelConfig(
        n_ctx=355,
        ce_ms_per_doc={CE: {1: 4.629, 2: 3.525, 5: 1.787, 10: 1.238, 20: 1.184}},
        gpu_usd_per_hour=0.80,
        cpu_usd_per_hour=0.0,
        token_prices={"anthropic/claude-opus-5": (5.0, 25.0)},
    ))


# ── tracker recording ─────────────────────────────────────────────────────────

def test_llm_usage_accumulates_per_model():
    t = tracker()
    t.record_llm_usage("anthropic/claude-opus-5", 96, 95)
    t.record_llm_usage("anthropic/claude-opus-5", 10, 10)
    t.record_llm_usage("anthropic/claude-haiku-4-5", 69, 40)
    snap = t.snapshot()
    assert snap["llm_usage"]["anthropic/claude-opus-5"] == (106, 105)
    assert snap["llm_usage"]["anthropic/claude-haiku-4-5"] == (69, 40)


def test_rerank_calls_recorded_individually_not_summed():
    """Each call's size sets its own per-doc rate, so sizes must not be merged."""
    t = tracker()
    t.record_rerank_call(CE, 20)
    t.record_rerank_call(CE, 5)
    assert t.snapshot()["rerank_calls"] == [(CE, 20), (CE, 5)]


def test_empty_rerank_call_is_ignored():
    t = tracker()
    t.record_rerank_call(CE, 0)
    assert t.snapshot()["rerank_calls"] == []


def test_recording_does_not_touch_the_budget():
    """Priced quantities are not budgeted; `tokens` in consumed is the assembly
    context budget, a different thing entirely."""
    t = tracker(rerank_docs=50, tokens=4000)
    t.record_llm_usage("anthropic/claude-opus-5", 500, 500)
    t.record_rerank_call(CE, 20)
    assert t.consumed.get("tokens", 0) == 0
    assert t.consumed.get("rerank_docs", 0) == 0
    assert not t.is_exhausted()


# ── snapshot -> QueryUsage ────────────────────────────────────────────────────

def test_query_usage_round_trips_from_budget_state():
    t = tracker()
    t.try_consume_rerank(25)
    t.record_rerank_call(CE, 20)
    t.record_rerank_call(CE, 5)
    t.record_llm_usage("anthropic/claude-opus-5", 96, 95)

    u = QueryUsage.from_budget_state(t.snapshot(), retrieval_ms=147.0)
    assert u.rerank_docs == 25
    assert u.rerank_calls == [(CE, 20), (CE, 5)]
    assert u.llm_tokens == {"anthropic/claude-opus-5": (96, 95)}
    assert u.reranker_model == CE
    assert u.retrieval_ms == 147.0
    assert u.used_api_model


def test_cost_sums_over_calls_at_each_call_own_rate():
    cm, t = model(), tracker()
    t.record_rerank_call(CE, 20)
    t.record_rerank_call(CE, 5)
    u = QueryUsage.from_budget_state(t.snapshot())
    expected_ms = 20 * 1.184 + 5 * 1.787
    assert cm.gpu_usd(u) == pytest.approx(expected_ms * 0.80 / 3_600_000.0)


def test_calls_larger_than_sbert_cap_are_priced_at_the_cap():
    """predict() splits a long list into batches of SBERT_BATCH_CAP."""
    cm, t = model(), tracker()
    t.record_rerank_call(CE, 100)
    u = QueryUsage.from_budget_state(t.snapshot())
    at_cap = cm._ce_ms_per_doc(CE, SBERT_BATCH_CAP)
    assert cm.gpu_usd(u) == pytest.approx(100 * at_cap * 0.80 / 3_600_000.0)


def test_flops_sum_over_calls_and_ignore_batching():
    cm, t = model(), tracker()
    t.record_rerank_call(CE, 20)
    t.record_rerank_call(CE, 5)
    split = cm.flops_per_query(QueryUsage.from_budget_state(t.snapshot()))

    t2 = tracker()
    t2.record_rerank_call(CE, 25)
    whole = cm.flops_per_query(QueryUsage.from_budget_state(t2.snapshot()))
    assert split == pytest.approx(whole)


# ── budget exhaustion (Rule 4: required) ──────────────────────────────────────

def test_exhausted_budget_before_any_rerank_costs_nothing():
    cm = model()
    t = tracker(rerank_docs=0)
    assert not t.try_consume_rerank(10)        # denied
    u = QueryUsage.from_budget_state(t.snapshot())
    assert u.rerank_calls is None              # reranker never ran
    assert cm.gpu_usd(u) == 0.0
    assert cm.cost_usd_per_query(u) == 0.0
    assert cm.flops_per_query(u) == 0.0


def test_partially_exhausted_budget_charges_only_what_ran():
    """A trial cut off mid-way is charged for the calls that happened."""
    cm = model()
    t = tracker(rerank_docs=25)
    t.try_consume_rerank(20)
    t.record_rerank_call(CE, 20)
    assert not t.try_consume_rerank(20)        # over limit, reranker not invoked
    u = QueryUsage.from_budget_state(t.snapshot())
    assert u.rerank_calls == [(CE, 20)]
    assert cm.gpu_usd(u) == pytest.approx(20 * 1.184 * 0.80 / 3_600_000.0)


def test_llm_cost_still_charged_when_rerank_budget_is_exhausted():
    """Exhausting the rerank budget must not make an API call retroactively free."""
    cm = model()
    t = tracker(rerank_docs=0)
    t.record_llm_usage("anthropic/claude-opus-5", 96, 95)
    u = QueryUsage.from_budget_state(t.snapshot())
    assert cm.gpu_usd(u) == 0.0
    assert cm.cost_usd_per_query(u) == pytest.approx((96 * 5.0 + 95 * 25.0) / 1e6)
    assert cm.flops_per_query(u) is None       # an API model participated


def test_exhausted_latency_budget_leaves_cost_well_defined():
    cm = model()
    t = tracker(latency_ms=0)
    t.record_rerank_call(CE, 5)
    u = QueryUsage.from_budget_state(t.snapshot(), retrieval_ms=999.0)
    assert cm.cost_usd_per_query(u) > 0


def test_denied_rerank_attempts_are_not_charged():
    """
    CostTracker.try_consume accumulates before checking the limit, so
    consumed["rerank_docs"] includes work that was denied and never ran. The
    legacy objective read that counter directly and therefore over-charged
    budget-exhausted trials; cost must follow rerank_calls instead.
    """
    cm = model()
    t = tracker(rerank_docs=20)
    t.try_consume_rerank(20)
    t.record_rerank_call(CE, 20)
    assert not t.try_consume_rerank(30)          # denied, reranker not invoked

    snap = t.snapshot()
    assert snap["rerank_docs"] == 50             # counter inflated by the denial
    u = QueryUsage.from_budget_state(snap)
    assert u.rerank_docs == 20                   # only what actually ran
    assert cm.gpu_usd(u) == pytest.approx(20 * 1.184 * 0.80 / 3_600_000.0)
