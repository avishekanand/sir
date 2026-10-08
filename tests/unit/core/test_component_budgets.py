"""Unit tests for component-scoped budgets (specs/component-scoped-budgets.md)."""

import pytest

from ragtune.core.budget import (
    CostBudget,
    CostTracker,
    group_by_component,
    parse_scoped_key,
)
from ragtune.core.types import UNLIMITED, ControllerTrace, RemainingBudgetView


def make_tracker(limits):
    return CostTracker(CostBudget(limits=limits), ControllerTrace())


# --- Key validation ---

def test_valid_scoped_key_parses():
    assert parse_scoped_key("rerank.latency_ms") == ("rerank", "latency_ms")


@pytest.mark.parametrize("key", ["rerenk.latency_ms", "rerank.latency", "rerank.docs.extra", "rerank."])
def test_invalid_scoped_key_rejected_at_budget_construction(key):
    with pytest.raises(ValueError, match="Unknown budget"):
        CostBudget(limits={key: 1})


def test_global_keys_stay_free_form():
    budget = CostBudget(limits={"tokens": 10, "my_custom_key": 3})
    assert budget.limits["my_custom_key"] == 3


# --- Remaining view ---

def test_absent_limits_report_unlimited():
    view = make_tracker({"latency_ms": 500}).remaining_view()
    assert view.remaining_tokens == UNLIMITED
    assert view.remaining_rerank_docs == UNLIMITED
    assert view.remaining_rerank_calls == UNLIMITED


def test_explicit_limits_unchanged_and_infinite_is_unlimited():
    tracker = make_tracker({"rerank_docs": 10, "tokens": float("inf")})
    tracker.try_consume("rerank_docs", 4)
    view = tracker.remaining_view()
    assert view.remaining_rerank_docs == 6
    assert view.remaining_tokens == UNLIMITED


def test_view_carries_scoped_remainders():
    tracker = make_tracker({"rerank.tokens": 300, "assembly.tokens": 50})
    tracker.try_consume("rerank.tokens", 100)
    assert tracker.remaining_view().scoped == {"rerank.tokens": 200, "assembly.tokens": 50}


def test_for_component_narrows_to_tighter_limit():
    view = RemainingBudgetView(
        remaining_tokens=1000,
        remaining_rerank_docs=UNLIMITED,
        remaining_rerank_calls=5,
        scoped={"rerank.tokens": 300, "rerank.docs": 4, "assembly.tokens": 10},
    )
    rerank = view.for_component("rerank")
    assert (rerank.remaining_tokens, rerank.remaining_rerank_docs, rerank.remaining_rerank_calls) == (300, 4, 5)
    assert view.for_component("estimation").remaining_tokens == 1000


# --- Measurement and attribution ---

def test_measure_records_inclusive_latency_for_nested_scopes(fake_clock):
    tracker = make_tracker({})
    with tracker.measure("estimation"):
        fake_clock.advance_ms(10)
        with tracker.measure("embedding"):
            fake_clock.advance_ms(20)
    assert tracker.consumed["estimation.latency_ms"] == pytest.approx(30)
    assert tracker.consumed["embedding.latency_ms"] == pytest.approx(20)


def test_measure_accumulates_across_calls_and_records_on_error(fake_clock):
    tracker = make_tracker({})
    with tracker.measure("rerank"):
        fake_clock.advance_ms(5)
    with pytest.raises(RuntimeError):
        with tracker.measure("rerank"):
            fake_clock.advance_ms(7)
            raise RuntimeError("reranker crashed")
    assert tracker.consumed["rerank.latency_ms"] == pytest.approx(12)


def test_consumption_is_attributed_to_innermost_component():
    tracker = make_tracker({})
    with tracker.measure("assembly"):
        tracker.try_consume_tokens(5)
        with tracker.measure("embedding"):
            tracker.try_consume_tokens(2)
    tracker.try_consume_tokens(1)  # outside any component: global only
    assert tracker.consumed["tokens"] == 8
    assert tracker.consumed["assembly.tokens"] == 5
    assert tracker.consumed["embedding.tokens"] == 2


def test_scoped_limit_denies_even_when_global_allows():
    tracker = make_tracker({"tokens": 100, "assembly.tokens": 10})
    with tracker.measure("assembly"):
        assert tracker.try_consume_tokens(8) is True
        assert tracker.try_consume_tokens(8) is False
    assert tracker.consumed["tokens"] == 16


def test_rerank_cost_object_is_attributed_to_rerank_dimensions():
    from ragtune.core.types import CostObject

    tracker = make_tracker({})
    with tracker.measure("rerank"):
        tracker.consume(CostObject(tokens=1024, docs=2, calls=1))
    assert tracker.consumed["rerank.tokens"] == 1024
    assert tracker.consumed["rerank.docs"] == 2
    assert tracker.consumed["rerank.calls"] == 1


def test_unlimited_attribution_adds_no_trace_events():
    tracker = make_tracker({})
    with tracker.measure("assembly"):
        tracker.try_consume_tokens(5)
    actions = [e.action for e in tracker.trace.events]
    assert actions == ["consume_tokens_unlimited"]


# --- Exhaustion ---

def test_component_exhaustion_is_scoped(fake_clock):
    tracker = make_tracker({"rerank_docs": 1, "rerank.latency_ms": 50})
    tracker.try_consume("rerank_docs", 1)  # global exhausted
    assert tracker.is_exhausted() is True
    assert tracker.component_exhausted("retrieval") is False
    assert tracker.component_exhausted("rerank") is False

    tracker = make_tracker({"rerank.latency_ms": 50})
    with tracker.measure("rerank"):
        fake_clock.advance_ms(60)
    assert tracker.is_exhausted() is False  # global view is unaffected
    assert tracker.is_exhausted("rerank") is True
    assert tracker.component_exhausted("estimation") is False


def test_zero_scoped_limit_is_exhausted_from_the_start():
    assert make_tracker({"reformulation.calls": 0}).component_exhausted("reformulation") is True


def test_group_by_component_ignores_global_and_unknown_keys():
    state = {"tokens": 9, "latency": 3.0, "rerank.docs": 4, "rerank.latency_ms": 1.5, "foo.bar": 1}
    assert group_by_component(state) == {"rerank": {"docs": 4, "latency_ms": 1.5}}
