import pytest
from ragtune.components.feedback import ReformIRConvergenceFeedback
from ragtune.core.types import RemainingBudgetView


def budget():
    return RemainingBudgetView(remaining_tokens=1000, remaining_rerank_docs=20, remaining_rerank_calls=5)


def test_first_call_always_continues():
    """On the very first call prev_weights is None — never stop."""
    fb = ReformIRConvergenceFeedback(convergence_threshold=0.01)
    stop, _ = fb.should_stop({}, budget(), {"reformir_weights": {"original": 0.5}})
    assert stop is False


def test_no_weights_in_estimates_always_continues():
    """If estimates carries no reformir_weights, never stop."""
    fb = ReformIRConvergenceFeedback()
    fb.should_stop({}, budget(), {})
    stop, _ = fb.should_stop({}, budget(), {})
    assert stop is False


def test_converged_weights_triggers_stop():
    """Weights barely changed → converged → stop."""
    fb = ReformIRConvergenceFeedback(convergence_threshold=0.05)
    w1 = {"original": 0.500, "rewrite_0": 0.300}
    w2 = {"original": 0.501, "rewrite_0": 0.299}  # max delta = 0.001 < 0.05

    fb.should_stop({}, budget(), {"reformir_weights": w1})  # prime prev_weights
    stop, reason = fb.should_stop({}, budget(), {"reformir_weights": w2})

    assert stop is True
    assert "converged" in reason


def test_large_delta_continues():
    """Weights changed significantly → not converged → continue."""
    fb = ReformIRConvergenceFeedback(convergence_threshold=0.05)
    w1 = {"original": 0.2, "rewrite_0": 0.8}
    w2 = {"original": 0.7, "rewrite_0": 0.3}  # max delta = 0.5 > 0.05

    fb.should_stop({}, budget(), {"reformir_weights": w1})
    stop, _ = fb.should_stop({}, budget(), {"reformir_weights": w2})

    assert stop is False


def test_new_source_key_in_second_call_handled():
    """A new source appearing in weights is treated as delta from 0."""
    fb = ReformIRConvergenceFeedback(convergence_threshold=0.05)
    w1 = {"original": 0.5}
    w2 = {"original": 0.501, "rewrite_0": 0.001}  # new key; max delta ≈ 0.001 < 0.05

    fb.should_stop({}, budget(), {"reformir_weights": w1})
    stop, _ = fb.should_stop({}, budget(), {"reformir_weights": w2})

    assert stop is True


def test_weights_update_each_iteration():
    """prev_weights is updated every call so convergence compares consecutive pairs."""
    fb = ReformIRConvergenceFeedback(convergence_threshold=0.05)
    w1 = {"original": 0.1}
    w2 = {"original": 0.8}   # large jump — not converged
    w3 = {"original": 0.801}  # tiny jump — converged

    fb.should_stop({}, budget(), {"reformir_weights": w1})
    stop2, _ = fb.should_stop({}, budget(), {"reformir_weights": w2})
    stop3, _ = fb.should_stop({}, budget(), {"reformir_weights": w3})

    assert stop2 is False
    assert stop3 is True
