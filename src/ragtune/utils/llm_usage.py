"""Record tokens billed by API models, for the cost model's dollar term."""

from __future__ import annotations

from typing import Any


def record_llm_usage(context: Any, model: str, response: Any) -> None:
    """
    Feed billed tokens from a litellm response to the request's CostTracker.

    Best-effort by design: a provider that omits usage must not break retrieval,
    but the trace records the gap so a silently unpriced call is visible rather
    than being counted as free. See specs/spec-cost-model.md section 5.
    """
    tracker = getattr(context, "tracker", None)
    if tracker is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        tracker.trace.add("cost", "llm_usage_missing", model=model)
        return
    tracker.record_llm_usage(
        model,
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )
