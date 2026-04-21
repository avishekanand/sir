# Controller Trace

## What It Is

Defined in [`core/types.py`](../../src/ragtune/core/types.py) (line 48). A trace is a flat list of `TraceEvent` objects — one appended per decision point across an entire query run.

```python
class TraceEvent(BaseModel):
    timestamp: float          # unix time
    component: str            # "controller" | "budget" | "estimator" | "reranker" | "reformulator"
    action: str               # what happened
    details: Dict[str, Any]   # event-specific payload
```

Every component that receives `RAGtuneContext` can write to the trace via `context.tracker.trace.add(component, action, **kwargs)`. The trace is the single audit log for a run — you can reconstruct exactly what happened and why.

The trace lives inside `ControllerOutput.trace` and is returned to the caller after every `run()`.

---

## Event Catalogue

### `component = "controller"`

Source: [`core/controller.py`](../../src/ragtune/core/controller.py)

| action | when | key details |
|---|---|---|
| `pool_init` | after all retrieval rounds complete | `count`, `reformulations`, `metrics` (overlap stats) |
| `reformulation_cache_hit` | same query reformulated earlier in session | `query` |
| `retrieval_skipped` | reformulation budget exhausted before a round | `query`, `reason="budget_exhausted"` |
| `feedback_stop` | feedback component triggered early stop | `reason` |
| `rerank_batch` | successful rerank iteration | `count`, `strategy`, `doc_ids`, `dropped_ids` |
| `rerank_error` | reranker threw an exception | `error`, `doc_ids` |

### `component = "budget"`

Source: [`core/budget.py`](../../src/ragtune/core/budget.py)

| action | when | key details |
|---|---|---|
| `consume_rerank_docs` | docs consumed normally | `count`, `total` |
| `consume_retrieval_calls` | retrieval call logged | `count`, `total` |
| `consume_tokens` | tokens consumed | `count`, `total` |
| `over_limit_rerank_docs` | batch pushed total over cap | `count`, `total`, `limit` |
| `deny_{cost_type}` | latency wall hit, operation blocked | `reason="latency_exceeded"`, `elapsed` |
| `consume_{cost_type}_unlimited` | consumption with no limit configured | `count`, `total` |

### `component = "estimator"`

Source: [`components/estimators.py`](../../src/ragtune/components/estimators.py)

| action | when | key details |
|---|---|---|
| `reformir_weights_updated` | ReformIR refitted its regression | `weights` (per-source float), `n_reranked` |

### `component = "reformulator"`

Source: [`components/reformulators.py`](../../src/ragtune/components/reformulators.py)

| action | when | key details |
|---|---|---|
| `reformir_querygym_ok` | ReformIR generated rewrites successfully | `count` |
| `reformir_querygym_fallback` | QueryGym failed, returned nothing | `reason` |
| `reformir_llm_error` | ReformIR LLM call threw | `error`, `model` |
| `llm_error` | LLMReformulator threw | `error`, `model` |

### `component = "reranker"`

Source: [`components/rerankers.py`](../../src/ragtune/components/rerankers.py)

| action | when | key details |
|---|---|---|
| `ollama_error` | Ollama call failed | `error`, `model` |

---

## Example: Healthy 3-Iteration Run

Query: *"What are the side effects of aspirin?"*  
Budget: 50 rerank docs, 2000 ms latency  
Estimator: `reformir`, Scheduler: `active-learning`

```json
[
  {"component": "budget",     "action": "consume_retrieval_calls",  "details": {"count": 1, "total": 1}},
  {"component": "budget",     "action": "consume_retrieval_calls",  "details": {"count": 1, "total": 2}},
  {"component": "controller", "action": "pool_init", "details": {
      "count": 32,
      "reformulations": ["What are the side effects of aspirin?", "aspirin adverse reactions"],
      "metrics": {"total_unique_docs": 32, "overlap_count": 8, "rewrite_utility_ratio": 0.56}
  }},

  {"component": "budget",     "action": "consume_rerank_docs",  "details": {"count": 5, "total": 5}},
  {"component": "controller", "action": "rerank_batch", "details": {
      "count": 5, "strategy": "cross_encoder",
      "doc_ids": ["d12", "d3", "d27", "d8", "d19"], "dropped_ids": null
  }},
  {"component": "estimator",  "action": "reformir_weights_updated", "details": {
      "weights": {"original": 0.42, "rewrite_0": 0.71, "rewrite_1": 0.19},
      "n_reranked": 5
  }},

  {"component": "budget",     "action": "consume_rerank_docs",  "details": {"count": 5, "total": 10}},
  {"component": "controller", "action": "rerank_batch", "details": {
      "count": 5, "strategy": "cross_encoder",
      "doc_ids": ["d1", "d5", "d31", "d22", "d7"], "dropped_ids": null
  }},
  {"component": "estimator",  "action": "reformir_weights_updated", "details": {
      "weights": {"original": 0.41, "rewrite_0": 0.72, "rewrite_1": 0.18},
      "n_reranked": 10
  }},

  {"component": "budget",     "action": "consume_rerank_docs",  "details": {"count": 5, "total": 15}},
  {"component": "controller", "action": "rerank_batch", "details": {
      "count": 5, "strategy": "llm",
      "doc_ids": ["d2", "d14", "d6", "d30", "d11"], "dropped_ids": null
  }}
]
```

Strategy escalated to `"llm"` in round 3 — the top two candidates had a priority gap < 5%, so the scheduler upgraded to a more expensive model to break the tie.

---

## Example: Budget Exhausted Mid-Run

The `over_limit_*` event fires when a batch pushes the running total past the cap. The controller exits on the next `is_exhausted()` check.

```json
[
  {"component": "controller", "action": "pool_init", "details": {"count": 50, "reformulations": [...], "metrics": {...}}},

  {"component": "budget",     "action": "consume_rerank_docs",    "details": {"count": 5, "total": 48}},
  {"component": "controller", "action": "rerank_batch",           "details": {"count": 5, "strategy": "cross_encoder", "doc_ids": [...], "dropped_ids": null}},

  {"component": "budget",     "action": "over_limit_rerank_docs", "details": {"count": 5, "total": 53, "limit": 50}},
  {"component": "controller", "action": "rerank_batch",           "details": {"count": 3, "strategy": "cross_encoder", "doc_ids": [...], "dropped_ids": null}}
]
```

Note: `over_limit_*` does not stop the run immediately — `is_exhausted()` in the loop condition catches it at the top of the next iteration.

---

## Example: Latency Wall Hit

`deny_*` fires when `elapsed_ms > latency_ms` limit. The operation is blocked before it starts.

```json
[
  {"component": "controller", "action": "pool_init", "details": {"count": 40, ...}},
  {"component": "budget",     "action": "consume_rerank_docs", "details": {"count": 5, "total": 5}},
  {"component": "controller", "action": "rerank_batch",        "details": {"count": 5, "strategy": "cross_encoder", ...}},

  {"component": "budget",     "action": "deny_rerank_docs", "details": {
      "reason": "latency_exceeded", "elapsed": 2043.7
  }},
  {"component": "controller", "action": "rerank_error", "details": {
      "error": "...", "doc_ids": ["d4", "d9"]
  }}
]
```

---

## Example: Feedback Early Stop

`ReformIRConvergenceFeedback` reads estimator weights from `EstimatorOutput.metadata` each iteration. When consecutive weight vectors shift by less than a threshold, it signals the controller to stop — even if budget remains.

```json
[
  {"component": "estimator",  "action": "reformir_weights_updated", "details": {
      "weights": {"original": 0.41, "rewrite_0": 0.71}, "n_reranked": 10
  }},
  {"component": "estimator",  "action": "reformir_weights_updated", "details": {
      "weights": {"original": 0.41, "rewrite_0": 0.70}, "n_reranked": 15
  }},
  {"component": "controller", "action": "feedback_stop", "details": {
      "reason": "weights_converged: delta=0.003 < threshold=0.01"
  }}
]
```

Source: [`components/feedback.py`](../../src/ragtune/components/feedback.py)

---

## Example: Partial Reranker Result

When a reranker returns scores for fewer docs than it was sent, the missing docs transition to `DROPPED`. The `dropped_ids` field (added in commit `66e11cf`) makes this visible in the trace.

```json
[
  {"component": "controller", "action": "rerank_batch", "details": {
      "count": 5,
      "strategy": "cross_encoder",
      "doc_ids": ["d1", "d2", "d3", "d4", "d5"],
      "dropped_ids": ["d3"]
  }}
]
```

Before the fix, `dropped_ids` was absent — you could not tell from the trace whether `d3` was dropped by the reranker or simply never selected.

---

## Example: Reformulation Error

```json
[
  {"component": "reformulator", "action": "llm_error", "details": {
      "error": "ConnectionRefusedError: [Errno 61] Connection refused",
      "model": "gpt-4o"
  }},
  {"component": "controller", "action": "pool_init", "details": {
      "count": 10,
      "reformulations": [],
      "metrics": {"total_unique_docs": 10, "overlap_count": 0}
  }}
]
```

The reformulator failure is logged but does not abort the run — the controller proceeds with an empty reformulation list and only the original query's retrieved docs.

---

## See Also

- [[controller-estimator-scheduler]] — how the controller loop uses the estimator and scheduler
- [`core/types.py`](../../src/ragtune/core/types.py) — `TraceEvent`, `ControllerTrace`, `ControllerOutput`
- [`core/budget.py`](../../src/ragtune/core/budget.py) — `CostTracker.try_consume()` emits all budget events
- [`components/feedback.py`](../../src/ragtune/components/feedback.py) — `ReformIRConvergenceFeedback` triggers `feedback_stop`
