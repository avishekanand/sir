This is the executable specification for **component-scoped budgets**: limiting and measuring individual pipeline stages (e.g. "only reranking, only latency") instead of only the whole query.

Origin: review comment by @VenkteshV on PR #31 (`src/ragtune/budget/loaders/vllm_budget.py`):

> is it possible to have flags so one can selectively measure cost/budget only for some components? Like, say i want to measure or budget only re-ranking and only from a latency perspective and not token costs. or maybe i just want to budget embedding time.

### **1. Current behavior (verified on `main` @ d13a540)**

1. `latency_ms` is a single global wall-clock gate. Nothing measures how long an individual stage takes.
2. Namespaced keys such as `rerank.latency_ms` are accepted by `CostBudget.limits` and accumulate in `try_consume()`, but `is_exhausted()` only checks `tokens`, `rerank_docs` and `latency_ms`, so they never stop anything.
3. `CostTracker.remaining_view()` reports `0` for any key that has no limit. A config that budgets only latency (no `rerank_docs`) therefore gets `remaining_rerank_docs == 0`, and both schedulers return `None`: **zero documents are reranked**. "Budget only reranking latency" is impossible today.

### **2. Key format**

A limit key is either a **global key** (unchanged: `tokens`, `rerank_docs`, `rerank_calls`, `retrieval_calls`, `reformulations`, `latency_ms`, or any custom key) or a **scoped key** `<component>.<dimension>`:

| Component | What is measured | Instrumented by |
|---|---|---|
| `retrieval` | every `retriever.retrieve()` call (incl. dense query encoding) | controller |
| `reformulation` | `reformulator.generate()` | controller |
| `estimation` | `estimator.value()` per loop iteration | controller |
| `embedding` | encoder calls inside components (opt-in) | `SimilarityEstimator`; any component via `tracker.measure("embedding")` |
| `rerank` | `reranker.rerank()` + score update per batch | controller |
| `assembly` | `assembler.assemble()` | controller |

| Dimension | Meaning | Source |
|---|---|---|
| `latency_ms` | cumulative wall time inside the component (nested scopes are inclusive) | `tracker.measure()` |
| `tokens` | `tokens` consumed while the component is active | attribution |
| `docs` | `rerank_docs` consumed while the component is active | attribution |
| `calls` | `rerank_calls` / `retrieval_calls` / `reformulations` consumed while active | attribution |

Scoped keys are validated when `CostBudget` is built: an unknown component or dimension raises `ValueError` naming the valid values (typos must not silently disable a budget).

### **3. Semantics**

- **Global keys behave exactly as before.** They are still consumed and still gate the loop.
- **Attribution.** While a component is active (`with tracker.measure(component)`), every global consumption is also recorded under `<component>.<dimension>`. The innermost active component receives the attribution. A consumption is allowed only if both the global and the scoped key allow it.
- **A scoped limit gates only its own component:**

| Exhausted key prefix | Effect |
|---|---|
| `rerank.*` | the iterative loop stops (no further batches) |
| `estimation.*` | the estimator is skipped; candidates keep their last priorities |
| `embedding.*` | `SimilarityEstimator` falls back to retrieval-score priorities |
| `retrieval.*` | no further supplemental (reformulation) retrievals |
| `reformulation.*` | the reformulator is not called |
| `assembly.tokens` | the assembler's `try_consume_tokens()` denies further documents |

- **Absent limit = unlimited.** `remaining_view()` reports `UNLIMITED` (`sys.maxsize`) for a dimension with no limit, instead of `0`. Configs that want "no reranking" must say `rerank_docs: 0` (as the `bm25_only` scenario already does).
- **Reading usage.** `final_budget_state` (i.e. `tracker.snapshot()`) contains every scoped key that was consumed, e.g. `final_budget_state["rerank.latency_ms"]`. The key you limit is the key you read.

### **4. API**

```python
# ragtune/core/types.py
UNLIMITED = sys.maxsize

class RemainingBudgetView(BaseModel):
    remaining_tokens: int
    remaining_rerank_docs: int
    remaining_rerank_calls: int
    scoped: Dict[str, int] = {}          # remaining amount per scoped limit
    def for_component(self, component: str) -> "RemainingBudgetView": ...
        # each field narrowed to min(global, "<component>.<dimension>")

# ragtune/core/budget.py
BUDGET_COMPONENTS = ("retrieval", "reformulation", "estimation", "embedding", "rerank", "assembly")
BUDGET_DIMENSIONS = ("latency_ms", "tokens", "docs", "calls")

class CostTracker:
    def measure(self, component: str) -> ContextManager[None]: ...
    def component_exhausted(self, component: str) -> bool: ...        # scoped keys only
    def is_exhausted(self, component: Optional[str] = None) -> bool: ...  # global (+ scoped)
```

Schedulers call `budget.for_component("rerank")` before sizing a batch.

### **5. CLI**

`ragtune run`:
- `--limit/-l KEY=VALUE` accepts scoped keys; `VALUE` of `none`/`off` removes a limit from the config. Malformed values are an error instead of a warning.
- `--only-limits`: ignore the config file's limits, enforce only `--limit` values.
- `--breakdown`: print per-component usage (latency, tokens, docs, calls) with limits.

`scripts/run_tool_retrieval.py` (config key / CLI flag):
- `limit_overrides` / `--limit KEY=VALUE` (repeatable): override every scenario's limits.
- `only_limits` / `--only-limits`: replace every scenario's limits.
- `report_keys` / `--report-keys k1,k2`: add `avg_<key>` columns (per-query mean of `final_budget_state[key]`).

Venky's examples:

```bash
# only reranking, only latency, no token costs
ragtune run cfg.yaml -q "..." --only-limits -l rerank.latency_ms=500 --breakdown
# only embedding time
ragtune run cfg.yaml -q "..." --only-limits -l embedding.latency_ms=200
python scripts/run_tool_retrieval.py --config configs/benchmark_skillret_bm25.yaml \
    --only-limits --limit rerank.latency_ms=500 --report-keys rerank.latency_ms,rerank.docs
```

### **6. Out of scope**

- Per-component **dollar** cost (the optional `cost_loader` stays a single rerank-batch estimator).
- Latency-aware batch sizing (schedulers only see doc/token/call remainders).
- The scheduler "token batch too large → return None" behavior from PR #31 (separate `fix/` PR).
- The offline estimator `ragtune budget`.

### **7. Verification checklist**

- [ ] Unknown component / dimension in a scoped key raises at `CostBudget` construction.
- [ ] `remaining_view()` reports `UNLIMITED` for absent keys; existing explicit limits unchanged.
- [ ] `measure()` records inclusive latency for nested scopes and attributes consumption to the innermost scope.
- [ ] Budget-exhaustion edge cases: `rerank.docs`, `rerank.latency_ms`, `assembly.tokens`, `reformulation.calls: 0`, `estimation.*`, `embedding.*` each stop exactly their own stage.
- [ ] `limits: {rerank.latency_ms: N}` alone reranks documents (gap 3 closed).
- [ ] Full unit suite: no new failures versus `main`.
