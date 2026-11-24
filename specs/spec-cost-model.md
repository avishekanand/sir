# Spec — FLOPs-and-dollar cost model

Status: draft
Sources:
- Peng et al., *Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers*
  (arXiv:2507.06223) — the FLOP estimator and the RPP/QPP metrics
- Kiran et al., *syftr: Pareto-Optimal Generative AI* (arXiv:2505.20266) and
  Barker et al., *Faster, Cheaper, Better* (arXiv:2502.18635) — prior RAG MOBO
  work, both of which price cost in money rather than FLOPs

## 1. Motivation

Objective 2 of the tuning study is currently `mean_rerank_docs`
(`tuning/optimizer.py:48`). It counts *documents reranked per query* and nothing
else. Three consequences, all measured on nfcorpus/bm25 with 50 queries:

| | NDCG@10 | mean_rerank_docs | measured ms/query |
|---|---|---|---|
| cross-encoder L-2 | 0.3069 | 39.2 | 197 |
| cross-encoder L-6 | 0.3339 | 39.2 | 234 |
| cross-encoder L-12 | 0.2996 | 39.2 | 308 |

1. **Reranker size is invisible.** All three score 39.2 docs, so the current
   cost axis rates them identically despite a 2.1x spread in compute. L-12 is
   both worse and dearer than L-6 — a genuinely dominated point the objective
   cannot express.
2. **LLM cost is entirely invisible.** One `claude-opus-5` reformulation costs
   $2.85/1k queries against $0.019 for L-6 reranking — **148x** — and
   `budget_reformulations` ranges to 5. Under the current objective every
   reformulation is free, so the tuner has no reason not to max it out.
3. **Nothing is comparable across hardware.** `rerank_docs` is a proxy that
   depends on which reranker is attached.

## 2. Why the unit must be money

FLOPs alone cannot span this search space. The E2R-FLOPs estimator needs layer
count, width, and FFN dimension; those are published for the cross-encoders but
**not for `claude-opus-5`**, so the API arm's FLOPs are not computable in
principle. For closed models the vendor's price is the only available proxy for
their compute. Hence: **money is the only unit that spans local and API
components**, which is also what syftr and *Faster, Cheaper, Better* use.

FLOPs do not disappear — they become the hardware-agnostic companion metric and
the way the local term is derived when no measurement exists for a device.

## 3. The cost vector

Every trial reports three quantities. Only the second is an optimizer objective.

| field | unit | spans | role |
|---|---|---|---|
| `flops_per_query` | FLOP | local models only | hardware-agnostic; `None` when an API model participated |
| `cost_usd_per_query` | USD | local + API | **objective 2 (minimize)** |
| `latency_ms` | ms | everything | reported; hardware-bound |

Derived for reporting only, never optimized: `RPP = ndcg / (flops/1e15)` and
`QPP = 1 / (flops/1e15)`, both from E2R-FLOPs §4.

### 3.1 FLOPs — encoder-only (cross-encoders)

E2R-FLOPs with `n_opt = 0` (a cross-encoder scores, it does not generate):

```
C_pair(n_ctx) = 2·N_enc·n_ctx + 4·n_layer·n_ctx²·d_attn
N_enc         = 2·d_model·n_layer·(2·d_attn + d_ff)
```

`N_enc` is the paper's own parameter approximation and is used rather than a
weight count so geometry can come from `AutoConfig` without downloading weights.
Checked against actual non-embedding parameter counts it is 1–2% low:

| model | formula `N_enc` | counted | GFLOP/pair @ n_ctx=355 |
|---|---|---|---|
| MiniLM-L-2 | 3.5M | 3.7M | 3.01 |
| MiniLM-L-6 | 10.6M | 10.8M | 8.83 |
| MiniLM-L-12 | 21.2M | 21.4M | 17.52 |

Per query: `flops = n_rerank_docs · C_pair(n_ctx)`.

### 3.2 Cost in USD

```
cost_usd = gpu_term + cpu_term + api_term                    (per query)

gpu_term = (P_gpu / 3.6e6) · n_rerank_docs · t_ce(model)      # t in ms
cpu_term = (P_cpu / 3.6e6) · t_retrieval
api_term = Σ_models [ tok_in·p_in + tok_out·p_out ] / 1e6
```

**`t_ce` depends on batch size, not just model.** `CrossEncoderReranker.rerank`
hands the scheduler's document list straight to `predict()`
(`components/rerankers.py:25-26`), so the effective GPU batch *is*
`scheduler_batch_size` — a knob the optimizer tunes over 1-20. Measured per-doc
cost over that range, fixed 200-doc workload, median of 5, CV <=3%:

| model | bs=1 | bs=2 | bs=5 | bs=10 | bs=20 | spread |
|---|---|---|---|---|---|---|
| L-2 | 3.00 | 2.99 | 1.43 | 1.03 | 0.87 | 3.4x |
| L-6 | 4.63 | 3.52 | 1.79 | 1.24 | 1.18 | **3.9x** |
| L-12 | 7.34 | 5.28 | 2.57 | 2.23 | 2.22 | 3.3x |

A scalar `ms_per_doc` would price `bs=1` and `bs=20` identically despite a 3.9x
real difference — reintroducing exactly the blind spot this spec removes. So
`ce_ms_per_doc` is a `{model: {batch_size: ms}}` grid, interpolated
piecewise-linearly and **clamped** outside it (the curve flattens by bs=10-20;
extrapolating the fall would under-price). A scalar entry is rejected, not
silently accepted. FLOPs are unaffected — batching changes wall-clock, not
arithmetic.

*Measurement note.* An earlier pass drew a fresh random document sample inside
each repeat, so the "repeats" timed different workloads; that produced CV ~12%
and an implausible non-monotonicity (bs=2 dearer than bs=1). One fixed sample
across all cells removes it. sentence-transformers 6.1.0 length-sorts and
interleaves before batching, so padding waste is already minimised and is not
the explanation.

**`t_ce` is a measured per-document time, not a FLOPs derivation.** Measured
times scale 1 : 1.73 : 3.21 across L-2/L-6/L-12 while FLOPs scale
1 : 2.93 : 5.81 — fixed per-pair overhead (tokenization, host-to-device copy,
kernel launch) and batch padding both blunt the spread, and achieved utilisation
is only 3.8–6.9% of the A6000's 38.7 TFLOPS fp32 peak. Deriving dollars from
FLOPs would systematically overcharge large rerankers. Where no measurement
exists for a device, fall back to `flops / (eta · FLOPS_peak)` and record that
the figure is extrapolated.

## 4. Calibrated constants

These are **assumptions that move conclusions** and are therefore versioned in
`src/ragtune/config/cost_model.yaml`, not hardcoded:

| constant | value used | provenance |
|---|---|---|
| `n_ctx` (nfcorpus) | 355 | measured: `n_q` mean 5.0 + `l_doc` mean 356.8, truncated at the tokenizer's 512 cap; 8.0% of pairs clipped |
| `t_ce` | `{model: {batch_size: ms}}` grid, 1-20 | measured, A6000, fixed 200-doc workload, median of 5, CV <=3% |
| `P_gpu` | $0.80 / GPU-hour | cloud A6000 on-demand; range $0.50–1.50 |
| `P_cpu` | $0.00 / hour | retrieval treated as free by default — see §7 |
| token prices | $/MTok in/out per model | provider published rates |

`n_ctx` **must be recalibrated per dataset** (scifact abstracts differ from
nfcorpus). It cancels in within-corpus comparisons — the RPP *ratios* are
identical at `n_ctx` 256 and 355 — but it scales absolute RPP/QPP by ~44%, so
any figure compared against published numbers depends on it.

## 4.1 Accounting: charge work performed, not work requested

Two discoveries during implementation changed how the local term is computed.

**`consumed["rerank_docs"]` is not a record of work done.**
`CostTracker.try_consume` accumulates into `consumed` *before* checking the
limit, so a denied request still increments the counter ("even denied attempts
count toward the running total", `core/budget.py`). A config that asks for more
reranking than its budget allows therefore inflates that counter without the
reranker ever running. The legacy objective read
`final_budget_state["rerank_docs"]` directly, so **it over-charged exactly the
trials that hit their budget** — biasing the existing Pareto front against
budget-constrained configurations.

The cost model therefore charges from `CostTracker.rerank_calls`, a per-call
record of invocations that actually happened. `rerank_docs` on `QueryUsage` is
derived from that record, not from `consumed`.

**The scheduler issues several rerank calls per query, at differing sizes.**
Each call's size sets its own per-document rate, so calls are recorded and
priced individually rather than summed into one mean batch. A call of `N`
documents is executed by `predict()` in batches of at most
`SBERT_BATCH_CAP = 32` (sentence-transformers' default), so its effective GPU
batch is `min(N, 32)`:

```
gpu_term = (P_gpu / 3.6e6) * sum( n * t_ce(model, min(n, 32)) for model, n in calls )
```

FLOPs sum plainly over calls, since batching changes wall-clock, not arithmetic.

## 5. Failure behaviour

A cost model that silently returns 0 corrupts the objective, so unknown inputs
are **errors, not defaults**:

- unknown reranker checkpoint (no `t_ce` and no geometry) → raise
- unknown LLM model (no token price) → raise
- `rerank_docs == 0` (budget exhausted before any reranking) → cost is the
  retrieval term only; this is a legitimate zero, not an error
- an API model participated → `flops_per_query` is `None`, never a partial sum
  silently presented as total compute

## 6. Deliverables

**PR 1a (this one)** — pure addition, no behaviour change:
- `src/ragtune/tuning/cost_model.py`
- `src/ragtune/config/cost_model.yaml`
- `scripts/calibrate_cost_model.py` — measures `n_ctx` over a dataset and
  `t_ce` per checkpoint, emits the YAML
- `tests/unit/tuning/test_cost_model.py`

**PR 1b** — wiring, behaviour change. Done:
- `CostTracker.record_llm_usage()` and `.record_rerank_call()`, exposed via
  `snapshot()`; neither is budget-enforced (priced, not limited)
- usage capture at all three LLM call sites — `LLMRewrite`, `ReformIR`
  (`components/reformulators.py`) and `LLMReranker` (`components/rerankers.py`)
  — via the shared `utils/llm_usage.py` helper; `CrossEncoderReranker` records
  each call's batch
- `QueryUsage.from_budget_state()` to build usage from a controller snapshot
- `cost_usd` and `flops` on `TrialObjectives`, computed in
  `tuning/evaluator.py` and `evaluate_controller_full()`
- objective 2 switched in `tuning/optimizer.py`, behind
  `TuningStudyConfig.use_cost_model` (default `False`, so existing studies keep
  their objective and stay comparable)
- `tests/unit/tuning/test_cost_accounting.py`: 13 tests, including the
  budget-exhaustion cases Rule 4 requires

Not in 1b: a `--use-cost-model` flag on `examples/beir_full_benchmark.py`, and
reporting `cost_usd` / RPP in `summary.md`. Those belong with the
matched-budget reporting work.

## 7. Validation

Measured on this branch, nfcorpus, A6000:

| config | $/1k q | PFLOP/q | RPP |
|---|---|---|---|
| L-6 bs=20 | 0.0103 | 3.41e-04 | 979 |
| L-6 bs=1 | **0.0403** | 3.41e-04 | 979 |
| L-2 bs=20 | 0.0076 | 1.14e-04 | **2700** |
| L-12 bs=20 | 0.0193 | 6.82e-04 | 439 |
| L-6 bs=20 + 1 haiku reformulation | 0.2793 | n/a (API) | - |
| L-6 bs=20 + 1 opus reformulation | 2.8653 | n/a (API) | - |
| L-6 bs=20 + 5 opus reformulations | 14.2853 | n/a (API) | - |

Four things the previous objective could not express, all now priced: the 3.9x
batch-size swing at identical FLOPs and identical NDCG; L-2 as the
efficiency-optimal reranker (2.7x the RPP of L-6 for 8% less NDCG); L-12 as
strictly dominated (worse NDCG *and* dearer); and LLM reformulation at 278x
reranking for one opus call, 1387x for five.

`tests/unit/tuning/test_cost_model.py`: 34 passed.

Note: 33 unit tests fail on this branch independently of this change
(`test_crumb_loader` 15, `test_llm_optimizer` 12, `test_flex_indexer` 5,
`test_config` 1), plus 3 collection errors from a missing `pyterrier_dr`.
Pre-existing; verified by the fact that this change modifies no tracked file.

## 8. Open questions

- `P_cpu = 0` makes retrieval free, so BM25 depth is costed only through its
  effect on reranked documents. Defensible when the GPU is the scarce resource;
  wrong for a CPU-bound deployment. Default 0, configurable.
- Padding waste is not modelled. E2R-FLOPs assumes none; with `l_doc` spanning
  4–2375 tokens, a naively batched run spends more than the estimate. Affects
  FLOPs only, not the measured-ms dollar term.
- Should `cost_usd` be per query or per 1k queries on the axis? Per query keeps
  it additive; per 1k is readable. Proposed: store per query, display per 1k.
