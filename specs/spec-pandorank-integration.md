# Spec: PandoRank as native RAGtune components

Executable specification for folding **PandoRank** (reservation-value
re-ranking with endogenous stopping; paper: *Cost-Aware, Feedback-Driven
Re-Ranking with Endogenous Stopping*, code: `avishekanand/pandorank-code`)
into the RAGtune active loop. Per repo convention this spec merges before any
implementation PR.

The governing constraint: **no phase may change a published number.** The
pandorank-code repo carries a regression suite that pins the paper's tables to
cached artefacts; those goldens are the acceptance tests of this migration and
are never re-pinned.

## 1. Why the fit is natural

RAGtune's controller loop already factors re-ranking the way PandoRank's
theory does:

| RAGtune contract | PandoRank concept |
|---|---|
| `BaseEstimator.value() -> {doc_id: EstimatorOutput(priority=...)}` | the **reservation value** σ_d — the price-adjusted worth of *inspecting* d |
| `BaseScheduler.select_batch(pool, budget)` | open boxes in decreasing σ under the budget |
| `BaseFeedback.should_stop(state, budget, estimates)` | the **endogenous stop**: best score in hand ≥ max remaining σ |
| `BaseReranker.rerank(...) -> {doc_id: score}` | the priced observation φ(q,d) |
| `CostBudget` / `CostTracker` | the budget the stopping rule prices against (κ per observation) |

Nothing in the loop changes; PandoRank arrives as one estimator, one
scheduler, one feedback policy, and reranker adapters.

## 2. Directory structure (target)

```text
src/ragtune/
├── components/
│   ├── pandorank/
│   │   ├── __init__.py
│   │   ├── solver.py         # reservation.py, verbatim: κ = E[(v-σ)+] by
│   │   │                     #   bracketed bisection, Gaussian + empirical.
│   │   │                     #   Pure numpy/scipy. No RAGtune imports.
│   │   ├── belief.py         # BayesianLinear / CalibratedBayesianLinear:
│   │   │                     #   per-query online (μ, s) from cheap features
│   │   │                     #   + the observations already paid for
│   │   ├── estimator.py      # @registry.estimator("pandorank")
│   │   ├── scheduler.py      # @registry.scheduler("weitzman")
│   │   └── feedback.py       # @registry.feedback("endogenous-stop")
│   └── rerankers.py          # + @registry.reranker("llm-pointwise") adapter
│                             #   (HFPointwiseRanker: Yes/No log-odds, one
│                             #    forward pass) and monoT5/RankLLaMA adapters
├── storage/
│   └── score_cache.py        # φ cache WITH the provenance sidecar: refuses
│                             #   to seed/merge across differing
│                             #   (model, max_doc_tokens). Ported behaviour,
│                             #   sqlite + json backends.
└── config/models.py          # + PandorankConfig{kappa, kappa_as_iqr_fraction,
                              #   max_scored, features, prior{m0,tau},
                              #   calibrate}
```

## 3. Component contracts

### 3.1 `PandorankEstimator` (`estimator.py`)

`value(pool, context)`:
1. Build features per eligible item from `item.sources` (first-stage scores —
   BM25, dense — are RAGtune's `sources` dict; rank position from
   `initial_rank`).
2. Fit the belief on items with `state == RERANKED`, using their
   `reranker_score` as observations. Fewer than `min_obs` observations → prior.
3. Solve σ for all eligible items **in one vectorised call**
   (`solver.gaussian_reservation`; the caller-side batching is a 1311×
   measured speed-up and its absence was a real 6-hour incident).
4. Return `EstimatorOutput(priority=σ_d, predicted_quality=μ_d)`.

`needs_reformulation`: default False (out of scope; the paper's rule does not
reformulate).

### 3.2 `WeitzmanScheduler` (`scheduler.py`)

`select_batch`: top-`batch_size` eligible by `priority_value` descending —
identical mechanics to `ActiveLearningScheduler` minus strategy escalation —
bounded by `budget.remaining_rerank_docs`. Weitzman's rule is exactly
"open in decreasing σ", so the scheduler stays thin; the intelligence is in
the estimator's priority.

### 3.3 `EndogenousStop` (`feedback.py`)

`should_stop(state, budget, estimates)`:
- `best_in_hand` = max `reranker_score` over opened items (from `state`).
- `max_remaining_sigma` = max of `estimates` over still-eligible items.
- Stop iff `best_in_hand >= max_remaining_sigma`, reason string carries both
  numbers.
- Degenerate cases match the solver's conventions: κ ≤ 0 → σ = +∞ → never
  stop early; no eligible items → stop with reason "exhausted".

### 3.4 κ pricing (`config`)

κ is meaningful only on a re-ranker's own reward scale. Config accepts either
`kappa` (absolute) or `kappa_as_iqr_fraction` + a calibration sample; the
IQR-based derivation is ported from `derive_kappa.py`. A configured κ recorded
against a *different* reranker id is a validation error, not a warning —
transplanted κ produced meaningless sweeps in the source project.

## 4. Migration phases — one PR each, each independently revertable

Branch/commit style per CLAUDE.md (`feat(pandorank): ...`); every phase keeps
BOTH repos' suites green. pandorank-code remains the source of truth until
phase P5 completes; nothing is deleted before then.

| phase | lands | acceptance (must pass before merge) | rollback |
|---|---|---|---|
| **P0** | in pandorank-code: the frozen **parity oracle** — a script that runs classic PandoRank on fixed scenarios (cached φ, fixed pools, fixed seed) and dumps the opened sequence, per-step σ, stop point and final ranking to a golden JSON | oracle is deterministic across two runs (bit-identical dump) | delete the script; nothing depends on it |
| **P1** | this spec, merged | review | revert |
| **P2** | `solver.py` + `belief.py` in ragtune, no wiring | ported solver goldens (independent `scipy.brentq` values), defining-equation + monotonicity properties, belief unit tests; **P0 oracle unaffected** (no code shared yet) | revert PR |
| **P3** | estimator + scheduler + feedback, registered; controller runs them behind config | **parity test**: on the P0 scenarios, the RAGtune loop opens the *same documents in the same order* and stops at the *same point* as the classic dump, and final scores match to 1e-9. Unit tests for each contract edge (κ≤0, empty pool, all-opened) | revert PR; classic path untouched |
| **P4** | reranker adapters + `score_cache.py` with provenance refusal | adapter equivalence: `HFPointwiseRanker` through the adapter returns bit-identical scores to direct calls on a fixed fixture; cache refuses a mismatched-provenance seed (test) | revert PR |
| **P5** | experiment bridge: pandorank-code's depth-1000 replay re-run **through the RAGtune stack** | the published-number regression goldens reproduce through RAGtune: per-collection fixed-100 baselines (0.2117 / 0.2182 / 0.2256 / 0.1375), exhaustive-1000, fixed-k peak value and location; plus one 10-query live GPU smoke, new vs old, scores bit-identical | revert PR; classic drivers still primary |
| **P6** | pandorank-code slims to experiments + archival; imports ragtune; deprecation shims for old entry points | full pandorank-code suite (571 unit + regression tier) green against the shims | shims make this reversible by construction |

Slow by design: phases merge at most one per review cycle, and P3 and P5 each
require the numeric acceptance run attached to the PR description.

## 5. The no-regression experimental plan

Four tiers, run at the cadence shown; goldens are published values and are
never re-pinned — a mismatch is a finding:

- **Tier A — every commit, seconds, no data.** Solver goldens vs independent
  brentq; defining-equation residual ≤ 1e-6; σ monotone in κ; batch ≡ scalar
  solves. (Exists: `pandorank-code/tests/regression/test_solver_golden.py`;
  ports to ragtune in P2.)
- **Tier B — every PR, minutes, frozen fixtures.** The P0 parity oracle:
  identical opened set, order, stop point, final ranking on ~6 scenarios
  spanning (κ low/high, features on/off, pool 50/1000, ties present). This is
  the tier that catches behavioural drift the unit tests cannot.
- **Tier C — per phase, ~15 min, data-gated.** The depth-1000 replay from the
  φ caches through whichever stack the phase touched. Values pinned:
  baselines, exhaustive-1000, fixed-k peaks (value *and* argmax k).
- **Tier D — before P6 only, GPU.** One live smoke (10 queries,
  crumb/stack_exchange): new stack vs old, φ scores bit-identical given the
  same model and truncation; then the 830-query agentic goldens re-checked
  from artefacts (recall from trajectories 59.47, accuracy 56.7/53.3,
  savings 30.2%/12.9%).

## 6. Known impedance mismatches (decided here, not discovered mid-PR)

1. **Priority semantics.** RAGtune sorts by `priority_value` with tie-break
   `(initial_rank, doc_id)`; classic PandoRank tie-breaks σ by first-stage
   order. Same rule — but the parity oracle asserts it, because ties are
   common (quantised log-odds) and an unstable tie-break silently changed a
   pool in the source project.
2. **Observation batching.** RAGtune reranks in batches (`batch_size`);
   classic PandoRank's theory covers batched openings (fair-cap calculus).
   Parity scenarios include batch_size ∈ {1, 5} and the classic batch size.
3. **`needs_reformulation`.** PandoRank's estimator returns False; the
   controller's reformulation stage must be a no-op under the PandoRank
   config. Asserted in P3 tests.
4. **Budget double-accounting.** RAGtune's `CostTracker` counts calls;
   PandoRank's κ prices *observations*. κ stays out of `CostBudget` — it is a
   price inside the estimator, not a cap. `max_scored` maps to
   `remaining_rerank_docs`.
5. **Python version.** ragtune pins `>=3.11,<3.12`; pandorank-code runs 3.11
   in production venvs already. No conflict, but the solver's tests must run
   under ragtune's lockfile (`uv.lock`), not pandorank-code's requirements.
