# Controller, Estimator, and Scheduler

## The Core Problem

Reranking is expensive — LLM calls, latency, token budget. You retrieve 50 documents but can only afford to rerank 10. RAGtune's architecture answers: **which 10, in what order, with what tool?**

Three components divide that responsibility cleanly:

| Component | Question it answers |
|---|---|
| **Estimator** | Which documents are *probably* worth reranking? |
| **Scheduler** | Given those estimates and the remaining budget, rerank *which batch next*? |
| **Controller** | Orchestrate the loop, enforce budget, manage state |

---

## The Loop

[`core/controller.py`](../../src/ragtune/core/controller.py) runs this cycle until the budget is exhausted (line 81):

```
while budget remains:
    1. Estimator.value(pool)     → set priority_value on each CANDIDATE doc
    2. Scheduler.select_batch()  → pick next batch using those priorities
    3. Reranker.rerank(batch)    → expensive score (cross-encoder or LLM)
    4. pool.update_scores()      → CANDIDATE → RERANKED (or DROPPED)
```

The controller owns the loop but does not know *how* to prioritize or *which* batch policy to use — it calls the abstractions in order and enforces the budget through [`CostTracker`](../../src/ragtune/core/budget.py).

Document state is managed by [`CandidatePool`](../../src/ragtune/core/pool.py):

```
CANDIDATE → IN_FLIGHT → RERANKED
                      ↘ DROPPED
```

---

## Estimator — cheap oracle

[`components/estimators.py`](../../src/ragtune/components/estimators.py)

**Purpose:** Predict document quality *without* running the expensive reranker. The result is a `priority_value` written onto each `PoolItem` in CANDIDATE state.

Four implementations ship:

### `BaselineEstimator` (line 8)
Uses raw retrieval score — the score returned by BM25 or the dense retriever. No learning, no state.

```python
score = max(item.sources.values()) if item.sources else 0.0
```

### `UtilityEstimator` (line 23)
Boosts candidates that share metadata fields (`source`, `section`, `category`) with already-reranked winners (docs scoring > 0.8). If a winner came from a particular section, other docs from that section are promoted.

### `SimilarityEstimator` (line 49)
Encodes candidates and winners with a sentence transformer (`all-MiniLM-L6-v2`), then boosts each candidate proportionally to its max cosine similarity to any winner. More principled than `UtilityEstimator` — works on content, not metadata.

```python
similarities = np.dot(norm_eligible, norm_winners.T)
max_sims = np.max(similarities, axis=1)
priorities[it.doc_id].priority *= (1.0 + max_sims[i] * boost_weight)
```

### `ReformIREstimator` (line 101) — most sophisticated
Once ≥3 documents have been reranked, it fits a constrained linear regression (via `scipy.optimize.lsq_linear`) that maps *retrieval source scores → cross-encoder score*. The model learns: "documents that scored well in reformulation 2 tend to be genuinely relevant." It then re-prioritizes all remaining candidates using the learned weights.

```python
# rows = reranked docs, cols = retrieval sources (original, rewrite_0, rewrite_1, ...)
X = np.array([[it.sources.get(s, 0.0) for s in all_sources] for it in reranked])
y = np.array([it.reranker_score for it in reranked])
result = scipy.optimize.lsq_linear(X, y, bounds=(0, 1))
self._learned_weights = dict(zip(all_sources, result.x))
```

Gets smarter on every iteration. The weights are written into `EstimatorOutput.metadata` so `ReformIRConvergenceFeedback` can monitor them for convergence.

**Why necessary:** Without an estimator, the scheduler has no signal beyond retrieval rank. The estimator is the "which docs are probably worth the cost?" judgment. Swap estimators without touching anything else.

---

## Scheduler — batch selection policy

[`components/schedulers.py`](../../src/ragtune/components/schedulers.py)

**Purpose:** Given the priorities the estimator just set, decide *which batch* to send and *with what strategy* (cross-encoder vs LLM).

Both current implementations are greedy top-N — sort by `priority_value`, take the first `batch_size`. The interesting part is **strategy routing**.

### `ActiveLearningScheduler` (line 7)

Greedy top-N with one escalation rule: if the top two candidates have a priority gap < 5%, it upgrades from `cross_encoder` to `llm` to break the tie with higher-fidelity signal.

```python
gap = selected[0].priority_value - selected[1].priority_value
if gap < 0.05 and current_strategy == "cross_encoder":
    current_strategy = "llm"
```

The name "active learning" is aspirational. A proper active learning scheduler would trade off *exploitation* (rerank the highest priority docs) against *exploration* (rerank uncertain docs to reduce estimator error). The current implementation only exploits.

### `GracefulDegradationScheduler` (line 47)

Spends LLM calls first (up to `llm_limit`), then falls back to cross-encoder (up to `cross_encoder_limit`), then stops. Budget-adaptive: uses the expensive tool where quality matters most (the top of the ranking) and the cheap tool for the rest.

```python
if num_llm < self.llm_limit:
    strategy = "llm"
elif num_ce < self.cross_encoder_limit:
    strategy = "cross_encoder"
else:
    return None
```

**Why separate from the estimator?** The estimator *scores*; the scheduler *selects*. These are independent policies. `ReformIREstimator` pairs with either scheduler without any code changes. The scheduler is also the right place to add exploration logic (uncertainty sampling, diversity-based selection) if that is ever implemented.

---

## Why all three are necessary

Collapsing any two creates a problem:

- **No estimator** → scheduler sorts by raw retrieval rank, no learning from prior batches
- **No scheduler** → controller embeds selection policy; swapping strategies requires editing the loop
- **No controller** → no budget enforcement, no state machine, no unified trace

The controller writes to [`ControllerTrace`](../../src/ragtune/core/types.py) at each step (pool init, rerank batch, errors, feedback stops), giving full observability into why each batch was picked and what the estimator predicted.

---

## See Also

- [[design/design_v0_57]] — latest design spec
- [`core/interfaces.py`](../../src/ragtune/core/interfaces.py) — `BaseEstimator`, `BaseScheduler` abstract interfaces
- [`core/types.py`](../../src/ragtune/core/types.py) — `EstimatorOutput`, `BatchProposal`, `RAGtuneContext`
- [`components/feedback.py`](../../src/ragtune/components/feedback.py) — `ReformIRConvergenceFeedback` (reads estimator weights to decide early stopping)
