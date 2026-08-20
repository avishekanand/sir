# Automatic Configuration of RAGtune Pipelines: Bayesian Optimization vs. LLM Reasoning Agents

**Author:** Rahul Seetharaman  
**Date:** August 2026  
**Status:** In-progress — pilot results reported; full multi-benchmark sweep pending

---

## Abstract

RAGtune's controller exposes roughly twenty tunable parameters spanning component selection (reranker, scheduler, estimator, reformulator) and numerical knobs (retrieval depth, pool size, rerank budget). Choosing good values by hand is error-prone and dataset-dependent. This report frames the configuration problem as a multi-objective optimization task — maximize retrieval quality (NDCG@10) while minimizing compute cost (rerank documents consumed) — and compares two fundamentally different optimizers: a Bayesian multi-objective surrogate (Optuna TPE) and an LLM reasoning agent that observes execution traces and proposes configurations through language. A pilot study on TREC-COVID (20 trials, 10 queries each) found both methods converging to the same peak quality (NDCG@10 = 0.423), but Bayesian TPE discovering a three-point Pareto front while the LLM agent found only one. The quality ceiling itself is attributed to three fixable issues — wrong BM25 parameters, insufficient retrieval depth, and evaluation noise — all resolved for the full benchmark sweep across TREC-COVID, NFCorpus, SciFact, and FIQA.

---

## 1. Introduction

RAGtune is not a single pipeline; it is a family of pipelines. The `RAGtuneController` assembles a reranker, a reformulator, a scheduler, an estimator, and a feedback mechanism into an iterative loop, with a `CostBudget` enforcing hard limits on rerank volume, reformulation calls, and wall time. Every component is swappable, and most carry numerical hyperparameters of their own. The result is a search space that is simultaneously discrete (which reranker?), conditional (cross-encoder model only matters if cross-encoder is selected), and high-dimensional (depth, batch size, pool cap, duplicate threshold, and so on).

In practice, this space is configured by hand or by copying from a prior successful run on a similar dataset. Neither approach is principled. A configuration tuned for a dense biomedical corpus (TREC-COVID) may perform poorly on a short-document collection (NFCorpus) or a QA dataset (FIQA) where query–document length ratios differ markedly. More concretely: the optimal rerank depth, scheduler batch size, and whether to rewrite queries at all vary significantly across domains.

The question this work addresses is: **given a fixed retriever and a new evaluation dataset, which automated method most efficiently discovers the Pareto-optimal region of the quality–cost space?**

Two paradigms are evaluated. The first is Bayesian multi-objective optimization using Optuna's Tree-structured Parzen Estimator (TPE), following the architecture of the syftr system (arXiv 2505.20266). The second is an LLM reasoning agent that observes a `TraceAggregate` of the previous trial, reflects on what went wrong or well, and proposes the next configuration as structured JSON. The comparison is not purely academic: each paradigm has a different sample-efficiency profile, a different failure mode at small budgets, and different transparency into its decision process.

---

## 2. Background: The RAGtune Pipeline

This section is brief because the reader is assumed to know RAGtune. It exists only to establish notation used later.

A `RAGtuneController.run(query)` executes the following steps:

1. **Retrieval.** The `BaseRetriever` fetches `original_query_depth` candidates from the index, populating the `CandidatePool` with documents in `CANDIDATE` state.
2. **Reformulation.** An optional `Reformulator` rewrites or expands the query, adding `depth_per_reformulation` additional candidates per variant.
3. **Iterative loop.** While the `CostBudget` permits:
   - The `Estimator` scores candidates (utility, similarity, or a learned regressor).
   - The `Scheduler` selects a batch of documents to rerank.
   - The `Reranker` scores the batch; documents advance to `RERANKED` state.
   - The `CandidatePool` is updated; a `FeedbackSignal` may trigger early exit.
4. **Assembly.** The `GreedyAssembler` selects the top-scoring reranked documents within a token budget.

`ControllerOutput.final_budget_state` reports what was consumed. The key cost signal for this study is `consumed["rerank_docs"]`: the total number of documents passed to the reranker, which proxies API inference cost or latency when using neural rerankers.

---

## 3. Research Problem

### 3.1 The Search Space

`RAGtuneSearchSpace` (defined in `src/ragtune/tuning/search_space.py`) describes the joint distribution over all tunable parameters. The space has the following structure for this study:

| Dimension | Type | Values / Range |
|---|---|---|
| `reranker_type` | Categorical | noop, cross-encoder, monoT5 |
| `reformulator_type` | Categorical | identity, llm_rewrite |
| `estimator_type` | Categorical | baseline, utility, similarity |
| `scheduler_type` | Categorical | active-learning, graceful-degradation |
| `feedback_type` | Categorical | none, budget-stop |
| `original_query_depth` | Integer (log) | 10 – 200 |
| `max_pool_size` | Integer (log) | 10 – 200 |
| `budget_rerank_docs` | Integer (log) | 5 – 200 |
| `scheduler_batch_size` | Integer (log) | 1 – 20 |
| `assembler_max_docs` | Integer | 3 – 20 |
| `near_duplicate_threshold` | Float | 0.5 – 0.95 |
| *(+ 10 conditional sub-params)* | | — |

The discrete dimensions alone yield 3 × 2 × 3 × 2 × 2 = 72 combinations. Multiplied by the continuous dimensions, the effective cardinality is in the hundreds of thousands. Grid search is intractable.

### 3.2 Objectives

The optimization is **multi-objective**. A configuration that achieves high NDCG@10 by reranking 200 documents is not equivalent to one that achieves the same NDCG@10 by reranking 20 documents — the latter is strictly preferable for cost-sensitive deployment. A single scalar objective (e.g. NDCG@10 − λ × cost) requires choosing λ, which amounts to a business decision. Instead, the goal is to find the **Pareto front**: the set of configurations where no other configuration is simultaneously better on both objectives.

Formally, a configuration **c** dominates **c'** if and only if:

```
NDCG@10(c) ≥ NDCG@10(c')  AND  rerank_docs(c) ≤ rerank_docs(c')
```

with at least one strict inequality. The Pareto front is the set of all non-dominated configurations. A practitioner with a cost budget of, say, 30 reranked documents per query picks the Pareto-front configuration nearest to that constraint; one with no cost constraint picks the highest-NDCG point.

### 3.3 Why Standard Tuning Fails

A naïve approach — grid search over reranker type at fixed depth — misses the interactions between components. Whether cross-encoder reranking adds value over noop BM25 depends on retrieval depth: if `original_query_depth = 20`, the pool contains only top-20 BM25 results, and a cross-encoder can only reorder documents already ranked high by BM25 — which adds little. At `original_query_depth = 150`, the pool may contain relevant documents that BM25 ranked 80th; cross-encoder reranking can rescue them. Joint optimization over depth and reranker type is necessary.

Similarly, the estimator interacts with the scheduler: the `active-learning` scheduler prioritizes documents by estimator score, so a weak estimator (baseline) reduces its benefit. The `similarity` estimator adds a model inference step per candidate, increasing latency but improving batch selection quality. These three-way interactions are precisely what multivariate Bayesian optimization handles well and what a sequential manual sweep handles poorly.

---

## 4. Optimization Methods

### 4.1 Bayesian Multi-objective TPE

The Bayesian optimizer uses Optuna's `TPESampler(multivariate=True, constant_liar=True)` with `directions=["maximize", "minimize"]`. Key properties:

- **Multivariate**: The sampler models correlations between parameters, so it can learn that `cross-encoder` + deep `original_query_depth` is a promising region, not just that each individual value is good in isolation.
- **Constant liar**: For sequential execution, marks in-flight trials as having the current best value, preventing the sampler from sending many trials to the same region before results come back.
- **Multi-objective**: `study.best_trials` returns the non-dominated subset of completed trials — the empirical Pareto front — using Optuna's built-in dominance logic.

Three pruners accelerate the search by aborting unpromising trials early:

- **CostPruner**: Kills any trial whose running mean `rerank_docs` exceeds `max_mean_rerank_docs` after a warmup period.
- **RuntimePruner**: Kills any trial whose projected total evaluation time (based on mean per-query latency × remaining queries) exceeds `max_trial_seconds`.
- **ParetoPruner** (post-warmup): Kills any trial whose optimistic upper bound on NDCG and lower bound on cost is still dominated by a Pareto-front point from completed trials.

The optimizer requires no interpretability and no domain knowledge. Its sample complexity scales with the number of trials, not the number of parameters.

### 4.2 LLM Reasoning Agent

The LLM agent replaces the Bayesian surrogate with a language model. At each iteration, it receives a `TraceAggregate` built from the previous trial's execution:

| Field | Meaning |
|---|---|
| `avg_pool_size` | Mean candidate pool size across queries |
| `avg_pct_pool_reranked` | Fraction of pool documents that were reranked |
| `feedback_stop_rate` | How often the feedback mechanism triggered early exit |
| `retrieval_skip_rate` | How often retrieval was skipped (pool already full) |
| `avg_rewrite_utility` | NDCG lift attributable to reformulated queries (0 if reformulator=identity) |
| `avg_rerank_errors` | Reranker exception rate |

The agent then receives a prompt that includes:
1. A description of each search space dimension and its legal values.
2. The full `TraceAggregate` for the last trial.
3. The NDCG@10 and mean `rerank_docs` achieved.
4. The history of all prior configurations tried (as a compact table).

It responds with the next configuration as a JSON object. The controller instantiates and runs the configuration, computes objectives, builds a new `TraceAggregate`, and feeds it back. This is the GEPA/TextGrad-style trace-reflection loop applied to RAGtune.

The agent's advantage is interpretability: its reasoning traces are human-readable and may reveal domain-specific insights (e.g. "rewrite utility is 0% suggesting reformulation is not helping on this dataset"). Its disadvantage is that it relies on the language model's prior knowledge about information retrieval components, which may or may not match the specific dataset distribution.

---

## 5. Experimental Setup

### 5.1 Datasets

Four BEIR benchmarks are evaluated, representing diverse retrieval settings:

| Dataset | Docs | Queries | Domain | BEIR NDCG@10 (BM25 reference) |
|---|---|---|---|---|
| TREC-COVID | 171,332 | 50 | Biomedical (COVID-19) | 0.656 |
| NFCorpus | 3,633 | 323 | Medical nutrition | 0.325 |
| SciFact | 5,183 | 300 | Scientific claim verification | 0.665 |
| FIQA | 57,638 | 648 | Financial QA | 0.236 |

All are loaded via PyTerrier's IRDS integration (`irds:beir/<name>/test`). BM25 indexes are built using PyTerrier's `IterDictIndexer`.

### 5.2 Fixed Retriever

Both optimizers share a fixed `PyTerrierRetriever` backed by a BM25 index with BEIR-standard parameters:

```
k1 = 0.9    (term frequency saturation)
b  = 0.4    (document length normalisation)
```

These match Anserini defaults used in the original BEIR benchmark paper. The Terrier-default of b=0.75 produces a measurable NDCG deficit on dense biomedical collections.

A critical implementation detail: PyTerrier 1.1+ routes queries through a MatchOps pipeline (`TerrierQLToMatchOpQL → SingleTermOp`) that unconditionally calls `checkForFields()` on the index. Indexes without field statistics raise a `JavaException` at query time. The fix is to pass `controls={"matchopql": "off"}` to `pt.terrier.Retriever`, which bypasses the MatchOps stage entirely. BM25 parameters are set via `properties={"c": "0.4", "k1": "0.9"}` (ApplicationSetup globals), not controls, because BM25 control parameters route to the field-aware BM25F model which also requires field statistics.

### 5.3 Search Space

Both optimizers search the same `RAGtuneSearchSpace` with the same bounds. The retriever is held fixed; only the reranking, scheduling, estimation, and reformulation pipeline is tuned. This is a deliberate choice: it isolates the optimizer comparison from the confound of index quality, and it reflects the practical deployment scenario where the retriever is already provisioned.

### 5.4 Evaluation Protocol

Each trial/iteration evaluates the proposed configuration against a random sample of evaluation queries, seeded for reproducibility (`seed=42`). Per-query metrics:

- **NDCG@10**: Computed from `ControllerOutput.documents` (ranked list) against the dataset's qrels using graded gain `2^rel − 1`.
- **Rerank docs**: `ControllerOutput.final_budget_state["rerank_docs"]` — the total documents passed to the reranker for this query.

Trial objectives are the mean of both quantities across all evaluation queries.

**Full benchmark settings** (after pilot corrections):

| Setting | Value |
|---|---|
| Trials / iterations per optimizer | 50 |
| Evaluation queries per trial | 50 |
| Pruners | CostPruner (max 200 docs), RuntimePruner (max 180 s) |

---

## 6. Pilot Study: TREC-COVID (20 Iterations × 10 Queries)

Before running the full benchmark, a smaller pilot was conducted on TREC-COVID to validate the infrastructure and characterize optimizer behavior at a small budget.

### 6.1 Results

| Metric | LLM Agent | Bayesian TPE |
|---|---|---|
| Best NDCG@10 | 0.4232 | 0.4232 (tied) |
| Min cost (mean rerank docs) | 10.0 | **3.0** |
| Pareto front size | 1 | **3** |
| Hypervolume | **4.23** | 4.17 |
| Wall time | 878 s | **743 s** |

Both methods produced the same peak quality. Bayesian TPE found a richer Pareto front (three non-dominated configurations vs. one) at lower minimum cost, and ran 15% faster. The LLM agent's single-point hypervolume advantage reflects that its one Pareto-optimal point happens to sit at a slightly favourable NDCG-cost tradeoff, but the absence of lower-cost alternatives limits practical utility.

### 6.2 LLM Agent Behaviour

The agent's reasoning traces reveal a characteristic failure mode at small evaluation budgets:

**Rapid convergence and local lock-in.** By iteration 4, the agent correctly identified that `noop` (pure BM25) outperforms cross-encoder reranking on this data. It then alternated between `noop` and `cross-encoder` for the remaining 16 iterations — effectively stuck in a two-point oscillation rather than exploring the scheduler and estimator dimensions.

**Zero rewrite utility throughout.** All 20 iterations returned `avg_rewrite_utility = 0%`. The reformulator was consistently configured as `identity` from early on, and the agent never meaningfully explored `llm_rewrite`. This is arguably correct behaviour (query rewriting does not help BM25 on biomedical queries), but it was reached by default rather than by deliberate exploration.

**A single late breakthrough.** At iteration 19, the agent reduced `original_query_depth` to 10 and `max_pool_size` to 10, cutting cost from 20.0 to 10.0 rerank docs while marginally raising NDCG to 0.423 (BM25 top-10 was slightly better than top-20 on this 10-query sample). This was the only trial to change retrieval depth despite it being one of the highest-leverage parameters.

The core problem is that 10 queries produces NDCG variance of roughly ±0.02. The agent cannot distinguish a genuinely better configuration from a noisy one, so its reflective analysis ("the cross-encoder is hurting; reverting to noop") may be acting on noise rather than signal.

---

## 7. Root Cause Analysis: The 0.42 NDCG Ceiling

The BEIR benchmark reports BM25 NDCG@10 = 0.656 on TREC-COVID (Anserini). Both optimizers plateaued at 0.423 — a 0.23 gap. Three compounding issues explain this:

### 7.1 Wrong BM25 Parameters

Terrier's default BM25 uses `b = 0.75` (aggressive length normalisation). The BEIR paper uses Anserini defaults: `b = 0.4`, `k1 = 0.9`. On TREC-COVID, which has unusually long documents (full COVID-19 abstracts and some full-text passages), aggressive length normalisation disproportionately penalises longer relevant documents. The parameter mismatch alone accounts for an estimated ~0.10 NDCG gap.

**Fix applied:** `properties={"c": "0.4", "k1": "0.9"}` in the PyTerrier retriever constructor, matching BEIR Anserini defaults.

### 7.2 Retrieval Depth Capped at 50

The pilot search space had `original_query_depth_range = (5, 50)`. At depth 50, the candidate pool contains at most 50 BM25-ranked documents. For a cross-encoder to improve over BM25, it needs to rescue relevant documents that BM25 ranked poorly — but those documents are not in the pool. Any configuration that selects `cross-encoder` reranking is therefore operating on a pool where BM25 has already done most of the work, and the reranker has no recovery opportunity.

**Fix applied:** `original_query_depth_range = (10, 200)`, `budget_rerank_docs_range = (5, 200)` in `RAGtuneSearchSpace`.

### 7.3 Evaluation Noise from 10 Queries

NDCG@10 variance on a 10-query sample is approximately ±0.02 (computed from bootstrap resampling of BEIR query distributions). Both optimizers are effectively fitting noise. The LLM agent interprets noisy feedback as signal; Bayesian TPE's surrogate incorporates the noise implicitly into its uncertainty estimate, but 10 queries is below the practical noise floor for reliable Pareto-front construction.

**Fix applied:** `n_eval_queries = 50` per trial, reducing noise by a factor of √5 ≈ 2.2.

### 7.4 A Note on the Remaining Gap

Even with fixes applied, Terrier's tokeniser differs from Anserini's (different stoplist, different stemmer defaults). A tokeniser mismatch of this magnitude typically accounts for 0.03–0.05 NDCG on BEIR. This is not fixed — it is accepted as the performance ceiling for Terrier-backed BM25 on these benchmarks.

---

## 8. Hypotheses for the Full Benchmark Run

With fixes in place (BEIR BM25 params, widened retrieval depth, 50 eval queries), the following outcomes are expected:

**H1: BM25 noop baseline recovers to 0.55–0.65 NDCG@10 on TREC-COVID.** The remaining gap to 0.656 is attributable to tokeniser differences; the BM25 parameter fix closes most of the gap.

**H2: Cross-encoder reranking adds lift on TREC-COVID and SciFact at depth ≥ 100.** With a pool of 100–200 candidates, a cross-encoder can rescue relevant documents misranked by BM25. On NFCorpus (3,633 total docs) and FIQA (short QA queries), the margin will be smaller.

**H3: Bayesian TPE maintains its Pareto-front advantage at 50 trials.** With systematic random-phase coverage (50 startup trials) before the surrogate takes over, TPE should find at least 3–5 non-dominated configurations per dataset. The LLM agent may still find the highest-quality single point but is unlikely to explore the cost dimension as thoroughly.

**H4: The LLM agent reasons more coherently with 50-query feedback.** Lower-noise NDCG estimates give the agent's reflections a better signal-to-act on. However, on small datasets like NFCorpus (323 queries total), the agent may still plateau early because the high-scoring configurations are genuinely few.

**H5: Dataset characteristics dominate optimizer choice.** On TREC-COVID and SciFact (documents with strong lexical overlap to queries), BM25 + noop will form the cost-efficient Pareto segment; cross-encoder adds value only at the high-depth end. On FIQA (open-domain QA with vocabulary mismatch), query rewriting may appear on the Pareto front for the first time.

---

## 9. Open Questions

**Can the LLM agent be steered away from local lock-in?** The pilot agent converged in 4 iterations because it received no explicit instruction to explore. Modifying the prompt to include an exploration mandate ("you must try at least one configuration with a reranker you have not tried before") could break the oscillation pattern.

**Is the TraceAggregate sufficient for the agent's reasoning?** The aggregate reports averages across queries, discarding query-level variance. A configuration that works well on 40/50 queries but catastrophically fails 10 may look similar to one with uniformly mediocre results. Per-query trace summaries (min, max, p10, p90) could give the agent better signal for detecting outlier-driven failures.

**How does optimizer performance scale with trial budget?** The pilot used 20 iterations. The full run uses 50. At 200 trials (a typical Bayesian sweep), Pareto-front coverage should be substantially better. Whether the LLM agent closes the gap at higher budgets — or whether its local lock-in compounds — is the central empirical question.

**Are multi-dataset transfer effects present?** If a configuration found to be Pareto-optimal on TREC-COVID generalises to SciFact without re-tuning, that would be evidence for dataset-invariant "good" configurations (likely: always-on graceful degradation with moderate depth). If configurations are highly dataset-specific, per-dataset tuning is unavoidable.

---

## 10. Implementation Notes

All code lives in the `feat/llm-optimizer` branch. Key files:

| File | Role |
|---|---|
| `src/ragtune/tuning/search_space.py` | `RAGtuneSearchSpace`: parameter distributions, `build_controller()` |
| `src/ragtune/tuning/evaluator.py` | `TrialEvaluator`: per-query loop, NDCG computation, pruner integration |
| `src/ragtune/tuning/optimizer.py` | Bayesian study setup, trial loop, Pareto extraction |
| `src/ragtune/adapters/pyterrier.py` | `PyTerrierRetriever` with `matchopql=off` and BEIR BM25 params |
| `examples/ragtune_benchmark.ipynb` | End-to-end Colab notebook: index build → BM25 sanity → LLM + Bayesian sweep → plots |
| `examples/colab_multi_benchmark.py` | Script version of the same workflow |

The LLM agent optimizer is implemented inside `ragtune_benchmark.ipynb` (cells 4–7) rather than as a standalone module; it calls `litellm.completion(model="gpt-4o-mini")` with a structured JSON response schema.

The evaluation metric (NDCG@10) is implemented in `TrialEvaluator` using graded gain `2^rel − 1` with IDCG normalisation over the full qrel set, matching the BEIR paper's definition. PyTerrier's built-in NDCG measures are not used here because the controller outputs a Python list of `ScoredDocument` objects, not a PyTerrier result DataFrame.
