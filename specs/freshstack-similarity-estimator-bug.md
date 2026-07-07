# FreshStack Benchmark: SimilarityEstimator Underperformance Analysis

## What We Observed

Running the FreshStack benchmark with two budget-20 scenarios:

| Scenario | α-NDCG@10 | Coverage@20 | Recall@50 | Avg Rerank Docs |
|---|---|---|---|---|
| Static Rerank (budget=20) | 0.0078 | 0.0395 | 0.0431 | 20.0 |
| RAGtune (budget=20) | 0.0075 | 0.0219 | 0.0431 | 20.0 |

Both rerank 20 documents, yet RAGtune (budget=20) scores nearly **half** Coverage@20 vs Static Rerank. With an adaptive estimator and more passes through the pool, it should be doing *better* — not worse.

## Root Cause: SimilarityEstimator is Anti-Diversity

`SimilarityEstimator.value()` (`src/ragtune/components/estimators.py:58`) runs after each batch to re-prioritize remaining candidates. Its logic:

```python
winners = [it for it in reranked if (it.reranker_score or 0) > 0.8]
...
priorities[it.doc_id].priority *= (1.0 + max_sims[i] * boost_weight)
```

After each batch it identifies "winners" (score > 0.8) and **boosts candidates that are similar to them**. The intent was "find more docs like what's already relevant." For precision-oriented metrics (MAP, NDCG on a single relevant doc) this can help. For FreshStack's metrics it is actively harmful.

**Why it hurts FreshStack:**

- **α-NDCG** rewards covering distinct nuggets. Boosting similar docs causes later batches to cluster around topics already covered in batch 1.
- **Coverage@20** explicitly measures how many nuggets are covered across the top-20 results. Redundant docs waste coverage slots.
- A "find more of the same" strategy is the inverse of what diversity-sensitive metrics reward.

## Why It Only Triggers for `yolo`, Not `langchain`

The winner detection uses a **hard absolute threshold** (`> 0.8`) against raw logit scores from `ms-marco-MiniLM-L-6-v2`. That model outputs unbounded floats (not calibrated probabilities). Whether 0.8 is exceeded depends entirely on the domain's query-document distribution:

- **yolo**: some docs score > 0.8 in batch 1 → similarity boosting activates → later batches pick redundant docs → Coverage@20 drops from 0.0395 to 0.0219.
- **langchain**: no docs cross the threshold → estimator silently falls back to raw FAISS ordering → same 20 docs as Static Rerank → identical metrics.

This makes the estimator's behavior domain-dependent and unpredictable.

## Why RAGtune budget=10 Outperforms budget=20

| Scenario | α-NDCG@10 | Coverage@20 |
|---|---|---|
| RAGtune (budget=10) | 0.0093 | 0.0395 |
| RAGtune (budget=20) | 0.0075 | 0.0219 |

Budget=10 runs 2 batches of 5. If no doc in batch 1 exceeds the 0.8 threshold, the similarity boosting never activates and the estimator stays at FAISS ordering for batch 2 — the same ordered set as Static Rerank for the first 10 docs, which has good diversity. Budget=20 runs 4 batches of 5, giving the buggy estimator more chances to inject redundant picks in batches 2–4.

## Static Rerank vs FAISS Baseline

A secondary observation: both reranked budget=20 scenarios score *lower* α-NDCG@10 than the FAISS No-Rerank baseline (0.0127). This is likely because `ms-marco-MiniLM-L-6-v2` is an MS MARCO web-search model and is out-of-domain for FreshStack's GitHub/StackOverflow coding corpus. The reranker is actively misprioritizing docs. This is a separate issue from the estimator bug.

## Files Involved

| File | Relevant Lines |
|---|---|
| `src/ragtune/components/estimators.py` | `SimilarityEstimator.value()` lines 58–99 |
| `scripts/benchmark_freshstack.py` | `build_scenarios()` — uses `SimilarityEstimator` |
