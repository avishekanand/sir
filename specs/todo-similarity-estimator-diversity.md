# TODO: Fix SimilarityEstimator for Diversity-Sensitive Metrics

**Context**: `specs/freshstack-similarity-estimator-bug.md` — read that first.

**Branch prefix**: `fix/similarity-estimator-diversity`

---

## Problem Statement

`SimilarityEstimator` currently boosts candidates that are *similar* to high-scoring docs found in previous batches. This is an anti-diversity strategy that hurts α-NDCG and Coverage — the primary metrics on FreshStack and any nugget-based benchmark. It also uses a hard absolute score threshold (`> 0.8`) on raw cross-encoder logits, making the behavior domain-dependent and unpredictable.

---

## Tasks

### 1. Fix winner detection — replace absolute threshold with relative selection

**File**: `src/ragtune/components/estimators.py`, `SimilarityEstimator.value()`

Current:
```python
winners = [it for it in reranked if (it.reranker_score or 0) > 0.8]
```

Replace with the top-N of already-reranked items by score (e.g., top half), which is scale-invariant regardless of the model's output distribution:
```python
reranked_sorted = sorted(reranked, key=lambda x: x.reranker_score or 0.0, reverse=True)
covered = reranked_sorted[: max(1, len(reranked_sorted) // 2)]
```

The variable should be renamed from `winners` to `covered` to reflect its new meaning.

### 2. Invert the signal — penalize similarity instead of boosting it (MMR-style)

**File**: same as above.

Current:
```python
boost_weight = 0.5
priorities[it.doc_id].priority *= (1.0 + max_sims[i] * boost_weight)
```

Replace with a diversity penalty:
```python
diversity_weight = 0.5
priorities[it.doc_id].priority *= (1.0 - max_sims[i] * diversity_weight)
priorities[it.doc_id].predicted_quality = 1.0 - max_sims[i]
```

This prioritizes candidates that are *unlike* what has already been reranked — the MMR principle. Docs covering new topics bubble up; redundant docs are deprioritized.

### 3. Update docstring

The class docstring at line 50 still says "Predicts utility using semantic similarity (Embeddings)." Update it to reflect the MMR intent:

```
Prioritizes candidates that cover topics not yet seen in reranked results
(MMR-style diversity). After each batch, docs semantically similar to
already-reranked content are down-weighted so that subsequent batches
broaden nugget coverage.
```

### 4. Add unit test

**File**: `tests/unit/components/test_estimators.py` (create if it doesn't exist)

Test: given a pool where some items are already reranked and two eligible candidates — one highly similar to the reranked set, one dissimilar — verify that `SimilarityEstimator.value()` assigns higher priority to the dissimilar candidate.

Use a fake embedding model or mock `SentenceTransformer.encode()` to avoid a real model dependency in unit tests (consistent with the fake-component pattern in `tests/conftest.py`).

### 5. Verify on FreshStack benchmark

Re-run `python scripts/benchmark_freshstack.py` and confirm:
- RAGtune (budget=20) ≥ Static Rerank (budget=20) on Coverage@20 for both domains
- RAGtune (budget=20) ≥ RAGtune (budget=10) on Coverage@20 (more budget should help)
- Results are no longer identical across langchain and yolo for the estimator activation path

Paste the output table in the PR description.

---

## What NOT to Change

- Do not touch `BaselineEstimator` or `UtilityEstimator` — they are not part of this bug.
- Do not change the `build_scenarios()` function in the benchmark — it correctly uses `SimilarityEstimator` for RAGtune scenarios.
- Do not change `diversity_weight = 0.5` to anything else without benchmark evidence — that value is a starting point, tuning belongs in a separate experiment PR.

---

## Definition of Done

- [ ] `SimilarityEstimator.value()` uses relative winner selection (top-half by score)
- [ ] Priority signal is inverted to penalize similarity (MMR-style)
- [ ] Docstring updated
- [ ] Unit test passes with mocked embeddings
- [ ] FreshStack table shows RAGtune (budget=20) ≥ Static Rerank (budget=20) on Coverage@20
- [ ] PR description includes the before/after benchmark table
