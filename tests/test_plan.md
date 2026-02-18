# Test Plan: RAGtune Core 0.55 (Final Status)

This document tracks the status and coverage of tests for the stateful iterative reranking architecture.

## Execution Summary
- **Total Tests**: 75
- **Passed**: 75
- **Failed**: 0
- **Verification Date**: 2026-02-18

## Status Tracker

| Group | Tests | Files | Status |
| :--- | :--- | :--- | :--- |
| **A. Pool** | A1-A8 | `test_pool_state_machine.py` | [x] PASSED |
| **B. Estimator** | B9-B12 | `test_estimator_contract.py` | [x] PASSED |
| **C. Scheduler** | C13-C17 | `test_scheduler_contract.py` | [x] PASSED |
| **D. CostTracker** | D18-D21 | `test_cost_tracker.py` | [x] PASSED |
| **E. Controller** | E22-E30 | `test_controller_loop.py` | [x] PASSED |
| **F. Reformulation** | F31-F41 | `test_reformulators.py` | [x] PASSED |
| **G. Aggregation** | G41-G44 | `test_pool_v0_55.py` | [x] PASSED |
| **I. Integration** | I1-I6 | `tests/integration/` | [x] PASSED |
| **J. Multi-round** | J7-J12 | `test_controller_0_55.py` | [x] PASSED |
| **K. Black-box E2E** | K13-K23 | `test_e2e_v0_55.py` | [x] PASSED |

## Detailed Coverage

### A) CandidatePool State Machine
- [x] **A1: Allowed transitions succeed**: CANDIDATE → IN_FLIGHT → RERANKED.
- [x] **A2: Illegal transition raises**: RERANKED → IN_FLIGHT (IllegalTransitionError).
- [x] **A3: Unknown ID handling**: Skip/None for missing doc_ids.
- [x] **A4: update_scores exclusivity**: Only moves IN_FLIGHT to RERANKED.
- [x] **A5: State exclusivity**: One state per document guaranteed by dict.
- [x] **A6: get_eligible filtering**: Correctly identifies candidate docs.
- [x] **A7: No lost docs**: Document count is invariant across transitions.
- [x] **A8: O(1) Stable Access**: Dictionary-backed lookups.

### B) Estimator Contracts
- [x] **B9: Eligible-only impact**: Only priority_value of candidates affected.
- [x] **B10: Determinism**: Repeatable prioritization.
- [x] **B11: Multi-source evidence**: Correct weighting across multiple retrieval paths.
- [x] **B12: Purity**: Estimators do not mutate global state.

### C) Scheduler Contracts
- [x] **C13: Eligibility enforcement**: Never selects IN_FLIGHT or RERANKED docs.
- [x] **C14: Empty handling**: Returns None if pool effectively exhausted.
- [x] **C15: Stable tie-breaking**: Uses (initial_rank, doc_id) for stability.
- [x] **C16: Budget awareness**: Batch size shrinks to fit remaining docs.
- [x] **C17: Strategy escalation**: Correctly moves between strategy levels.

### D) CostTracker Invariants
- [x] **D18: Hard limits**: try_consume returns False when over-capacity.
- [x] **D19: Controller mutation**: Tracker prevents external mutations.
- [x] **D20: Monotonicity**: Budget consumed only increases (tracks overage).
- [x] **D21: Latency budget**: Marker exhausted on timeout.

### E) Controller Iterative Loop (Isolated)
- [x] **E22: Loop termination**: Stops when budget exhausted or Scheduler returns None.
- [x] **E23: State sequencing**: Verified via trace events.
- [x] **E24: Exception handling**: Reranker failure leads to DROPPED state.
- [x] **E25: Partial results**: Results not returned by reranker are DROPPED.
- [x] **E26: Traceability**: records rerank_batch, rerank_error.
- [x] **E27: Assembly correctness**: GreedyAssembler sorts by priority then initial_rank.
- [x] **E28: Final score precedence**: Reranker > Priority > Retrieval.
- [x] **E29: Budget Snapshot**: Final budget state correctly captured.
- [x] **E30: Trace Aggregation**: Cumulative events across rounds recorded.

### F) LLM Reformulation (v0.55)
- [x] **F31: Clean JSON parsing**: Returns list of strings length $m$.
- [x] **F32: Strip code fences**: Removes ` ```json ` blocks correctly.
- [x] **F33: Text wrapping**: Extracts JSON from leading/trailing conversational text.
- [x] **F34: Original query drop**: Automatically excludes the input query from results.
- [x] **F35: Whitespace cleanup**: Drops empty or purely whitespace strings.
- [x] **F36: Near-duplicate filtering**: Filters variations with >0.8 similarity.
- [x] **F37: Non-JSON resilience**: Returns empty list + no exception on junk text.
- [x] **F38: Malformed JSON resilience**: Graceful empty list for unclosed brackets etc.
- [x] **F39: Deterministic mock**: Reproducible behavior with fixed mock responses.
- [x] **F40: Async parity**: `agenerate` matches `generate` behavior.
- [x] **F41: Max token enforcement**: Correctly passes `max_tokens` to LLM completion.

### G) Multi-round Aggregation (v0.55)
- [x] **G41: Dedup by doc_id**: Single entry maintained across multiple retrieval rounds.
- [x] **G42: Provenance merging**: `sources` dict records all query rounds finding the doc.
- [x] **G43: Field correctness**: `min_rank` and `appearances_count` updated correctly.
- [x] **G44: Max pool cap**: Deterministic pruning (score then ID) when above limit.

### I) Integration Scenarios
- [x] **I1: Local improve**: Simulated reranking improves top-k ordering.
- [x] **I2: Budget Tradeoff**: verified performance/cost scaling.
- [x] **I3: Union Provenance**: Multi-source retrieval deduping.
- [x] **I4: Determinism**: Identical runs yield identical scores.
- [x] **I5: PyTerrier Integration**: Integration with PT BatchRetrieve.
- [x] **I6: LangChain Adapter**: Smoke test for LC RAG integration.

### J) Multi-round Integration (v0.55)
- [x] **J7: Happy path**: Verified (m+1) retrieval calls and correct pool size.
- [x] **J8: LLM failure fallback**: Controller proceeds with only original results.
- [x] **J9: Partial retrieval success**: Loop proceeds if some rounds are empty.
- [x] **J10: Budget enforcement**: Supplemental rounds skipped if budget insufficient.
- [x] **J11: Reformulation caching**: Identical queries hit internal cache.
- [x] **J12: Trace richness**: Logs contain query variations and reward/overlap metrics.

### K) Black-box E2E Performance & Policy (v0.55)
- [x] **K13: No rewrite budget**: Falls back to original-only retrieval gracefully.
- [x] **K14: No extra retrieval budget**: Stops after LLM rewrite if retrieval budget exhausted.
- [x] **K15: Partial supplemental retrieval**: Executes only as many rewrite rounds as budget allows.
- [x] **K16: Rerank budget exhaustion**: Pipeline terminates early without over-spending.
- [x] **K17: Estimator gating (Skip)**: Skip rewrites for high-confidence/low-noise pools.
- [x] **K18: Estimator gating (Trigger)**: Triggers rewrites for low-recall/marginal pools.
- [x] **K19: Estimator disagreement**: Deterministic resolution policy (Pessimistic: any rewrite request wins).
- [x] **K20: Budget-aware Estimator**: Estimator shifts policy based on remaining tokens/latency.
- [x] **K21: Rewrite Deduplication**: Avoids redundant retrieval calls for duplicate rewrite variations.
- [x] **K22: Noise Penalty**: Handles off-topic rewrites by penalizing them in pool metrics.
- [x] **K23: Observability Invariants**: Verifies cumulative spend <= budget for complex interwoven flows.
