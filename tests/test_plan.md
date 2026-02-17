# Test Plan: RAGtune Core 0.54 (Final Status)

This document tracks the status and coverage of tests for the stateful iterative reranking architecture.

## Execution Summary
- **Total Tests**: 44
- **Passed**: 44
- **Failed**: 0
- **Verification Date**: 2026-02-15

## Status Tracker

| Group | Tests | Files | Status |
| :--- | :--- | :--- | :--- |
| **A. Pool** | A1-A8 | `test_pool_state_machine.py` | [x] PASSED |
| **B. Estimator** | B9-B12 | `test_estimator_contract.py` | [x] PASSED |
| **C. Scheduler** | C13-C17 | `test_scheduler_contract.py` | [x] PASSED |
| **D. CostTracker** | D18-D21 | `test_cost_tracker.py` | [x] PASSED |
| **E. Controller** | E22-E30 | `test_controller_loop.py` | [x] PASSED |
| **I. Integration** | I1-I6 | `tests/integration/` | [x] PASSED |

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

### I) Integration Scenarios
- [x] **I1: Local improve**: Simulated reranking improves top-k ordering.
- [x] **I2: Budget Tradeoff**: verified performance/cost scaling.
- [x] **I3: Union Provenance**: Multi-source retrieval deduping.
- [x] **I4: Determinism**: Identical runs yield identical scores.
- [x] **I5: PyTerrier Integration**: Integration with PT BatchRetrieve.
- [x] **I6: LangChain Adapter**: Smoke test for LC RAG integration.
