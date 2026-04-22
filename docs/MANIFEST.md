# RAGtune Documentation Manifest

_Last updated: 2026-04-22_

This file is the authoritative record of documentation coverage. Update it when you add a feature or write a doc. Run `python scripts/check_docs.py` to get a coverage summary.

Status legend: `✅ done` | `⚠️ partial` | `❌ missing`

---

## Core

| Component | Source | Doc file | Example | Status |
|---|---|---|---|---|
| RAGtuneController | `src/ragtune/core/controller.py` | `docs/concepts/controller-estimator-scheduler.md` | `examples/quickstart.py` | ✅ done |
| CandidatePool | `src/ragtune/core/pool.py` | `docs/concepts/controller-estimator-scheduler.md` | — | ⚠️ partial |
| CostBudget | `src/ragtune/core/budget.py` | `docs/concepts/controller-estimator-scheduler.md` | `examples/demo_pyterrier_budgets.py` | ⚠️ partial |
| ScoredDocument / RAGtuneContext | `src/ragtune/core/types.py` | — | — | ❌ missing |
| ControllerTrace | `src/ragtune/core/types.py` | `docs/concepts/controller-trace.md` | — | ✅ done |

---

## Estimators

| Component | Source | Doc file | Example | Status |
|---|---|---|---|---|
| BaselineEstimator | `src/ragtune/components/estimators.py` | `docs/concepts/controller-estimator-scheduler.md` | `examples/quickstart.py` | ✅ done |
| SimilarityEstimator | `src/ragtune/components/estimators.py` | `docs/concepts/controller-estimator-scheduler.md` | — | ⚠️ partial |
| ReformIREstimator | `src/ragtune/components/estimators.py` | `docs/concepts/controller-estimator-scheduler.md` | — | ⚠️ partial |
| UtilityEstimator | `src/ragtune/components/estimators.py` | — | — | ❌ missing |

---

## Schedulers

| Component | Source | Doc file | Example | Status |
|---|---|---|---|---|
| ActiveLearningScheduler | `src/ragtune/components/schedulers.py` | `docs/concepts/controller-estimator-scheduler.md` | `examples/demo_active_learning.py` | ✅ done |
| GracefulDegradationScheduler | `src/ragtune/components/schedulers.py` | — | `examples/demo_scheduler.py` | ⚠️ partial |

---

## Rerankers

| Component | Source | Doc file | Example | Status |
|---|---|---|---|---|
| NoOpReranker | `src/ragtune/components/rerankers.py` | — | `examples/quickstart.py` | ⚠️ partial |
| CrossEncoderReranker | `src/ragtune/components/rerankers.py` | — | — | ❌ missing |
| MonoT5Reranker | `src/ragtune/components/rerankers.py` | — | `examples/demo_trec_covid_comparison.py` | ⚠️ partial |
| OllamaListwiseReranker | `src/ragtune/components/rerankers.py` | — | — | ❌ missing |
| SimulatedReranker | `src/ragtune/components/rerankers.py` | — | `examples/demo_trec_covid_comparison.py` | ⚠️ partial |

---

## Reformulators

| Component | Source | Doc file | Example | Status |
|---|---|---|---|---|
| IdentityReformulator | `src/ragtune/components/reformulators.py` | — | `examples/quickstart.py` | ⚠️ partial |
| ReformIRReformulator | `src/ragtune/components/reformulators.py` | — | — | ❌ missing |

---

## Feedback

| Component | Source | Doc file | Example | Status |
|---|---|---|---|---|
| BudgetStopFeedback | `src/ragtune/components/feedback.py` | — | — | ❌ missing |
| ReformIRConvergenceFeedback | `src/ragtune/components/feedback.py` | — | — | ❌ missing |

---

## Adapters

| Component | Source | Doc file | Example | Status |
|---|---|---|---|---|
| PyTerrierRetriever | `src/ragtune/adapters/pyterrier.py` | — | `examples/demo_pyterrier_bright.py` | ⚠️ partial |
| LangChainRetriever | `src/ragtune/adapters/langchain.py` | — | `examples/demo_langchain_retriever.py` | ⚠️ partial |

---

## CLI

| Command | Source | Doc file | Example | Status |
|---|---|---|---|---|
| `ragtune init` | `src/ragtune/cli/main.py` | `docs/cli.md` | — | ✅ done |
| `ragtune list` | `src/ragtune/cli/main.py` | `docs/cli.md` | — | ✅ done |
| `ragtune validate` | `src/ragtune/cli/main.py` | `docs/cli.md` | — | ✅ done |
| `ragtune run` | `src/ragtune/cli/main.py` | `docs/cli.md` | `examples/simple_pipeline.yaml` | ✅ done |
| `ragtune visualize` | `src/ragtune/cli/main.py` | `docs/cli.md` | — | ✅ done |

---

## Dataset Integration

| Dataset | Task type | Qrel source | Onboarding guide | Status |
|---|---|---|---|---|
| trec-covid | retrieval_graded | gold | `docs/onboarding/dataset_integration.md` | ✅ done |
| nfcorpus | retrieval_graded | gold | `docs/onboarding/dataset_integration.md` | ✅ done |
| scifact | retrieval_binary | gold | `docs/onboarding/dataset_integration.md` | ✅ done |
| BEIR (any) | retrieval_binary/graded | gold | `docs/onboarding/dataset_integration.md` + `examples/onboard_beir_dataset.py` | ✅ done |
| QA (factoid, no qrels) | open_qa_factoid | proxy_exact / proxy_token_f1 | `docs/onboarding/dataset_integration.md` + `examples/onboard_qa_dataset.py` | ✅ done |
| Custom (no ir_datasets) | any | custom | `docs/onboarding/dataset_integration.md` + `examples/onboard_custom_dataset.py` | ✅ done |

---

## Summary

Run `python scripts/check_docs.py` for live counts.

| Category | Documented | Partial | Missing | Total |
|---|---|---|---|---|
| Core | 2 | 2 | 1 | 5 |
| Estimators | 3 | 0 | 1 | 4 |
| Schedulers | 1 | 1 | 0 | 2 |
| Rerankers | 0 | 3 | 2 | 5 |
| Reformulators | 0 | 1 | 1 | 2 |
| Feedback | 0 | 0 | 2 | 2 |
| Adapters | 0 | 2 | 0 | 2 |
| CLI | 5 | 0 | 0 | 5 |
| Datasets | 6 | 0 | 0 | 6 |
| **Total** | **17** | **9** | **7** | **33** |
