# RAGtune Documentation

## Overview

RAGtune is budget-aware iterative RAG middleware. Rather than a static retrieve-then-rerank pipeline, it runs a feedback loop that uses each reranked batch to inform which documents to prioritize next — achieving higher recall at the same or lower cost.

---

## Structure

```
docs/
├── README.md                   ← this file
│
├── design/                     ← versioned design specifications
│   ├── design_v0_54.md         baseline iterative loop
│   ├── design_v0_55.md         smart estimator + real rerankers
│   ├── design_v0_56.md         declarative config (v0.2 schema), full CLI lifecycle
│   ├── design_v0_57.md         pipeline visualization + interactive editing  ← latest
│   └── design_uncertainty_SIR.MD  U-SIR: Bayesian uncertainty modeling (in progress)
│
├── concepts/                   ← architecture explainers
│   ├── controller-estimator-scheduler.md  how the main loop works
│   ├── controller-trace.md     trace event catalogue + example traces
│   └── architecture.md         high-level system overview
│
├── presentations/              ← slide decks
│   ├── ragtune-feedback-driven-retrieval-group-talk.md   Marp source
│   └── ragtune-feedback-driven-retrieval-group-talk.pdf  rendered PDF
│
├── experiments-grid.md         ablation study: budget / estimator / feedback across 3 datasets
├── experiments-grid.pdf        rendered PDF of the experiments report
├── roadmap.md                  development phases and planned work
├── RELEASE_LOG.md              version history
└── cli.md                      CLI reference (ragtune init / index / validate / run / visualize)
```

---

## Quick Navigation

| I want to... | Go to |
|---|---|
| Understand how the loop works | `concepts/controller-estimator-scheduler.md` |
| Read trace events and debug a run | `concepts/controller-trace.md` |
| See the latest design spec | `design/design_v0_57.md` |
| Read the uncertainty modeling proposal | `design/design_uncertainty_SIR.MD` |
| See experiment results across datasets | `experiments-grid.md` |
| View or present the slides | `presentations/ragtune-feedback-driven-retrieval-group-talk.pdf` |
| Re-render the slides | `marp presentations/ragtune-feedback-driven-retrieval-group-talk.md --pdf` |
| Check what's planned next | `roadmap.md` |

---

## Key Results (Summary)

From the ablation study across NFCorpus, SciFact, and TREC-COVID:

| Config | Avg NDCG@5 | Rerank docs | Latency |
|---|---|---|---|
| BM25 only | 0.656 | 0 | 17ms |
| MonoT5 tight (5 docs) | 0.720 | 5 | 574ms |
| MonoT5 medium (15 docs) | 0.749 | 15 | 1563ms |
| MonoT5 loose (30 docs) | 0.748 | 30 | 3117ms |
| **Convergence feedback** | **0.760** | **10** | **1008ms** |

Convergence feedback matches or beats the 30-doc brute-force run while using 1/3 the documents and time. Full breakdown in `experiments-grid.md`.
