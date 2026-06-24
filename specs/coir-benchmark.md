# CoIR Benchmark Integration Spec

## What problem does this solve?

RAGtune currently lacks evaluation on code-domain retrieval tasks. FreshStack covers
GitHub/StackOverflow as a corpus, but does not test query-to-code or query-to-SQL
retrieval — the modalities where budgeted reranking is most commercially relevant.

CoIR (Code Information Retrieval) is a community benchmark that covers five distinct
code retrieval subtasks with public BEIR-format datasets. Adding it lets us validate
RAGtune's budget-constrained reranking on code queries and measure the strategy gap
(No-Rerank vs. Static vs. RAGtune) in a domain that is meaningfully different from
general web or technical forum retrieval.

## What changed and why?

**New file: `scripts/benchmark_coir.py`**

Follows the same design as `benchmark_obliq.py` and `benchmark_crumb.py`:
- `load_task(name)` → HuggingFace BEIR-format load (corpus / queries / qrels configs)
- `build_retriever(corpus, qrels)` → FAISS index with gold-doc preservation, 5 000-doc cap
- `score_results(results, qrels)` → macro-averaged NDCG@10, Recall@10, Recall@50
- `run_faiss_baseline / run_controller_scenario` → identical pattern to OBLIQ/CRUMB
- `build_scenarios(retriever)` → No-Rerank, Static Rerank (budget=20), RAGtune (budget=10/20)

**Datasets:** five tasks from `CoIR-Retrieval` on HuggingFace:

| Dataset ID | Task |
|---|---|
| `stackoverflow-qa` | Stack Overflow question → answer retrieval |
| `codefeedback-st` | Single-turn code feedback matching |
| `apps` | Algorithmic problem → solution retrieval |
| `cosqa` | Natural-language code search |
| `synthetic-text2sql` | Natural language → SQL retrieval |

**Makefile:** adds `run-coir` target.

**pyproject.toml:** adds `datasets>=2.0.0` to `[benchmarks]` extras (required for HuggingFace loading).

## How to test

```bash
# Smoke test (two datasets, 20 queries each)
python scripts/benchmark_coir.py

# Override datasets and query count
COIR_DATASETS=cosqa,synthetic-text2sql COIR_QUERIES=10 python scripts/benchmark_coir.py

# Via Makefile
make run-coir
```

Expected output: a pandas DataFrame with columns `dataset`, `scenario`, `NDCG@10`,
`Recall@10`, `Recall@50`, `Avg Rerank Docs`, `Avg Latency (ms)`.

## What the reviewer should focus on

1. `load_task` column name handling — CoIR uses `_id`, `query-id`, `corpus-id` (BEIR
   standard), verify these are consistent across all five datasets.
2. Gold-doc preservation in `build_retriever` — same guard as FreshStack/CRUMB, ensure
   no gold doc is silently dropped by the 5 000-doc cap.
3. Default dataset list and query count are conservative (2 datasets × 20 queries) to
   keep CI runtime reasonable.
