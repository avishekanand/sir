# Benchmark Infrastructure Restructuring

## Status: Draft Proposal

| Field | Value |
|---|---|
| Author | Shuvam Banerji Seal |
| Created | 2026-05-14 |
| Status | Draft / Discussion |
| Depends on | None |

## 1. Motivation

The current benchmark code is scattered across two locations with no consistent structure. Benchmark scripts live under `scripts/` alongside unrelated utility scripts, synthetic benchmark tests live under `tests/benchmarks/` as standalone Python files, and experiment configuration is hardcoded in script arguments rather than declared in config files. Adding a new benchmark requires understanding the internals of `experiment_grid.py` or `benchmark_suite.py`, copying patterns from existing scripts, and manually managing result output paths. There is no standard interface for a benchmark, no result contract, and no config-driven way to select which benchmarks to run. This creates a barrier for contributors who want to add a new dataset or evaluation scenario without modifying core infrastructure.

## 2. Proposed Directory Layout

A new top-level `benchmarks/` directory replaces both `scripts/benchmark_*` and `tests/benchmarks/` as the single location for all benchmark code. The directory is self-contained: each benchmark has its own folder, its own config, and its own entry point. A root config.yaml selects which benchmarks to run and where results go.

```
benchmarks/
  config.yaml                  # Root config: which benchmarks to run, global overrides
  results.db                   # SQLite database for all benchmark results (gitignored)
  core/                        # Shared benchmark infrastructure
    __init__.py
    runner.py                  # Reads config.yaml, discovers benchmarks, runs them
    db.py                      # SQLite schema, write/read helpers
    metrics.py                 # Metric calculation (NDCG, Recall, MRR, latency)
    reporting.py               # Results aggregation, query/export from SQLite
    fixtures.py                # Synthetic data generators for fast checks
  <benchmark_name>/            # One folder per benchmark
    __init__.py
    benchmark.py               # Required: implements run(config) -> dict
    config.yaml                # Optional: per-benchmark config defaults
    README.md                  # Required: dataset source, citation, expected metrics
```

### 2.1 Storage: SQLite, not committed files

All benchmark results (metrics, per-query results, and metadata) are stored in a single
SQLite database at `benchmarks/results.db`. This file is gitignored — it is a local
artifact, never committed. Each benchmark run becomes a row keyed by `commit_hash` and
`timestamp`, preserving full history without clobbering prior results.

Benchmark data (qrels, corpora) is **never committed** to the repository. All data is
downloaded at runtime via HuggingFace datasets, ir_datasets, or a provided download
script in the benchmark folder. This resolves the open question about committed `data/`
directories: nothing is committed.

### 2.2 Gitignore rules

```
benchmarks/results.db
benchmarks/*.db
```

Large datasets are downloaded at runtime and stored in a local cache directory outside
the repository (e.g., `~/.cache/sir-benchmarks/`), not under `benchmarks/`.

## 3. Root config.yaml Schema

```yaml
benchmarks:
  # List of benchmark folder names to run (flat names or grouped paths)
  include:
    - trec-covid-budget-ablation
    - nfcorpus-estimator-comparison
    - bright-domain-evaluation
    - synthetic/loop_efficiency
    - synthetic/intelligence_gain

  # Global overrides applied to every benchmark
  overrides:
    max_queries: 50
    rerank_docs: 15
    batch_size: 5

database:
  path: benchmarks/results.db    # SQLite database (gitignored)

environment:
  seed: 42
  device: cpu                    # cpu or cuda
  verbose: false
```

The runner reads `include`, iterates over each benchmark folder, loads its `config.yaml`,
merges global overrides, calls `benchmark.run(merged_config)`, and writes results as a
new row in the SQLite database. Each row is keyed by `(commit_hash, timestamp)`, so
multiple runs of the same benchmark are preserved rather than clobbered.

## 4. Benchmark Interface

Every benchmark folder must expose a `run` function in its `benchmark.py` that conforms to this contract:

```python
def run(config: dict) -> dict:
    """Execute the benchmark and return structured results.

    Args:
        config: Merged configuration (root overrides + benchmark config.yaml)

    Returns:
        dict with keys:
            - metrics: dict of scalar metrics (ndcg_at_5, recall_at_5, mrr, latency_ms, rerank_docs)
            - results: list of dicts with per-query results
            - metadata: dict with commit_hash, dataset, config_file, random_seed, hardware, timestamp
    """
    ...
```

### 4.1 Required return keys

The six metadata fields below are mandated by the project wiki's
[Reproducibility Rules](https://github.com/avishekanand/sir/wiki/Reproducibility-Rules).
Every benchmark result must contain all six — a result missing any field is incomplete.

| Key | Type | Example |
|---|---|---|
| `metrics.ndcg_at_5` | float | 0.6576 |
| `metrics.recall_at_5` | float | 0.1173 |
| `metrics.mrr` | float | 0.6485 |
| `metrics.latency_ms` | float | 1524.0 |
| `metrics.rerank_docs` | float | 14.2 |
| `metadata.commit_hash` | str | a3f9c12 |
| `metadata.dataset` | str | beir/nfcorpus |
| `metadata.config_file` | str | benchmarks/trec-covid-budget-ablation/config.yaml |
| `metadata.random_seed` | int | 42 |
| `metadata.hardware` | str | A100 80GB |
| `metadata.timestamp` | str | 2026-05-10T14:22:00Z |
| `results[].query_id` | str | q0 |
| `results[].found_at_rank` | int | 1 |
| `results[].ndcg_at_5` | float | 0.85 |
| `results[].latency_ms` | float | 312 |

The runner writes these fields to the SQLite database. Per the wiki: "If another researcher
cannot reproduce an experiment from the repository artifacts alone — the spec, the config,
and the commit hash — the experiment is considered incomplete."

## 5. Migration Path

### 5.1 What moves from scripts/ to benchmarks/

| Current location | New location |
|---|---|
| `scripts/benchmark_utils.py` | `benchmarks/core/metrics.py` |
| `scripts/experiment_grid.py` | Split into per-benchmark folders |
| `scripts/benchmark_suite.py` | Logic merged into `benchmarks/core/runner.py` |
| `scripts/benchmark_bright.py` | Removed (replaced by bright folder) |
| `scripts/sample_bright.py` | Benchmarks download data themselves |
| `scripts/summarize_results.py` | `benchmarks/core/reporting.py` |
| `tests/benchmarks/loop_efficiency.py` | `benchmarks/synthetic/loop_efficiency/` |
| `tests/benchmarks/intelligence_gain.py` | `benchmarks/synthetic/intelligence_gain/` |

### 5.2 Initial benchmark folders to create

1. **synthetic/loop_efficiency** (from `tests/benchmarks/loop_efficiency.py`)
   Tests how efficiently the iterative loop finds golden documents in a noisy pool.
   Synthetic data, runs in under 2 minutes.

2. **synthetic/intelligence_gain** (from `tests/benchmarks/intelligence_gain.py`)
   Tests estimator quality for finding semantically similar golden documents.
   Synthetic data, runs in under 2 minutes.

3. **trec-covid-budget-ablation** (from experiment_grid Group B)
   Budget ablation on TREC-COVID with MonoT5 at Tight/Medium/Loose levels.

4. **nfcorpus-estimator-comparison** (from experiment_grid Group C)
   Estimator comparison (Baseline vs Similarity vs ReformIR) on NFCorpus.

5. **bright-domain-evaluation** (from `scripts/benchmark_bright.py`)
   Full BRIGHT evaluation across biology, coding, mathematics domains.

### 5.3 Deprecation timeline

Phase 1 (first implementation PR): Create `benchmarks/` directory with `core/` infrastructure
including the SQLite schema (`core/db.py`), migrate all synthetic benchmarks
(`tests/benchmarks/`), add the new structure alongside existing scripts.

Phase 2 (subsequent PR): Migrate `scripts/experiment_grid.py` groups into individual
benchmark folders.

Phase 3 (final PR): Remove old scripts and update `Makefile` targets.

## 6. Adding a New Benchmark

New benchmarks use a two-stage merge workflow via a long-lived `benchmark` integration
branch. This keeps unverified benchmark code off `main` without requiring CI for
benchmarks (which are too slow and flaky for the CI pipeline).

### Branch workflow

1. Fork the repository and create a branch from `benchmark` (not `main`).
2. Under `benchmarks/`, create a new folder `benchmarks/<name>/` (or
   `benchmarks/<group>/<name>/` for grouped benchmarks like `synthetic/`).
3. Add `benchmark.py` with a `run(config) -> dict` function.
4. Add `config.yaml` with benchmark-specific defaults.
5. Add `README.md` describing the dataset source, citation, and expected metrics.
6. Update `benchmarks/config.yaml` to include the new benchmark name in `include`.
7. Run locally: `python benchmarks/core/runner.py --config benchmarks/config.yaml`.
8. Verify results appear in the local SQLite database (`benchmarks/results.db`).
9. Submit a PR targeting the `benchmark` branch.

### Merge gates

- **Code-complete**: The PR is reviewed and merged into `benchmark`. At this point the
  benchmark code is correct but results have not been validated.
- **Benchmark-complete**: Once the benchmark has been run and its results validated
  (metrics match expected values, no regressions), the `benchmark` branch merges into
  `main`.

No changes to `benchmarks/core/` are required unless a new metric or output format is
needed.

## 7. Open Questions for Discussion

1. Should the runner be a standalone script (`python benchmarks/core/runner.py --config benchmarks/config.yaml`) or integrated into the Makefile as `make run-benchmarks` (updated)? Both are possible; the Makefile target would call the runner internally.

2. **(Resolved)** Benchmark data and results are never committed. Data is downloaded at
   runtime; results live in a local SQLite database (`benchmarks/results.db`, gitignored).
   Each run is a row keyed by `(commit_hash, timestamp)`, preserving full history.

3. How should GPU-dependent benchmarks (e.g., those requiring a cross-encoder or LLM)
   signal their hardware requirements? Options include a `requires: gpu` field in each
   benchmark's `config.yaml` or a `skip_if_unavailable` mechanism in the runner.

4. **(Resolved)** Results are stored in SQLite with a fixed schema, not as standalone
   JSON/CSV files. The six reproducibility metadata fields (commit_hash, dataset,
   config_file, random_seed, hardware, timestamp) are columns in the runs table.
   Migration to a standard schema (e.g., MLCommons) is a future concern if needed.
