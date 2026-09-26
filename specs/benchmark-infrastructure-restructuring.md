---
spec_id: BENCH-001
author: Shuvam Banerji Seal
status: review
component: benchmarks/
issue: https://github.com/avishekanand/sir/issues/6
related_papers: "FreshStack (2504.13128), BRIGHT"
created: 2026-05-14
---

# Benchmark Infrastructure Restructuring

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

**Benchmark discovery:** Each `include:` entry maps to a filesystem path relative to the
`benchmarks/` directory. The runner imports `benchmark.py` from that path:
- `trec-covid-budget-ablation` → `benchmarks/trec-covid-budget-ablation/benchmark.py`
- `synthetic/loop_efficiency` → `benchmarks/synthetic/loop_efficiency/benchmark.py`

Both flat (`benchmarks/<name>/`) and grouped (`benchmarks/<group>/<name>/`) layouts are
supported. The `include:` value is the path after `benchmarks/`, using `/` separators for
nested directories.

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
[CI & Benchmarking](https://github.com/avishekanand/sir/wiki/CI-and-Benchmarking) page
("Benchmark Discipline — Six Required Fields"). These are distinct from the
[Reproducibility Rules](https://github.com/avishekanand/sir/wiki/Reproducibility-Rules)
fields used for experiment logs in `docs/experiments/`. Every benchmark result must
contain all six — a result missing any field is incomplete.

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

The runner writes these fields to the SQLite database. Per the Reproducibility Rules wiki:
"If another researcher cannot reproduce an experiment from the repository artifacts alone —
the spec, the config, and the commit hash — the experiment is considered incomplete."

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

**CI migration note:** The current fast CI pipeline runs `BENCH_FAST=1 pytest tests/benchmarks`.
Moving synthetic benchmarks out of `tests/benchmarks/` will remove them from CI — this is
intentional. Per review feedback, benchmarks must not run in CI (too slow and flaky).
Phase 1 must:
1. Remove `BENCH_FAST=1 pytest tests/benchmarks` from the CI pipeline.
2. Update `make run-benchmarks` to call the new runner (`python benchmarks/core/runner.py`).
3. Delete the old `tests/benchmarks/` directory after migration.

The runner (`python benchmarks/core/runner.py`) becomes the sole entry point for all
benchmark execution.

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

**Note:** This workflow applies to benchmark *implementation* PRs. Spec PRs for new
benchmarks follow the standard lifecycle (spec branch → `main`), per the wiki's
[Lifecycle of Work](https://github.com/avishekanand/sir/wiki/Lifecycle-of-Work).

### Bootstrap

The `benchmark` integration branch is created once by the PI (or project admin) from
`main` and kept long-lived. It is never deleted — it accumulates validated benchmark code
until a merge window opens to bring it into `main`.

### Branch workflow

1. Fork the repository and create a feature branch from `benchmark` (not `main`).
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

## 7. Test Plan

The following tests must be implemented in Phase 1 alongside the infrastructure:

| Test file | Test function | What it verifies |
|---|---|---|
| `tests/unit/test_benchmark_runner.py` | `test_runner_discovers_flat_benchmarks` | Runner finds `benchmarks/<name>/benchmark.py` |
| `tests/unit/test_benchmark_runner.py` | `test_runner_discovers_nested_benchmarks` | Runner finds `benchmarks/<group>/<name>/benchmark.py` |
| `tests/unit/test_benchmark_runner.py` | `test_runner_merges_overrides` | Root config overrides are merged with per-benchmark config |
| `tests/unit/test_benchmark_runner.py` | `test_runner_writes_to_sqlite` | Results are written as a new row in the database |
| `tests/unit/test_benchmark_runner.py` | `test_runner_preserves_history` | Multiple runs of the same benchmark create separate rows |
| `tests/unit/test_benchmark_db.py` | `test_db_schema_has_six_metadata_fields` | Runs table has all six required metadata columns |
| `tests/unit/test_benchmark_db.py` | `test_db_query_by_commit_hash` | Results can be queried by commit hash |
| `tests/unit/test_benchmark_metrics.py` | `test_ndcg_at_5_calculation` | NDCG@5 matches expected values on known inputs |
| `tests/unit/test_benchmark_metrics.py` | `test_recall_at_5_calculation` | Recall@5 matches expected values on known inputs |

All tests use synthetic data (no API calls, no real datasets). The existing synthetic
benchmarks (`loop_efficiency`, `intelligence_gain`) serve as integration smoke tests.

## 8. Out of Scope

This spec does **not** cover:

1. **CI pipeline redesign** — Benchmarks are removed from CI per this spec, but the
   broader CI pipeline configuration (GitHub Actions workflows, other CI checks) is
   not in scope.
2. **Benchmark result visualization** — Reporting/query tools are in scope
   (`benchmarks/core/reporting.py`), but dashboards, plots, or web UIs are not.
3. **New benchmark implementations** — This spec defines the infrastructure. Specific
   benchmarks (TREC-COVID, NFCorpus, BRIGHT) are migrated from existing code, not
   newly implemented here.
4. **Experiment logging system** — The `docs/experiments/` workflow (Reproducibility
   Rules wiki) is a separate system. Benchmark results in SQLite do not replace
   experiment logs.
5. **GPU hardware provisioning** — The spec addresses how benchmarks signal GPU
   requirements, not how GPU resources are provisioned or managed.

## 9. Resolved Design Decisions

These items were originally open questions and have been resolved (per review feedback):

1. **Storage format:** Results are stored in SQLite (`benchmarks/results.db`, gitignored),
   not as committed JSON/CSV files. Each run is a row keyed by `(commit_hash, timestamp)`,
   preserving full history without clobbering prior results.

2. **Data commitment:** Benchmark data (qrels, corpora) is never committed to the
   repository. All data is downloaded at runtime via HuggingFace datasets, ir_datasets,
   or a provided download script.

3. **Metadata contract:** The six benchmark metadata fields are:
   `commit_hash`, `dataset`, `config_file`, `random_seed`, `hardware`, `timestamp`.
   These are columns in the SQLite runs table, mandated by the wiki's
   [CI & Benchmarking](https://github.com/avishekanand/sir/wiki/CI-and-Benchmarking) page.

## 10. Open Questions for Discussion

1. Should the runner be a standalone script (`python benchmarks/core/runner.py --config benchmarks/config.yaml`) or integrated into the Makefile as `make run-benchmarks` (updated)? Both are possible; the Makefile target would call the runner internally.

2. How should GPU-dependent benchmarks (e.g., those requiring a cross-encoder or LLM)
   signal their hardware requirements? Options include a `requires: gpu` field in each
   benchmark's `config.yaml` or a `skip_if_unavailable` mechanism in the runner.
