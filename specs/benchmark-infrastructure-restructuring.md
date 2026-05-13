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
  results/                     # Auto-generated, gitignored
    <benchmark_name>/
      metrics.json             # Structured metrics (NDCG, Recall, MRR, latency)
      results.csv              # Per-query results with all metadata
      run.log                  # Timestamps, warnings, errors
  core/                        # Shared benchmark infrastructure
    __init__.py
    runner.py                  # Reads config.yaml, discovers benchmarks, runs them
    metrics.py                 # Metric calculation (NDCG, Recall, MRR, latency)
    reporting.py               # Results aggregation, CSV/JSON output
    fixtures.py                # Synthetic data generators for fast CI checks
  <benchmark_name>/            # One folder per benchmark
    __init__.py
    benchmark.py               # Required: implements run(config) -> dict
    config.yaml                # Optional: per-benchmark config defaults
    README.md                  # Required: dataset source, citation, expected metrics
    data/                      # Optional: small fixture files, ignored for large datasets
```

### 2.1 Gitignore rules

```
benchmarks/results/
benchmarks/*/data/large/
```

Large datasets are never stored in the repository. They are downloaded at runtime via HuggingFace datasets, ir_datasets, or a provided download script in the benchmark folder.

## 3. Root config.yaml Schema

```yaml
benchmarks:
  # List of benchmark folder names to run
  include:
    - trec-covid-budget-ablation
    - nfcorpus-estimator-comparison
    - bright-domain-evaluation

  # Global overrides applied to every benchmark
  overrides:
    max_queries: 50
    rerank_docs: 15
    batch_size: 5

results:
  path: benchmarks/results       # Relative to repo root
  format: csv                    # csv or json
  overwrite: true                # Overwrite existing results or error

environment:
  seed: 42
  device: cpu                    # cpu or cuda
  verbose: false
```

The runner reads `include`, iterates over each benchmark folder, loads its `config.yaml`, merges global overrides, calls `benchmark.run(merged_config)`, and writes results to `results/<benchmark_name>/`.

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
            - metadata: dict with dataset, model, commit_hash, seed, hardware, timestamp
    """
    ...
```

### 4.1 Required return keys

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

These six metadata fields are required by the existing reproducibility rules in the project wiki. Every result file must contain them.

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

Phase 1 (this PR): Create `benchmarks/` directory with `core/` infrastructure, migrate all synthetic benchmarks (`tests/benchmarks/`), add the new structure alongside existing scripts.

Phase 2 (next PR): Migrate `scripts/experiment_grid.py` groups into individual benchmark folders.

Phase 3 (final PR): Remove old scripts and update `Makefile` targets.

## 6. CI Integration

The fast CI pipeline (target under 10 minutes) runs only synthetic benchmarks:

```bash
pytest benchmarks/ --benchmark-mode=synthetic
```

The slow CI pipeline (nightly) runs all benchmarks:

```bash
python benchmarks/core/runner.py --config benchmarks/config.yaml
```

This separation matches the existing CI and Benchmarking wiki page which distinguishes fast synthetic checks from slow full runs.

## 7. Adding a New Benchmark

A contributor adding a new benchmark follows these steps:

1. Fork the repository.
2. Under `benchmarks/`, create a new folder `benchmarks/<name>/`.
3. Add `benchmark.py` with a `run(config) -> dict` function.
4. Add `config.yaml` with benchmark-specific defaults.
5. Add `README.md` describing the dataset source, citation, and expected metrics.
6. Update `benchmarks/config.yaml` to include the new benchmark name in `include`.
7. Run locally: `python benchmarks/core/runner.py --config benchmarks/config.yaml`.
8. Verify results appear in `benchmarks/results/<name>/metrics.json`.
9. Submit a PR.

No changes to `benchmarks/core/` are required unless a new metric or output format is needed.

## 8. Open Questions for Discussion

1. Should the runner be a standalone script (`python benchmarks/core/runner.py --config benchmarks/config.yaml`) or integrated into the Makefile as `make run-benchmarks` (updated)? Both are possible; the Makefile target would call the runner internally.

2. Should per-benchmark `data/` directories accept small lookup files (e.g., 100-200 KB qrel files) in version control, or should all data always be downloaded at runtime? The proposal allows both: small files go in `data/`, large downloads go in `data/large/` which is gitignored and has a download script.

3. How should GPU-dependent benchmarks (e.g., those requiring a cross-encoder or LLM) signal their hardware requirements? Options include a `requires: gpu` field in each benchmark's `config.yaml` or a `skip_if_unavailable` mechanism in the runner.

4. Should the `metrics.json` output format be standardized across ML research projects (e.g., MLCommons schema) or kept project-specific? Keeping it project-specific for now (with the six required reproducibility fields) allows faster iteration; migration to a standard schema is a future concern.
