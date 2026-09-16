# Benchmarking & Prompt Optimization

Reference for the two scripts added in `bench/beir-exhaustive-sweep` and
`feat/prompt-optimization`:

| Script | What it searches | Artifact produced |
|---|---|---|
| `examples/beir_full_benchmark.py` | pipeline **configurations**, across a (dataset × retriever × seed) grid | Pareto pipeline YAMLs, CSVs, plots |
| `examples/prompt_optimizer.py` | **prompt text** for the LLM-backed components | `prompts.yaml` fragments |

Both reuse RAGtune's existing evaluation machinery (`ragtune.tuning.evaluator`),
so their NDCG@10 numbers are directly comparable to each other and to
`ragtune tune`.

> **Verification status.** Everything below marked *measured* was observed on a
> 40-core CPU box during development. The GPU figures are extrapolations, clearly
> marked as such. The dense-retriever, MonoT5, hybrid-RRF, LLM-reranker and
> prompt-optimizer paths have **not** been executed end to end — run
> `--preset smoke` on your hardware before trusting a long sweep.

---

## 1. Installation & running

### 1.1 Install

```bash
# 1. environment (repo needs Python >= 3.9)
conda create -y -n ragtune python=3.11
conda activate ragtune

# 2. RAGtune + benchmark dependencies
cd /path/to/sir
pip install -e ".[tuning]"
pip install python-terrier matplotlib

# 3. optional, per feature
pip install pyterrier-t5      # --rerankers monot5
pip install pyterrier-dr      # --retrievers dense-* / hybrid-rrf
export OPENAI_API_KEY=...     # --optimizers llm, --rerankers llm, prompt_optimizer.py

# 4. Java — a JDK, not a JRE (pyjnius needs javac)
conda install -y -c conda-forge openjdk=21

# 5. verify
python -c "import pyterrier as pt, ragtune; print(pt.__version__, pt.__file__)"
python examples/beir_full_benchmark.py --preset smoke --plan
```

Before a long sweep, also confirm the dataset IDs resolve:

```bash
python examples/beir_full_benchmark.py --validate-datasets --datasets standard
```

### 1.2 Run the sweep

```bash
# 0. ALWAYS size it first — prints the grid and cost projection, evaluates nothing
python examples/beir_full_benchmark.py --preset exhaustive --plan

# 1. the practical full sweep on one GPU (~45 GPU-h)
nohup python examples/beir_full_benchmark.py \
    --preset gpu-full --device cuda \
    --index-dir ./indexes --out-dir ./benchmark_results \
    > bench.log 2>&1 &
tail -f bench.log

# 2. the maximal grid (~629 GPU-h) — shard by dataset group, one per machine
python examples/beir_full_benchmark.py --preset exhaustive --device cuda \
    --datasets small       --out-dir ./results_small
python examples/beir_full_benchmark.py --preset exhaustive --device cuda \
    --datasets standard    --out-dir ./results_standard
python examples/beir_full_benchmark.py --preset exhaustive --device cuda \
    --datasets cqadupstack --out-dir ./results_cqa
python examples/beir_full_benchmark.py --preset exhaustive --device cuda \
    --datasets heavy --allow-heavy --out-dir ./results_heavy

# 3. after a crash: same flags + --resume (finished units skipped, indexes reused)
python examples/beir_full_benchmark.py --preset gpu-full --device cuda --resume

# 4. rebuild tables/plots from state.json alone — no JVM, no re-evaluation
python examples/beir_full_benchmark.py --preset gpu-full --report-only

# 5. prompt optimization (needs OPENAI_API_KEY)
python examples/prompt_optimizer.py --dataset nfcorpus --target reranker --dry-run
python examples/prompt_optimizer.py --dataset nfcorpus --target reranker \
    --iterations 20 --n-queries 20 --rerank-depth 10
```

**Sharding notes.** Give every shard its own `--out-dir`: each writes a
`state.json` and two shards sharing one would overwrite each other's progress.
A shared `--index-dir` is fine and saves re-indexing, as long as two shards
aren't building the *same* dataset's index at the same time — so shard by
dataset group, never by retriever. There is no built-in merge step; combine
shards afterwards by concatenating their `trials.csv` / `baseline.csv`.

**Before committing to days of GPU time**, run the smoke preset end to end on
the target machine — it exercises indexing, retrieval, reranking, tuning and
reporting in about 20 minutes:

```bash
python examples/beir_full_benchmark.py --preset smoke
```

### Three installation traps

| Trap | Symptom | Fix |
|---|---|---|
| **A JRE is not a JDK** | `Exception: Unable to find javac` from pyjnius | Install a real JDK. The script auto-detects `<sys.prefix>/lib/jvm` and sets `JAVA_HOME` + `JVM_PATH`, so a conda `openjdk` is enough — but `java -version` succeeding proves nothing, since pyjnius resolves `javac`. |
| **Wrong `pyterrier` package** | `module 'pyterrier' has no attribute 'started'`, no `get_dataset` | The real package is **`python-terrier`**. An unrelated PyPI package named `pyterrier` installs a module of the same name and shadows it. `pip uninstall -y pyterrier python-terrier && pip install python-terrier`. The script raises a targeted error if it detects this. |
| **`conda activate` silently not applying** | Packages install into the base env; the script runs under the wrong Python | Use the interpreter path directly: `/opt/conda/envs/ragtune/bin/python examples/...` |

`make run SCRIPT=...` does **not** work for these scripts — the Makefile hardcodes
`venv/bin/python` and the repo has no `venv/`. Call `python` directly.

---

## 2. Benchmark script

### 2.1 Model

The grid is **cells**; each cell is one `(dataset, retriever, seed)` triple. Every
cell runs up to three stages:

| Stage | What it does | Why it exists |
|---|---|---|
| `sanity` | Raw first-stage NDCG@10, no RAGtune loop, vs published BM25 | Catches broken indexes and query-parser failures in seconds, before hours of tuning |
| `baseline` | One fixed pipeline per (reranker, checkpoint) | The untuned reference number that tuning must beat |
| `tune` | TPE / GEPA / random search over the config space | Produces the Pareto front |

Select with `--stages sanity,baseline,tune`.

### 2.2 Coverage

**Datasets** — 26 BEIR tasks, every `irds` ID verified against the live
`ir_datasets` registry with exact document counts.

| Group | Members |
|---|---|
| `tiny` | nfcorpus, scifact |
| `small` | + arguana, scidocs, fiqa |
| `standard` | + trec-covid, touche2020 |
| `cqadupstack` | 12 StackExchange subforums (android … wordpress) |
| `heavy` | nq, dbpedia, hotpotqa, fever, climate-fever, msmarco — all >1M docs, gated behind `--allow-heavy` |
| `all` | everything above |

BEIR's `signal1m`, `trec-news`, `robust04` and `bioasq` require licensed corpora
and are **not** in the public registry (verified absent) — they are deliberately
omitted rather than listed and left to fail mid-sweep.

**Retrievers**

| Kind | Names |
|---|---|
| Lexical | `bm25`, `dph`, `pl2`, `dirichlet`, `tfidf` |
| Dense | `dense-minilm`, `dense-mpnet`, `dense-bge`, `dense-gte`, `dense-e5`, `dense-tasb`, `dense-ance`, `dense-tct` |
| Hybrid | `hybrid-rrf` (RRF over `--hybrid-sparse` + `--hybrid-dense`) |

Groups: `lexical`, `dense`, `small`, `standard`, `all`.

Dense retrievers carry the encoder settings their checkpoints require —
`dense-bge` uses CLS pooling plus a query instruction prefix, `dense-e5` uses the
mandatory `query:` / `passage:` prefixes. Getting these wrong silently degrades
quality rather than erroring, which is why they live in the catalogue rather than
in flags.

**Rerankers** — `noop`, `cross-encoder`, `monot5`, `llm`. Checkpoints are a
separate axis: `--ce-models` accepts `tiny` / `standard` / `all`, `--monot5-models`
accepts `standard` / `all`, and both also take an explicit comma-separated list.

**Optimizers** — `bayes` (Optuna TPE), `llm` (GEPA agent), `random` (control).
Include `random`: at small budgets it is the only way to tell whether the other
two are beating chance.

### 2.3 Presets

Always `--plan` first; it prints the grid and a cost projection without
evaluating anything.

| Preset | Grid | Projected cost |
|---|---|---|
| `smoke` | 2 datasets × 2 retrievers × 2 rerankers, 5 queries, 6 trials | ~20 min CPU (*measured*: 35 min for one dataset) |
| `standard` | 7 datasets × 4 retrievers, 50 queries, 50 trials, bayes+random | hours on GPU |
| `gpu-full` | 7 × 6 retrievers × 3 rerankers × 2 seeds, 50 q × 50 trials | **~45 GPU-h** (*estimated*) |
| `exhaustive` | 7 × 14 retrievers × 7 reranker checkpoints × 3 seeds, 100 q × 100 trials | **~629 GPU-h** (*estimated*) — split across machines by dataset group |

Explicit flags always beat the preset, so `--preset gpu-full --datasets nfcorpus`
does what it looks like.

### 2.4 CLI reference

```
grid
  --preset {smoke,standard,exhaustive,gpu-full}
  --datasets        names or a group (default nfcorpus,scifact)
  --retrievers      names or a group (default bm25,dph)
  --rerankers       noop,cross-encoder,monot5,llm
  --stages          sanity,baseline,tune
  --optimizers      bayes,llm,random
  --space           restricted | full   (full adds llm_rewrite/reformir — needs a key)
  --allow-heavy     permit >1M-doc corpora

evaluation
  --n-queries 50    queries per config evaluation
  --budget 30       trials (bayes/random) or iterations (llm) per cell
  --seed 42 / --seeds 42,43,44
  --max-cost 200            cost pruner + hypervolume reference
  --max-trial-seconds 600   runtime pruner (see the warning in §2.7)

retrieval
  --retrieval-depth 200     first-stage depth; also caps the tuner's depth range
  --bm25-k1 0.9 --bm25-b 0.4
  --dense-backend {np,torch,faiss_flat,faiss_hnsw}
  --encode-batch-size 32
  --device cpu|cuda
  --hybrid-sparse bm25 --hybrid-dense dense-minilm --rrf-k 60

baseline pipeline
  --baseline-depth 100 --baseline-rerank-budget 50
  --ce-models / --monot5-models / --llm-reranker-model

llm optimizer
  --llm-model gpt-4o-mini --llm-temperature 0.7
  --gepa-startup 3 --gepa-minibatch 10 --gepa-merge-every 10

io
  --index-dir ./indexes --out-dir ./benchmark_results
  --search-space-json FILE  arbitrary RAGtuneSearchSpace overrides, merged last
  --resume --report-only --no-plots
  --plan --validate-datasets --list-knobs
```

### 2.5 Outputs

```
benchmark_results/
├── state.json                 checkpoint — one entry per completed unit
├── sanity.csv                 raw retrieval quality per (dataset, retriever, seed)
├── baseline.csv               fixed-pipeline reference numbers
├── trials.csv                 every trial, with all p_* parameters
├── summary.md                 the printed tables
├── plots/<dataset>_<retriever>.png     Pareto front + convergence
└── pareto_configs/<dataset>_<retriever>_<optimizer>_s<seed>/*.yaml
```

The Pareto YAMLs are runnable directly:

```bash
ragtune run pareto_configs/nfcorpus_bm25_bayes_s42/pareto_trial_3_ndcg0.371_cost11.yaml -q "your query"
```

### 2.6 Recovering an interrupted run

State is written after **every** unit, so:

```bash
# continue where it stopped — finished units are skipped, indexes reused
python examples/beir_full_benchmark.py <same flags> --resume

# rebuild all tables/plots from state.json alone (no JVM, no re-evaluation)
python examples/beir_full_benchmark.py <same flags> --report-only
```

`--report-only` exists because reporting runs at the very end; without it a
killed run loses its tables even though every number is already on disk.

### 2.7 Cost model

*Measured* on a 40-core CPU box, nfcorpus, MiniLM-L6 cross-encoder:

| Quantity | Value |
|---|---|
| Retrieval + assembly, no reranking | 63 ms/query |
| Cross-encoder @ 42.6 docs/query | 17,019 ms/query |
| → per reranked document | **398 ms** |
| One tune cell (6 trials × 5 queries) | 875 s |

Per cell, 50 trials × 50 queries = 2,500 query-evaluations:

| Configuration | Per cell |
|---|---|
| noop only | 2.6 min (*measured basis*) |
| cross-encoder, CPU | ~12 h (*measured basis*) |
| cross-encoder, modern GPU | ~15–40 min (*extrapolated*) |
| + MonoT5-base | ~1–3 h (*extrapolated*) |

**The pruner interacts with this.** `--max-trial-seconds` defaults to 600, and
`RuntimePruner` projects total trial time after 3 queries. On CPU a cross-encoder
trial projects 50 × 17 s = 850 s > 600, so it is pruned in warmup — the run
finishes fast and the search collapses onto `noop`, producing a meaningless
Pareto front. On CPU either raise it past 1200 or drop the cross-encoder. The
GPU presets set 1800.

**398 ms/doc for a 22M-param model on 40 cores is suspiciously slow.** The likely
cause is the scheduler dispatching tiny batches — trials sampled
`scheduler_batch_size=2`, i.e. ~21 separate `CrossEncoder.predict()` calls per
query, each paying fixed overhead. Raising the lower bound of
`scheduler_batch_size_range` may buy a large constant factor. *Unverified.*

Estimate for your own hardware instead of trusting the table:

```bash
python examples/beir_full_benchmark.py --stages baseline \
    --datasets nfcorpus --retrievers bm25 --rerankers noop,cross-encoder --n-queries 10
# take ms/query, multiply by 2500 per cell
```

### 2.8 Tunable knobs

`--list-knobs` prints all of these from the live `RAGtuneSearchSpace`.

**Swept by the tuner** — reranker / reformulator / estimator / scheduler /
feedback type; `ce_model`, `monot5_model`, `monot5_batch_size`,
`similarity_model`, `reformulator_model`; `original_query_depth`,
`depth_per_reformulation`, `max_pool_size`, `near_duplicate_threshold`,
`scheduler_batch_size`, `gd_llm_limit`, `gd_ce_limit`, `assembler_max_docs`,
`budget_rerank_docs`, `budget_reformulations`, `reformulator_n_variants`,
`min_reranked_for_regression`, `budget_stop_token_threshold`.

**Fixed per run, exposed as flags** — Terrier weighting model, BM25 k1/b,
retrieval depth, dense encoder and search backend, hybrid fusion components.

**Hard-coded; needs a code edit** — the `tokens`, `latency_ms` and
`retrieval_calls` budgets are literals in `RAGtuneSearchSpace.build_controller`;
`k=10` is fixed in `ragtune.tuning.evaluator.ndcg_at_k` (change there for @5/@100
or to add MRR/Recall).

**Two traps.** `assembler_max_docs_range` starts at 3 upstream, and any value
below 10 mathematically caps NDCG@10 — this script clamps it to (10, 20).
`original_query_depth` above `--retrieval-depth` is wasted, so the tuner's depth
range is derived from it.

---

## 3. Prompt optimizer

### 3.1 What it is — and is not

It evolves **prompt text**. The two existing optimizers both search structured
parameters and never change a prompt:

| | Search space | Proposer |
|---|---|---|
| `tuning/optimizer.py` | structured params | Optuna TPE |
| `tuning/llm_optimizer.py` | structured params | LLM reflection (GEPA) |
| `examples/prompt_optimizer.py` | **prompt strings** | LLM reflection (GEPA) |

The loop shape is the one already validated in `llm_optimizer.py` — Pareto pool
with instance-wise parent selection, minibatch screening, single-change
reflective mutation. Only the genome differs. (This is also what GEPA is in the
literature, where the evolved artifact *is* the prompt; RAGtune's use of the
machinery on a config space is the adaptation.)

### 3.2 Targets

| Target | Component | Placeholders | Output contract |
|---|---|---|---|
| `reranker` | `LLMReranker` (pointwise) | `{query}`, `{document}` | JSON object with a numeric `relevance_score` |
| `reformulator` | `LLMReformulator` | `{query}`, `{m}` | JSON containing a list of query strings |

Candidates are rejected **before evaluation** when a placeholder is missing, a
brace is unescapable, or the template exceeds `--max-prompt-chars`. This matters:
an invalid template raises inside `str.format()` per query, and the evaluator
scores that 0.0 — so without validation the optimizer learns from a crash rather
than from a bad prompt. Rejections are fed back into the reflection context.

### 3.3 Usage

```bash
# cost projection only, no API calls
python examples/prompt_optimizer.py --dataset nfcorpus --target reranker --dry-run

# optimize
python examples/prompt_optimizer.py --dataset nfcorpus --target reranker \
    --iterations 20 --n-queries 20 --rerank-depth 10 --minibatch 5

# the honest read: evolved vs hand-written, across seeds
python examples/prompt_optimizer.py --dataset scifact --target reranker \
    --compare prompt_opt_results/prompts_best.yaml --seeds 42,43,44
```

Key flags: `--optimizer-model` (proposes prompts) vs `--component-model` (runs
inside the pipeline); `--n-failures` (worst queries shown to the reflector);
`--seed-prompt` (start from a given YAML instead of `prompts.yaml`).

### 3.4 Cost

Pointwise reranking is **one API call per document**, so a full evaluation costs
`n_queries × rerank_depth` calls:

```
calls ≈ iterations × (minibatch × depth)     # screening
      + accepted   × (n_queries × depth)     # full evaluations
      + iterations                           # reflection
```

At defaults (20 iterations, 20 queries, depth 10, minibatch 5) that is ~3,220
calls. The script prints the projection and refuses to exceed `--max-calls`
(default 10,000) without `--yes`.

### 3.5 Outputs

```
prompt_opt_results/
├── prompts_best.yaml        ready to paste into config/prompts.yaml
├── prompt_iter*_ndcg*.yaml  one per Pareto-front prompt
├── history.json             every candidate, with rationale and rejections
└── comparison.json          --compare results
```

---

## 4. Reading the results

- **`vs baseline` can be negative and that's not a bug.** At small budgets most
  trials are random startup draws (`n_startup_trials = max(5, budget//8)`), so
  the tuner can legitimately finish below a well-chosen fixed config.
- **Differences under ~0.02 NDCG are noise** at 50 queries. The report prints a
  warning when run on a single seed. Use `--seeds 42,43,44` and read the ± column.
- **Hypervolume** is the dominated area under the Pareto front (maximize NDCG,
  minimize cost), with the reference cost at `--max-cost`. Higher is better; it
  rewards fronts that are both high-quality *and* cheap.
- **Zero-NDCG configs can appear on the Pareto front.** Optuna treats the
  cheapest point as non-dominated even at NDCG 0.0, and `extract_pareto_configs`
  writes it out as a "winning" config. The hypervolume calculation filters
  `ndcg > 0`, but the YAML still gets written. Sanity-check Pareto configs before
  adopting one.

---

## 5. Known issues

| Issue | Status |
|---|---|
| **`llm` reranker is broken.** `LLMReranker` reads `config.get("prompts.reranking.pointwise")`, but `get()` reads `defaults.yaml` rather than the prompt store, and the key in `prompts.yaml` is `pointwise_scoring`. Returns `None`, then raises `AttributeError`. | **Verified.** Needs its own `fix/` branch. `prompt_optimizer.py` works around it by seeding the prompt where the component actually reads (spec §3.1). |
| **`noop` is charged the same cost as a real reranker.** The budget tracker counts documents *dispatched* to the reranker regardless of whether it does work, so a free `noop` config looks as expensive as a cross-encoder on the Pareto plot. | **Verified** (noop baseline reported 42.6 rerank docs). Framework behaviour in `CostTracker`, not the scripts. |
| **`GreedyAssembler` pins reranked docs above non-reranked ones regardless of score.** May explain configs that score exactly 0.0 — a scheduler handing it poorly-chosen docs would put them at the top of the ranking. | **Hypothesis, unverified.** |
| PR size: the benchmark script is ~1600 lines, against a 500-line rule (Rule 3). | Needs either a split into an `examples/beir_bench/` package or an explicit exception before review. |
| `src/ragtune/.env` is untracked and **not** covered by `.gitignore`. | Nothing stops a future `git add -A` from committing credentials. |

---

## 6. Related

- `specs/spec-mobo-tuning.md` — the config search space and evaluation protocol
- `specs/spec-prompt-optimization.md` — the design this prompt optimizer implements
- `examples/ragtune_benchmark.ipynb` — the Colab notebook these scripts derive from
- `docs/benchmarks.md` — benchmark triage and integration status
