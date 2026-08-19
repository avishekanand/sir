# ── Cell 1: Environment setup ─────────────────────────────────────────────────
# Run once per Colab session.
!apt-get install -y -q default-jdk 2>/dev/null
!pip install -q python-terrier ir-datasets litellm optuna matplotlib

import os, sys
REPO_DIR = "/content/ragtune_repo"
if not os.path.exists(REPO_DIR):
    !git clone -b feat/llm-optimizer https://github.com/avishekanand/sir.git {REPO_DIR} --quiet
os.chdir(REPO_DIR)
!git checkout feat/llm-optimizer
!git pull origin feat/llm-optimizer --quiet
!pip install -q -e ".[tuning]"
if f"{REPO_DIR}/src" not in sys.path:
    sys.path.insert(0, f"{REPO_DIR}/src")

import subprocess
java = subprocess.run(['java', '-version'], capture_output=True, text=True)
print("Java:", java.stderr.split('\n')[0])
print("Working dir:", os.getcwd())

# ── Cell 2: API key ───────────────────────────────────────────────────────────
import os
try:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("Key loaded from Colab secrets.")
except Exception:
    os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY-HERE"

# ── Cell 3: PyTerrier init ────────────────────────────────────────────────────
import pyterrier as pt
if not pt.started():
    pt.init()
print("PyTerrier", pt.__version__)

# ── Cell 4: Dataset catalogue ─────────────────────────────────────────────────
# Supported BEIR benchmarks.  Comment out any you don't want to index/run.
# Index build times (Colab T4): trec-covid ~5 min, nfcorpus ~30 s,
#   scifact ~45 s, fiqa ~3 min.
DATASETS = {
    "trec-covid": {
        "irds_id":    "irds:beir/trec-covid",
        "index_path": "./idx_trec_covid",
        "n_queries":  50,          # full set — every TREC-COVID topic
    },
    "nfcorpus": {
        "irds_id":    "irds:beir/nfcorpus/test",
        "index_path": "./idx_nfcorpus",
        "n_queries":  50,          # 323 total; 50 is a representative sample
    },
    "scifact": {
        "irds_id":    "irds:beir/scifact/test",
        "index_path": "./idx_scifact",
        "n_queries":  50,          # 300 total
    },
    "fiqa": {
        "irds_id":    "irds:beir/fiqa/test",
        "index_path": "./idx_fiqa",
        "n_queries":  50,          # 648 total
    },
}

# ── Cell 5: Build indexes (skip if already built) ─────────────────────────────
from pathlib import Path

def build_index(irds_id: str, index_path: str) -> None:
    if Path(index_path + "/data.properties").exists():
        print(f"  [{index_path}] already built — skipping")
        return
    print(f"  [{index_path}] building …")
    ds = pt.get_dataset(irds_id)
    indexer = pt.IterDictIndexer(
        index_path,
        overwrite=True,
        meta={"docno": 26, "text": 131072},
        text_attrs=["text"],
    )
    indexer.index(ds.get_corpus_iter())
    n = pt.IndexFactory.of(index_path).getCollectionStatistics().getNumberOfDocuments()
    print(f"  [{index_path}] done — {n:,} documents")

print("Building indexes …")
for name, cfg in DATASETS.items():
    print(f"\n{name}:")
    build_index(cfg["irds_id"], cfg["index_path"])
print("\nAll indexes ready.")

# ── Cell 6: Benchmark settings ────────────────────────────────────────────────
# N_EVAL_QUERIES: number of queries per optimizer iteration/trial.
# BUDGET: total iterations (LLM) / trials (Bayesian) — matched for fair comparison.
# BM25 parameters match BEIR paper Anserini defaults (k1=0.9, b=0.4) for all datasets.
N_EVAL_QUERIES = 50      # full query sample for low-variance NDCG estimates
BUDGET         = 50      # per-dataset optimizer budget
SEED           = 42
LLM_MODEL      = "gpt-4o-mini"
BM25_K1        = 0.9     # BEIR-tuned BM25 k1
BM25_B         = 0.4     # BEIR-tuned BM25 b

# Search space identical for both optimizers.
SEARCH_SPACE = {
    "reranker_types":      ["noop", "cross-encoder", "monot5"],
    "reformulator_types":  ["identity"],
    "estimator_types":     ["baseline", "utility", "similarity"],
    "scheduler_types":     ["active-learning", "graceful-degradation"],
    "feedback_types":      ["none", "budget-stop"],
}

print(f"Budget: {BUDGET} evals × {N_EVAL_QUERIES} queries "
      f"= {BUDGET * N_EVAL_QUERIES} total query runs per optimizer per dataset")
print(f"BM25: k1={BM25_K1}, b={BM25_B}  (BEIR Anserini defaults)")

# ── Cell 7: Per-dataset benchmark loop ───────────────────────────────────────
import time
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import ragtune.adapters, ragtune.components   # populate registry
from ragtune.adapters.pyterrier import PyTerrierRetriever
from ragtune.tuning.evaluator import EvalDataset
from ragtune.tuning.study_config import TuningStudyConfig, DatasetConfig
from ragtune.tuning.optimizer import run_study, extract_pareto_configs
from ragtune.tuning.llm_optimizer import (
    LLMOptimizerConfig, LLMAgentOptimizer, compute_pareto_front,
)

all_results = {}   # dataset_name → {"llm": ..., "bayes": ..., "retriever": ...}

for ds_name, ds_cfg in DATASETS.items():
    print(f"\n{'='*65}")
    print(f" Dataset: {ds_name}")
    print(f"{'='*65}")

    # ── Build retriever with BEIR-tuned BM25 parameters ───────────────────────
    # Pass pt_transformer directly so BM25 controls are explicit.
    br = pt.BatchRetrieve(
        ds_cfg["index_path"],
        wmodel="BM25",
        controls={"BM25.b": str(BM25_B), "BM25.k_1": str(BM25_K1)},
        metadata=["docno", "text"],
    )
    fixed_retriever = PyTerrierRetriever(pt_transformer=br)

    # ── Load queries + qrels ──────────────────────────────────────────────────
    full_dataset = EvalDataset.from_pyterrier_irds(
        irds_id=ds_cfg["irds_id"],
        n_queries=ds_cfg["n_queries"],
        seed=SEED,
    )
    print(f" Loaded {len(full_dataset.queries)} queries")

    # ── LLM agent optimizer ───────────────────────────────────────────────────
    llm_out_dir = f"./results_{ds_name}_llm"
    llm_cfg = LLMOptimizerConfig(
        name=f"llm-{ds_name}",
        llm_model=LLM_MODEL,
        temperature=0.7,
        n_iterations=BUDGET,
        n_eval_queries=N_EVAL_QUERIES,
        seed=SEED,
        output_dir=llm_out_dir,
        search_space_overrides=SEARCH_SPACE,
    )
    llm_opt = LLMAgentOptimizer(config=llm_cfg)
    print(f" Running LLM optimizer ({BUDGET} iterations) …")
    t0 = time.time()
    llm_history = llm_opt.run(fixed_retriever, full_dataset)
    llm_time = time.time() - t0
    llm_pareto = compute_pareto_front(llm_history)
    llm_valid  = [c for c in llm_history if not c.error]

    # ── Bayesian (Optuna TPE) optimizer ───────────────────────────────────────
    bayes_out_dir = f"./results_{ds_name}_bayes"
    bayes_cfg = TuningStudyConfig(
        name=f"bayes-{ds_name}",
        dataset=DatasetConfig(name=ds_name, irds_id=ds_cfg["irds_id"]),
        n_trials=BUDGET,
        n_startup_trials=max(5, BUDGET // 8),
        n_eval_queries=N_EVAL_QUERIES,
        seed=SEED,
        n_parallel_workers=1,
        max_mean_rerank_docs=200.0,
        max_trial_seconds=300.0,
        pareto_warmup_trials=max(5, BUDGET // 8),
        output_dir=bayes_out_dir,
        search_space_overrides=SEARCH_SPACE,
    )
    print(f" Running Bayesian optimizer ({BUDGET} trials) …")
    t1 = time.time()
    study = run_study(bayes_cfg, fixed_retriever, full_dataset)
    bayes_time = time.time() - t1
    bayes_complete = [t for t in study.trials if t.values is not None]

    all_results[ds_name] = {
        "llm_history":   llm_history,
        "llm_valid":     llm_valid,
        "llm_pareto":    llm_pareto,
        "llm_time":      llm_time,
        "study":         study,
        "bayes_complete": bayes_complete,
        "bayes_time":    bayes_time,
        "retriever":     fixed_retriever,
    }
    print(f" LLM done in {llm_time:.0f}s  |  Bayes done in {bayes_time:.0f}s")

print("\n\nAll datasets complete.")

# ── Cell 8: Summary table ─────────────────────────────────────────────────────
def _best_ndcg_llm(res):
    return max((c.ndcg_at_10 for c in res["llm_valid"]), default=0.0)

def _best_ndcg_bayes(res):
    return max((t.values[0] for t in res["bayes_complete"] if t.values), default=0.0)

def _min_cost_llm(res):
    return min((c.mean_rerank_docs for c in res["llm_valid"]), default=999.0)

def _min_cost_bayes(res):
    return min((t.values[1] for t in res["bayes_complete"] if t.values), default=999.0)

def hypervolume_2d(pairs, ref_cost=200.0):
    pts = sorted([(n, c) for n, c in pairs if c < ref_cost and n > 0], key=lambda p: p[1])
    hv, prev = 0.0, 0.0
    for ndcg, cost in pts:
        hv += ndcg * (cost - prev)
        prev = cost
    return hv

REF_COST = 200.0

print(f"\n{'='*80}")
print(f" Multi-benchmark summary  |  {BUDGET} evals × {N_EVAL_QUERIES} queries  |  BM25 k1={BM25_K1} b={BM25_B}")
print(f"{'='*80}")
print(f" {'Dataset':<14} {'LLM NDCG':>10} {'Bayes NDCG':>11} {'LLM HV':>9} {'Bayes HV':>9} {'Bayes Pareto':>13}")
print(f"{'-'*80}")

for ds_name, res in all_results.items():
    llm_ndcg   = _best_ndcg_llm(res)
    bayes_ndcg = _best_ndcg_bayes(res)
    llm_hv = hypervolume_2d(
        [(c.ndcg_at_10, c.mean_rerank_docs) for c in res["llm_pareto"]], REF_COST
    )
    bayes_hv = hypervolume_2d(
        [(t.values[0], t.values[1]) for t in res["study"].best_trials if t.values], REF_COST
    )
    n_pareto_bayes = len(res["study"].best_trials)
    win_marker = " ◀" if llm_ndcg > bayes_ndcg else ("  " if llm_ndcg == bayes_ndcg else "")
    print(
        f" {ds_name:<14}"
        f" {llm_ndcg:>9.4f}{win_marker:<2}"
        f" {bayes_ndcg:>11.4f}"
        f" {llm_hv:>9.2f}"
        f" {bayes_hv:>9.2f}"
        f" {n_pareto_bayes:>13d}"
    )

print(f"{'='*80}")
print("◀ = LLM agent wins on best NDCG@10 for this dataset")

# ── Cell 9: Convergence plots (one row per dataset) ───────────────────────────
import matplotlib.pyplot as plt
import numpy as np

n_ds = len(all_results)
fig, axes = plt.subplots(n_ds, 2, figsize=(14, 4 * n_ds))
if n_ds == 1:
    axes = [axes]

def _best_so_far(values):
    best, out = 0.0, []
    for v in values:
        if v is not None:
            best = max(best, v)
        out.append(best)
    return out

for row_idx, (ds_name, res) in enumerate(all_results.items()):
    ax_pareto, ax_conv = axes[row_idx]
    llm_valid  = res["llm_valid"]
    llm_pareto = res["llm_pareto"]
    study      = res["study"]
    bayes_complete = res["bayes_complete"]

    # ── Pareto scatter ────────────────────────────────────────────────────────
    ax_pareto.scatter(
        [c.mean_rerank_docs for c in llm_valid],
        [c.ndcg_at_10 for c in llm_valid],
        alpha=0.25, color="#2196F3", s=25, label="LLM — all evals",
    )
    ax_pareto.scatter(
        [t.values[1] for t in bayes_complete],
        [t.values[0] for t in bayes_complete],
        alpha=0.25, color="#F44336", s=25, marker="s", label="Bayes — all evals",
    )

    def _pareto_step(points_xy, ax, color, label):
        if not points_xy:
            return
        pts = sorted(points_xy, key=lambda p: p[0])
        xs, ys = zip(*pts)
        ax.step(xs, ys, where="pre", color=color, linewidth=2.2, alpha=0.85)
        ax.scatter(xs, ys, color=color, s=90, zorder=6, label=label,
                   edgecolors="white", linewidths=0.8)

    _pareto_step(
        [(c.mean_rerank_docs, c.ndcg_at_10) for c in llm_pareto],
        ax_pareto, "#1565C0", "LLM — Pareto",
    )
    _pareto_step(
        [(t.values[1], t.values[0]) for t in study.best_trials if t.values],
        ax_pareto, "#B71C1C", "Bayes — Pareto",
    )
    ax_pareto.set_xlabel("Mean rerank docs (cost ↓)", fontsize=10)
    ax_pareto.set_ylabel("NDCG@10 (quality ↑)", fontsize=10)
    ax_pareto.set_title(f"{ds_name} — Pareto Front", fontsize=11, fontweight="bold")
    ax_pareto.legend(fontsize=8)
    ax_pareto.grid(True, alpha=0.3)

    # ── Convergence ───────────────────────────────────────────────────────────
    llm_curve = _best_so_far(
        [c.ndcg_at_10 if not c.error else None for c in res["llm_history"]]
    )
    bayes_curve = _best_so_far([
        t.values[0] if t.values else None
        for t in sorted(study.trials, key=lambda x: x.number)
    ])
    iters = range(1, BUDGET + 1)
    ax_conv.plot(iters, llm_curve,   color="#1565C0", linewidth=2.2,
                 marker="o", markersize=4, label="LLM agent")
    ax_conv.plot(iters, bayes_curve, color="#B71C1C", linewidth=2.2,
                 marker="s", markersize=4, linestyle="--", label="Bayesian TPE")
    ax_conv.set_xlabel("Iteration / Trial", fontsize=10)
    ax_conv.set_ylabel("Best NDCG@10 so far", fontsize=10)
    ax_conv.set_title(f"{ds_name} — Convergence", fontsize=11, fontweight="bold")
    ax_conv.legend(fontsize=9)
    ax_conv.grid(True, alpha=0.3)
    ax_conv.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("multi_benchmark_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: multi_benchmark_comparison.png")

# ── Cell 10: BM25 noop baseline per dataset ───────────────────────────────────
# Shows what pure BM25 (no reranking, default depth) achieves on each dataset.
# Useful sanity check against published BEIR numbers.
print("\n=== BM25 noop baselines (sanity check against BEIR paper Table 2) ===\n")
print(f"  {'Dataset':<14} {'NDCG@10 (noop)':>16}  {'BEIR BM25 (ref)':>18}")
print(f"  {'-'*52}")

BEIR_REFERENCE = {
    "trec-covid": 0.656,
    "nfcorpus":   0.325,
    "scifact":    0.665,
    "fiqa":       0.236,
}

for ds_name, res in all_results.items():
    # Find the best noop candidate (no reranking)
    noop_ndcg = max(
        (c.ndcg_at_10 for c in res["llm_valid"]
         if c.params.get("reranker_type") == "noop"),
        default=None,
    )
    if noop_ndcg is None:
        noop_ndcg = max(
            (t.values[0] for t in res["bayes_complete"]
             if t.values and t.params.get("reranker_type") == "noop"),
            default=float("nan"),
        )
    ref = BEIR_REFERENCE.get(ds_name, float("nan"))
    gap = noop_ndcg - ref if not (noop_ndcg != noop_ndcg) else float("nan")
    print(f"  {ds_name:<14} {noop_ndcg:>16.4f}  {ref:>18.3f}  (gap {gap:+.3f})")
