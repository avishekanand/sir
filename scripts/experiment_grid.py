"""
RAGtune Experiment Grid
=======================
Systematic ablation study: quality (NDCG@5, Recall@5, MRR) vs latency
across budget levels, estimators, feedback policies, and rerankers.

Datasets (via ir_datasets + PyTerrier BM25):
  - beir/trec-covid   (50K docs, 20 queries, graded qrels)
  - beir/nfcorpus     (~3.6K docs, 50 queries, graded qrels)
  - beir/scifact      (~5K docs, 50 queries, binary qrels)

Usage:
  python scripts/experiment_grid.py                   # Groups A-D, no external services
  python scripts/experiment_grid.py --with-ollama     # + Group E (Ollama required)
  python scripts/experiment_grid.py --with-llm        # + Group F (LLM for reformulations)
  python scripts/experiment_grid.py --dataset trec-covid  # single dataset
"""

import os
import sys
import time
import argparse
import datetime
import numpy as np
import pandas as pd
import ir_datasets
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import pyterrier as pt
if not pt.started():
    pt.init()

from rich.console import Console
from rich.table import Table
from rich import box

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, os.path.dirname(__file__))

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.core.types import ScoredDocument
from ragtune.adapters.pyterrier import PyTerrierRetriever
from ragtune.components.rerankers import (
    NoOpReranker, MonoT5Reranker, OllamaListwiseReranker,
)
from ragtune.components.reformulators import IdentityReformulator, ReformIRReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import (
    BaselineEstimator, SimilarityEstimator, ReformIREstimator,
)
from ragtune.components.feedback import (
    BudgetStopFeedback, ReformIRConvergenceFeedback,
)
from ragtune.utils.config import config

# Override retrieval depth so experiments have a meaningful candidate pool.
# The controller uses these config values; 50 gives a good pool size for
# benchmarking without being too expensive.
config.set("retrieval.original_query_depth", 50)
config.set("retrieval.max_pool_size", 100)

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name: str
    ir_id: str          # ir_datasets identifier
    doc_cap: int        # max docs to index (0 = all)
    n_queries: int      # how many queries to evaluate
    graded: bool        # True = graded NDCG, False = binary

DATASETS = [
    DatasetConfig("trec-covid", "beir/trec-covid",       50_000, 20, graded=True),
    DatasetConfig("nfcorpus",   "beir/nfcorpus/test",         0, 50, graded=True),
    DatasetConfig("scifact",    "beir/scifact/test",           0, 50, graded=False),
]

# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    group: str
    name: str
    reranker: str           # "noop" | "monot5" | "ollama"
    reformulator: str       # "identity" | "reformir"
    estimator: str          # "baseline" | "similarity" | "reformir"
    feedback: Optional[str] # None | "budget-stop" | "convergence-tight" | "convergence-loose"
    rerank_docs: int
    batch_size: int
    requires_ollama: bool = False
    requires_llm: bool = False

EXPERIMENTS: List[ExperimentConfig] = [
    # ── Group A: BM25 baseline ────────────────────────────────────────────────
    ExperimentConfig("A", "bm25_only",          "noop",   "identity", "baseline",   None,                0,  1),

    # ── Group B: Budget ablation (MonoT5) ────────────────────────────────────
    ExperimentConfig("B", "monot5_tight",        "monot5", "identity", "baseline",   None,                5,  2),
    ExperimentConfig("B", "monot5_medium",       "monot5", "identity", "baseline",   None,               15,  5),
    ExperimentConfig("B", "monot5_loose",        "monot5", "identity", "baseline",   None,               30, 10),

    # ── Group C: Estimator ablation (medium budget) ──────────────────────────
    ExperimentConfig("C", "baseline_est",        "monot5", "identity", "baseline",   None,               15,  5),
    ExperimentConfig("C", "similarity_est",      "monot5", "identity", "similarity", None,               15,  5),
    ExperimentConfig("C", "reformir_est",        "monot5", "identity", "reformir",   None,               15,  5),

    # ── Group D: Feedback ablation (ReformIR estimator, loose budget) ────────
    ExperimentConfig("D", "no_feedback",         "monot5", "identity", "reformir",   None,               30,  5),
    ExperimentConfig("D", "convergence_tight",   "monot5", "identity", "reformir",   "convergence-tight", 30,  5),
    ExperimentConfig("D", "convergence_loose",   "monot5", "identity", "reformir",   "convergence-loose", 30,  5),

    # ── Group E: Ollama reranker (requires Ollama) ───────────────────────────
    ExperimentConfig("E", "ollama_medium",       "ollama", "identity", "baseline",   None,                5,  2,
                     requires_ollama=True),

    # ── Group F: Full ReformIR pipeline (requires LLM for reformulations) ────
    ExperimentConfig("F", "baseline_pipeline",   "monot5", "identity", "baseline",   None,               15,  5,
                     requires_llm=True),
    ExperimentConfig("F", "reformir_pipeline",   "monot5", "reformir", "reformir",   "convergence-tight", 30,  5,
                     requires_llm=True),
]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & indexing
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(cfg: DatasetConfig):
    """Returns (docs_for_index, queries, qrels_dict).

    qrels_dict: {(query_id, doc_id): relevance}
    docs_for_index: generator of {"docno": ..., "text": ...}
    """
    console.print(f"[dim]  Loading {cfg.ir_id} …[/dim]")
    ds = ir_datasets.load(cfg.ir_id)

    # Qrels
    qrels: Dict[Tuple[str, str], int] = {}
    for qr in ds.qrels_iter():
        qrels[(qr.query_id, qr.doc_id)] = qr.relevance

    # Queries (only those with at least one relevant document)
    queries = []
    for q in ds.queries_iter():
        has_rel = any(rel > 0 for (qid, _), rel in qrels.items() if qid == q.query_id)
        if has_rel:
            queries.append({"id": q.query_id, "text": q.text})
        if len(queries) >= cfg.n_queries:
            break

    # Corpus (generator, capped)
    def doc_iter():
        for i, doc in enumerate(ds.docs_iter()):
            if cfg.doc_cap and i >= cfg.doc_cap:
                break
            text = getattr(doc, "title", "") + " " + getattr(doc, "text", getattr(doc, "abstract", ""))
            yield {"docno": doc.doc_id, "text": text.strip()}

    return doc_iter, queries, qrels


def build_or_load_index(cfg: DatasetConfig, doc_iter_fn) -> pt.BatchRetrieve:
    """Builds a BM25 index if not cached, otherwise loads from disk."""
    index_dir = os.path.join(os.path.dirname(__file__), "..", "data", "indices", cfg.name)
    index_path = os.path.abspath(index_dir)

    if os.path.exists(os.path.join(index_path, "data.properties")):
        console.print(f"[dim]  Loading cached index from {index_path}[/dim]")
        index_ref = pt.IndexFactory.of(index_path)
    else:
        os.makedirs(index_path, exist_ok=True)
        console.print(f"[dim]  Building index at {index_path} …[/dim]")
        indexer = pt.IterDictIndexer(
            index_path, overwrite=True,
            meta={"docno": 64, "text": 4096},
        )
        index_ref = indexer.index(doc_iter_fn())

    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", metadata=["docno", "text"], num_results=100)
    return bm25


# ─────────────────────────────────────────────────────────────────────────────
# Controller factory
# ─────────────────────────────────────────────────────────────────────────────

def build_controller(exp: ExperimentConfig, retriever: PyTerrierRetriever) -> RAGtuneController:
    # Reranker
    if exp.reranker == "noop":
        reranker = NoOpReranker()
    elif exp.reranker == "monot5":
        reranker = MonoT5Reranker()
    elif exp.reranker == "ollama":
        reranker = OllamaListwiseReranker(model_name="deepseek-r1:8b", base_url="http://localhost:11434")
    else:
        raise ValueError(f"Unknown reranker: {exp.reranker}")

    # Reformulator
    if exp.reformulator == "identity":
        reformulator = IdentityReformulator()
    elif exp.reformulator == "reformir":
        reformulator = ReformIRReformulator(model="ollama/deepseek-r1:8b", api_base="http://localhost:11434")
    else:
        raise ValueError(f"Unknown reformulator: {exp.reformulator}")

    # Estimator
    if exp.estimator == "baseline":
        estimator = BaselineEstimator()
    elif exp.estimator == "similarity":
        estimator = SimilarityEstimator()
    elif exp.estimator == "reformir":
        estimator = ReformIREstimator(min_reranked_for_regression=3)
    else:
        raise ValueError(f"Unknown estimator: {exp.estimator}")

    # Feedback
    feedback = None
    if exp.feedback == "budget-stop":
        feedback = BudgetStopFeedback()
    elif exp.feedback == "convergence-tight":
        feedback = ReformIRConvergenceFeedback(convergence_threshold=0.01)
    elif exp.feedback == "convergence-loose":
        feedback = ReformIRConvergenceFeedback(convergence_threshold=0.05)

    budget = CostBudget(limits={
        "tokens":          100_000,
        "rerank_docs":     exp.rerank_docs,
        "rerank_calls":    50,
        "retrieval_calls": 5,
        "reformulations":  3,
        "latency_ms":      120_000,
    })

    return RAGtuneController(
        retriever=retriever,
        reformulator=reformulator,
        reranker=reranker,
        assembler=GreedyAssembler(max_docs=20),
        scheduler=ActiveLearningScheduler(batch_size=exp.batch_size, strategy=exp.reranker),
        estimator=estimator,
        budget=budget,
        feedback=feedback,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def graded_ndcg_at_k(documents: List[ScoredDocument], qrels: Dict, query_id: str, k: int = 5) -> float:
    """Graded NDCG@k using qrel relevance values (0/1/2/3)."""
    rels = [qrels.get((query_id, doc.id), 0) for doc in documents[:k]]
    def dcg(r): return sum((2**v - 1) / np.log2(i + 2) for i, v in enumerate(r))
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal)
    return dcg(rels) / idcg if idcg > 0 else 0.0


def recall_at_k(documents: List[ScoredDocument], qrels: Dict, query_id: str, k: int = 5) -> float:
    relevant = {did for (qid, did), rel in qrels.items() if qid == query_id and rel > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for doc in documents[:k] if doc.id in relevant)
    return min(hits / len(relevant), 1.0)


def mrr(documents: List[ScoredDocument], qrels: Dict, query_id: str) -> float:
    relevant = {did for (qid, did), rel in qrels.items() if qid == query_id and rel > 0}
    for i, doc in enumerate(documents):
        if doc.id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def count_iterations(output) -> int:
    return sum(1 for e in output.trace.events if e.action == "rerank_batch")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    exp: ExperimentConfig,
    controller: RAGtuneController,
    queries: List[Dict],
    qrels: Dict,
    graded: bool,
) -> List[Dict[str, Any]]:
    rows = []
    for q in queries:
        try:
            output = controller.run(q["text"])
        except Exception as e:
            console.print(f"[red]  ERROR query {q['id']}: {e}[/red]")
            continue

        docs = output.documents
        bs = output.final_budget_state

        ndcg = graded_ndcg_at_k(docs, qrels, q["id"]) if graded else graded_ndcg_at_k(docs, qrels, q["id"])
        rows.append({
            "group":              exp.group,
            "config":             exp.name,
            "query_id":           q["id"],
            "ndcg@5":             ndcg,
            "recall@5":           recall_at_k(docs, qrels, q["id"]),
            "mrr":                mrr(docs, qrels, q["id"]),
            "latency_ms":         bs.get("latency", 0),
            "rerank_docs_used":   bs.get("rerank_docs", 0),
            "reformulations_used":bs.get("reformulations", 0),
            "n_iterations":       count_iterations(output),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_group_table(group: str, rows: List[Dict], dataset_name: str):
    group_rows = [r for r in rows if r["group"] == group]
    if not group_rows:
        return

    df = pd.DataFrame(group_rows)
    summary = (
        df.groupby("config")
        .agg(
            ndcg5=("ndcg@5",             "mean"),
            recall5=("recall@5",          "mean"),
            mrr_=("mrr",                  "mean"),
            latency=("latency_ms",        "mean"),
            rerank_docs=("rerank_docs_used","mean"),
            iterations=("n_iterations",   "mean"),
        )
        .reset_index()
    )
    summary["efficiency"] = summary["ndcg5"] / (summary["latency"] / 1000 + 1e-9)

    labels = {
        "A": "BM25 Baseline",
        "B": "Budget Ablation",
        "C": "Estimator Ablation",
        "D": "Feedback Ablation",
        "E": "Ollama vs MonoT5",
        "F": "Full ReformIR Pipeline",
    }
    title = f"Group {group} — {labels.get(group, '')}  [{dataset_name}]"
    t = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
    t.add_column("Config",     style="cyan", no_wrap=True)
    t.add_column("NDCG@5",    justify="right", style="bold green")
    t.add_column("Recall@5",  justify="right")
    t.add_column("MRR",       justify="right")
    t.add_column("Latency ms",justify="right")
    t.add_column("Rerank docs",justify="right")
    t.add_column("Iterations", justify="right")
    t.add_column("Efficiency", justify="right", style="dim")

    for _, r in summary.iterrows():
        t.add_row(
            r["config"],
            f"{r['ndcg5']:.4f}",
            f"{r['recall5']:.4f}",
            f"{r['mrr_']:.4f}",
            f"{r['latency']:.0f}",
            f"{r['rerank_docs']:.1f}",
            f"{r['iterations']:.1f}",
            f"{r['efficiency']:.4f}",
        )
    console.print(t)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAGtune experiment grid")
    parser.add_argument("--with-ollama", action="store_true", help="Include Group E (Ollama required)")
    parser.add_argument("--with-llm",    action="store_true", help="Include Group F (LLM for reformulations)")
    parser.add_argument("--dataset",     default=None,        help="Run single dataset by name (trec-covid|nfcorpus|scifact)")
    args = parser.parse_args()

    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d.name == args.dataset]
        if not datasets:
            console.print(f"[red]Unknown dataset '{args.dataset}'. Choose from: {[d.name for d in DATASETS]}[/red]")
            sys.exit(1)

    experiments = [
        e for e in EXPERIMENTS
        if (not e.requires_ollama or args.with_ollama)
        and (not e.requires_llm or args.with_llm)
    ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    all_rows: List[Dict] = []

    for ds in datasets:
        console.print(f"\n[bold blue]═══ Dataset: {ds.name} ═══[/bold blue]")

        doc_iter_fn, queries, qrels = load_dataset(ds)
        bm25 = build_or_load_index(ds, doc_iter_fn)
        retriever = PyTerrierRetriever(bm25)

        console.print(f"[dim]  {len(queries)} queries, index ready.[/dim]")

        dataset_rows: List[Dict] = []

        for exp in experiments:
            console.print(f"\n[yellow]  ▶ {exp.group}/{exp.name}[/yellow]"
                          f"  reranker={exp.reranker} estimator={exp.estimator}"
                          f" feedback={exp.feedback} rerank_docs={exp.rerank_docs}")

            try:
                controller = build_controller(exp, retriever)
            except Exception as e:
                console.print(f"[red]    Skipping (build failed): {e}[/red]")
                continue

            rows = run_experiment(exp, controller, queries, qrels, ds.graded)
            for r in rows:
                r["dataset"] = ds.name
            dataset_rows.extend(rows)
            all_rows.extend(rows)

            if rows:
                avg_ndcg = np.mean([r["ndcg@5"] for r in rows])
                avg_lat  = np.mean([r["latency_ms"] for r in rows])
                console.print(f"    NDCG@5={avg_ndcg:.4f}  latency={avg_lat:.0f}ms  ({len(rows)} queries)")

        # Print grouped tables for this dataset
        for group in sorted({e.group for e in experiments}):
            print_group_table(group, dataset_rows, ds.name)

        # Per-dataset CSV
        if dataset_rows:
            csv_path = os.path.join(results_dir, f"experiment_grid_{ds.name}_{timestamp}.csv")
            pd.DataFrame(dataset_rows).to_csv(csv_path, index=False)
            console.print(f"[dim]  Saved: {csv_path}[/dim]")

    # Combined CSV across all datasets
    if all_rows:
        csv_path = os.path.join(results_dir, f"experiment_grid_all_{timestamp}.csv")
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        console.print(f"\n[bold green]All results saved → {csv_path}[/bold green]")


if __name__ == "__main__":
    main()
