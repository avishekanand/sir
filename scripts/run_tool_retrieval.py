"""
RAGtune Tool & Skill Retrieval Benchmark Runner
Uses VenkteshVs new data pipeline (BaseDataLoader + RetrievalEvaluator).

Usage:
  python scripts/run_tool_retrieval.py --all
  python scripts/run_tool_retrieval.py --benchmark sra --subset toolqa
  python scripts/run_tool_retrieval.py --benchmark toolret
  python scripts/run_tool_retrieval.py --benchmark skillret
  python scripts/run_tool_retrieval.py --list
  python scripts/run_tool_retrieval.py --n-queries -1    # all queries
"""

import os, sys, time, argparse, datetime, tempfile
import numpy as np, pandas as pd, torch

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import pyterrier as pt

if not pt.java.started():
    pt.java.init()

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, _root)

from rich.console import Console
from rich.table import Table
from rich import box

from src.ragtune.data.loaders import ToolRetLoader, SkillRetLoader, SRABenchLoader
from src.ragtune.data.constants import TOOLRET_SUBSETS, SRA_BENCH_SUBSETS, Split
from src.ragtune.evaluation import RetrievalEvaluator

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.adapters.pyterrier import PyTerrierRetriever
from ragtune.components.rerankers import NoOpReranker, CrossEncoderReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import BaselineEstimator, SimilarityEstimator
from ragtune.utils.config import config

config.set("retrieval.original_query_depth", 100)
config.set("retrieval.max_pool_size", 200)

console = Console()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import re as _re


def sanitize_query(text):
    if not text:
        return ""
    text = _re.sub(r"\$[^$]*\$", "", text)
    text = _re.sub(r"\$\$[^$]*\$\$", "", text)
    text = _re.sub(r"\\[a-zA-Z]+", "", text)
    text = _re.sub(r"\\(.)", r"\1", text)
    text = _re.sub(r"\^+", " ** ", text)
    text = text.replace("{", "(").replace("}", ")")
    text = _re.sub(r"#\d+", "", text)
    text = text.replace("#", " ").replace("*", " x ")
    text = _re.sub(r"_(\d)", r"\1", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = _re.sub(r"\s+", " ", text).strip()
    return text[:2000]


ALL_EXPERIMENTS = [
    ("bm25_only", "noop", "baseline", 0, 1),
    ("crossenc_tight", "cross-encoder", "baseline", 5, 2),
    ("crossenc_medium", "cross-encoder", "baseline", 15, 5),
    ("crossenc_loose", "cross-encoder", "baseline", 30, 10),
    ("crossenc_sim_tight", "cross-encoder", "similarity", 5, 2),
    ("crossenc_sim_medium", "cross-encoder", "similarity", 15, 5),
    ("crossenc_sim_loose", "cross-encoder", "similarity", 30, 10),
]
QUICK_EXPERIMENTS = [
    ("bm25_only", "noop", "baseline", 0, 1),
    ("crossenc_loose", "cross-encoder", "baseline", 30, 10),
    ("crossenc_tight", "cross-encoder", "baseline", 5, 2),
]


def make_bm25(corpus):
    tmp = os.path.join(tempfile.mkdtemp(), "idx")
    return pt.terrier.Retriever(
        pt.IterDictIndexer(
            tmp, overwrite=True, meta={"docno": 128, "text": 4096}
        ).index([{"docno": did, "text": d["text"]} for did, d in corpus.items()]),
        wmodel="BM25",
        metadata=["docno", "text"],
        num_results=100,
    )


def make_controller(exp_name, rtype, etype, rdocs, bsize, retriever):
    reranker = NoOpReranker() if rtype == "noop" else CrossEncoderReranker()
    estimator = BaselineEstimator() if etype == "baseline" else SimilarityEstimator()
    budget = CostBudget(
        limits={
            "tokens": 100_000,
            "rerank_docs": rdocs,
            "rerank_calls": 50,
            "retrieval_calls": 5,
            "reformulations": 3,
            "latency_ms": 120_000,
        }
    )
    return RAGtuneController(
        retriever=retriever,
        reformulator=IdentityReformulator(),
        reranker=reranker,
        assembler=GreedyAssembler(max_docs=20),
        scheduler=ActiveLearningScheduler(batch_size=bsize),
        estimator=estimator,
        budget=budget,
    )


def run_experiments(queries_dict, qrels, retriever, experiments, evaluator, k=10):
    rows = []
    for exp_name, rtype, etype, rdocs, bsize in experiments:
        t0 = time.time()
        ctrl = make_controller(exp_name, rtype, etype, rdocs, bsize, retriever)
        results = {}
        for qid, text in queries_dict.items():
            clean = sanitize_query(text)
            if not clean:
                continue
            try:
                out = ctrl.run(clean)
                results[qid] = {
                    (d.id if hasattr(d, "id") else str(d)): 1.0 - i * 0.01
                    for i, d in enumerate(out.documents)
                }
            except:
                pass
        metrics = evaluator.evaluate(qrels, results, k_values=[k])
        ndcg = metrics.get("ndcg", {}).get(f"NDCG@{k}", 0)
        rows.append(
            {
                "config": exp_name,
                f"ndcg@{k}": ndcg,
                "time_s": time.time() - t0,
                "n_queries": len(results),
            }
        )
        console.print(
            f"    {exp_name:24s} NDCG@{k}={ndcg:.4f} ({time.time() - t0:.1f}s, {len(results)} queries)"
        )
    return pd.DataFrame(rows)


def run_benchmark(benchmark_key, experiments, n_queries, subsets=None):
    evaluator = RetrievalEvaluator(k_values=[10])
    if benchmark_key == "sra":
        targets = subsets or SRA_BENCH_SUBSETS
    elif benchmark_key == "toolret":
        targets = subsets or TOOLRET_SUBSETS
    else:
        targets = subsets or ["test"]
    console.print(
        f"\n[bold blue]=== {benchmark_key.upper()} ({len(targets)} subsets) ===[/bold blue]"
    )
    all_rows = []
    for i, name in enumerate(targets, 1):
        console.print(
            f"\n[bold yellow]--- [{i}/{len(targets)}] {name} ---[/bold yellow]"
        )
        t0 = time.time()
        try:
            if benchmark_key == "sra":
                loader = SRABenchLoader(dataset=name, n_queries=n_queries)
            elif benchmark_key == "toolret":
                loader = ToolRetLoader(dataset=name, n_queries=n_queries)
            else:
                loader = SkillRetLoader(dataset=name, n_queries=n_queries)
            corpus, queries, qrels = loader.load()
        except Exception as e:
            console.print(f"  [red]LOAD ERR[/red]: {e}")
            continue
        console.print(
            f"  corpus: {len(corpus)} | queries: {len(queries)} | qrels: {len(qrels)}"
        )
        if not queries:
            console.print("  [yellow]SKIP: 0 queries[/yellow]")
            continue
        bm25 = make_bm25(corpus)
        retriever = PyTerrierRetriever(bm25)
        t1 = time.time()
        bl_results = {}
        for qid, text in queries.items():
            clean = sanitize_query(text)
            if not clean:
                continue
            try:
                res = bm25.transform(pd.DataFrame([{"qid": qid, "query": clean}]))
                bl_results[qid] = {
                    row["docno"]: float(row["score"]) for _, row in res.iterrows()
                }
            except:
                pass
        bl_metrics = evaluator.evaluate(qrels, bl_results)
        bl_ndcg = bl_metrics.get("ndcg", {}).get("NDCG@10", 0)
        console.print(f"  BM25 NDCG@10={bl_ndcg:.4f} ({time.time() - t1:.1f}s)")
        rt_df = run_experiments(
            queries,
            qrels,
            retriever,
            [e for e in experiments if e[0] != "bm25_only"],
            evaluator,
        )
        bl_row = pd.DataFrame(
            [
                {
                    "config": "bm25_only",
                    "ndcg@10": bl_ndcg,
                    "time_s": time.time() - t1,
                    "n_queries": len(queries),
                }
            ]
        )
        full = pd.concat([bl_row, rt_df], ignore_index=True)
        full["subset"] = name
        all_rows.append(full)
        console.print(f"  [{name}] done in {time.time() - t0:.0f}s")
    if not all_rows:
        return None
    full = pd.concat(all_rows, ignore_index=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    path = f"results/benchmark_{benchmark_key}_full_{ts}.csv"
    full.to_csv(path, index=False)
    console.print(f"[bold green]Saved: {path}[/bold green]")
    return full


BENCHMARKS = {
    "toolret": TOOLRET_SUBSETS,
    "skillret": ["test"],
    "sra": SRA_BENCH_SUBSETS,
}


def main():
    parser = argparse.ArgumentParser(
        description="RAGtune Tool & Skill Retrieval Benchmarks"
    )
    parser.add_argument("--benchmark", choices=["toolret", "skillret", "sra"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--subset", nargs="+")
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    experiments = ALL_EXPERIMENTS if args.full else QUICK_EXPERIMENTS
    if args.list:
        console.print(
            "\n[bold]SRA-Bench:[/bold]",
            *[f"  {s}" for s in SRA_BENCH_SUBSETS],
            sep="\n",
        )
        console.print(
            "\n[bold]ToolRet:[/bold]", *[f"  {s}" for s in TOOLRET_SUBSETS], sep="\n"
        )
        console.print("\n[bold]SkillRet:[/bold]  test (4,997 queries)")
        return
    benchmarks = list(BENCHMARKS.keys()) if args.all else [args.benchmark]
    mode = "full (7)" if args.full else "quick (3)"
    console.print(f"[bold green]GPU: {torch.cuda.get_device_name(0)}[/bold green]")
    console.print(f"[dim]Mode: {mode} | n-queries: {args.n_queries}[/dim]")
    for bkey in benchmarks:
        run_benchmark(bkey, experiments, args.n_queries, args.subset)


if __name__ == "__main__":
    main()
