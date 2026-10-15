"""
RAGtune Specialized Benchmark Runner
=====================================
Runs benchmarks for datasets requiring non-BEIR loaders:
- FEVER (5.4M full corpus, ir_datasets but too large for experiment_grid)
- CQADupStack (12 StackExchange subsets)
- FreshStack (nugget-based, HuggingFace)
- CRUMB (multi-task, HuggingFace)
- OBLIQ-Bench (latent queries, HuggingFace)
- CRAG (QA benchmark, HuggingFace)

All BEIR datasets are handled by experiment_grid.py.
This script exists because these datasets use different data formats
(HuggingFace datasets, multi-split structures, nugget-based relevance)
that don't fit the DatasetConfig/ir_datasets pattern in experiment_grid.py.

Usage:
  python scripts/run_benchmark.py --dataset fever
  python scripts/run_benchmark.py --dataset freshstack
  python scripts/run_benchmark.py --dataset crumb
  python scripts/run_benchmark.py --dataset obliq
  python scripts/run_benchmark.py --dataset cqadupstack
  python scripts/run_benchmark.py --dataset crag
  python scripts/run_benchmark.py --all
"""

import os
import sys
import time
import argparse
import datetime
import tempfile
import numpy as np
import pandas as pd
import torch

import pyterrier as pt

if not pt.java.started():
    pt.java.init()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from rich.console import Console
from rich.table import Table
from rich import box

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.adapters.pyterrier import PyTerrierRetriever
from ragtune.components.rerankers import NoOpReranker, CrossEncoderReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import BaselineEstimator, SimilarityEstimator
from ragtune.utils.config import config
from benchmark_utils import (
    ndcg_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k_from_ids,
    recall_at_k_from_ids,
)

config.set("retrieval.original_query_depth", 100)
config.set("retrieval.max_pool_size", 200)

console = Console()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENTS = [
    ("bm25_only", "noop", "baseline", 0, 1),
    ("crossenc_tight", "cross-encoder", "baseline", 5, 2),
    ("crossenc_medium", "cross-encoder", "baseline", 15, 5),
    ("crossenc_loose", "cross-encoder", "baseline", 30, 10),
    ("crossenc_sim_tight", "cross-encoder", "similarity", 5, 2),
    ("crossenc_sim_medium", "cross-encoder", "similarity", 15, 5),
    ("crossenc_sim_loose", "cross-encoder", "similarity", 30, 10),
]


# ── Helpers ───────────────────────────────────────────────────────────────────


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


def run_all_experiments(queries, qrels, retriever):
    rows = []
    for exp_name, rtype, etype, rdocs, bsize in EXPERIMENTS:
        ctrl = make_controller(exp_name, rtype, etype, rdocs, bsize, retriever)
        for q in queries:
            try:
                out = ctrl.run(q["text"])
                bs = out.final_budget_state
                rows.append(
                    {
                        "config": exp_name,
                        "query_id": q["id"],
                        "ndcg@5": ndcg_at_k(out.documents, qrels, q["id"]),
                        "recall@5": recall_at_k(out.documents, qrels, q["id"]),
                        "mrr": mrr(out.documents, qrels, q["id"]),
                        "latency_ms": bs.get("latency", 0),
                        "rerank_docs": bs.get("rerank_docs", 0),
                    }
                )
            except Exception as e:
                console.print(f"  [red]ERR {q['id']}[/red]: {e}")
    return rows


def save_results(df, name):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    path = f"results/benchmark_{name}_{ts}.csv"
    df.to_csv(path, index=False)
    console.print(f"[bold green]Saved → {path}[/bold green]")


# ── Dataset Loaders ───────────────────────────────────────────────────────────


def run_fever():
    """FEVER: 5.4M docs, ir_datasets. Full corpus indexing."""
    import ir_datasets

    console.print("[dim]Loading FEVER (5.4M docs)...[/dim]")
    ds = ir_datasets.load("beir/fever/test")
    qrels = {(qr.query_id, qr.doc_id): qr.relevance for qr in ds.qrels_iter()}
    relevant_qids = {qid for (qid, _), r in qrels.items() if r > 0}
    queries = [
        {"id": q.query_id, "text": q.text}
        for q in ds.queries_iter()
        if q.query_id in relevant_qids
    ][:50]
    console.print(f"  {len(queries)} queries, {len(qrels)} qrels")

    idx_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "indices", "fever_full")
    )
    if os.path.exists(os.path.join(idx_dir, "data.properties")):
        console.print("[dim]  Using cached index[/dim]")
        idx_ref = pt.IndexFactory.of(idx_dir)
    else:
        os.makedirs(idx_dir, exist_ok=True)
        console.print("[dim]  Building index (takes ~24 min)...[/dim]")
        ds2 = ir_datasets.load("beir/fever/test")
        idx_ref = pt.IterDictIndexer(
            idx_dir, overwrite=True, meta={"docno": 128, "text": 4096}
        ).index(
            {
                "docno": d.doc_id,
                "text": f"{getattr(d, 'title', '')} {getattr(d, 'text', '')}".strip(),
            }
            for d in ds2.docs_iter()
        )

    bm25 = pt.terrier.Retriever(
        idx_ref, wmodel="BM25", metadata=["docno", "text"], num_results=100
    )
    retriever = PyTerrierRetriever(bm25)
    rows = run_all_experiments(queries, qrels, retriever)
    save_results(pd.DataFrame(rows), "fever")
    return rows


def run_cqadupstack():
    """CQADupStack: 12 subsets with separate indices."""
    import ir_datasets

    subsets = [
        "beir/cqadupstack/android",
        "beir/cqadupstack/english",
        "beir/cqadupstack/gaming",
        "beir/cqadupstack/gis",
        "beir/cqadupstack/mathematica",
        "beir/cqadupstack/physics",
        "beir/cqadupstack/programmers",
        "beir/cqadupstack/stats",
        "beir/cqadupstack/tex",
        "beir/cqadupstack/unix",
        "beir/cqadupstack/webmasters",
        "beir/cqadupstack/wordpress",
    ]

    all_rows = []
    for si, subset in enumerate(subsets):
        name = subset.split("/")[-1]
        console.print(f"\n[yellow]  ▶ {si + 1}/12: {name}[/yellow]")
        ds = ir_datasets.load(subset)
        qrels = {(qr.query_id, qr.doc_id): qr.relevance for qr in ds.qrels_iter()}
        queries = [
            {"id": q.query_id, "text": q.text}
            for q in ds.queries_iter()
            if any(r > 0 for (qid, _), r in qrels.items() if qid == q.query_id)
        ][:50]

        idx_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "indices",
                f"cqadupstack_{name}",
            )
        )
        if os.path.exists(os.path.join(idx_dir, "data.properties")):
            idx_ref = pt.IndexFactory.of(idx_dir)
        else:
            os.makedirs(idx_dir, exist_ok=True)
            idx_ref = pt.IterDictIndexer(
                idx_dir,
                overwrite=True,
                meta={"docno": 64, "text": 4096},
                fields=["text"],
            ).index(
                {
                    "docno": d.doc_id,
                    "text": f"{getattr(d, 'title', '')} {getattr(d, 'text', '')}".strip(),
                }
                for d in ds.docs_iter()
            )

        bm25 = pt.terrier.Retriever(
            idx_ref, wmodel="BM25", metadata=["docno", "text"], num_results=100
        )

        for q in queries:
            try:
                res = bm25.transform(
                    pd.DataFrame([{"qid": q["id"], "query": q["text"]}])
                )
                t5 = res.sort_values("score", ascending=False)["docno"].values.tolist()[
                    :5
                ]
                ndcg = ndcg_at_k_from_ids(t5, qrels, q["id"], k=5)
                recall = recall_at_k_from_ids(t5, qrels, q["id"], k=5)
                all_rows.append(
                    {
                        "subset": name,
                        "query_id": q["id"],
                        "ndcg@5": ndcg,
                        "recall@5": recall,
                    }
                )
            except Exception as e:
                console.print(f"  [red]ERR {q['id']}[/red]: {e}")

    save_results(pd.DataFrame(all_rows), "cqadupstack")
    return all_rows


def run_freshstack():
    """FreshStack: nugget-based relevance, HuggingFace datasets."""
    from datasets import load_dataset

    domains = ["angular", "godot", "langchain", "laravel", "yolo"]
    all_rows = []

    for domain in domains:
        console.print(f"\n[yellow]  ▶ {domain}[/yellow]")
        qds = load_dataset("freshstack/queries-oct-2024", domain, split="test")
        cds = load_dataset("freshstack/corpus-oct-2024", domain, split="train")

        queries, qrels = [], {}
        for i, q in enumerate(qds):
            qid = q.get("query_id", f"q{i}")
            queries.append(
                {
                    "id": qid,
                    "text": (
                        str(q.get("query_title", "") or "")
                        + " "
                        + str(q.get("query_text", "") or "")
                    )[:500],
                }
            )
            for nugget in q.get("nuggets") or []:
                for doc_id in list(nugget.get("relevant_corpus_ids") or []):
                    qrels[(qid, str(doc_id))] = 1
            if i >= 19:
                break

        relevant_ids = {d for _, d in qrels}
        corpus, found = [], set()
        for c in cds:
            did = str(c.get("_id", "") or "")
            if did:
                corpus.append({"docno": did, "text": str(c.get("text", "") or "")})
                if did in relevant_ids:
                    found.add(did)
            if len(corpus) >= 20000 and len(found) == len(relevant_ids):
                break

        if not corpus or len(found) == 0:
            console.print(
                f"  [yellow]SKIP: {len(found)}/{len(relevant_ids)} relevant docs[/yellow]"
            )
            continue

        tmp = os.path.join(tempfile.mkdtemp(), "idx")
        bm25 = pt.terrier.Retriever(
            pt.IterDictIndexer(
                tmp, overwrite=True, meta={"docno": 128, "text": 4096}, fields=["text"]
            ).index(iter(corpus)),
            wmodel="BM25",
            metadata=["docno", "text"],
            num_results=100,
        )

        for q in queries:
            try:
                res = bm25.transform(
                    pd.DataFrame([{"qid": q["id"], "query": q["text"]}])
                )
                t10 = res.sort_values("score", ascending=False)[
                    "docno"
                ].values.tolist()[:10]
                ndcg = ndcg_at_k_from_ids(t10, qrels, q["id"], k=10)
                rel = {
                    did for (qid, did), r in qrels.items() if qid == q["id"] and r > 0
                }
                recall = (
                    min(sum(1 for d in t10 if d in rel) / len(rel), 1.0) if rel else 0.0
                )
                all_rows.append(
                    {
                        "domain": domain,
                        "query_id": q["id"],
                        "ndcg@10": ndcg,
                        "recall@10": recall,
                    }
                )
            except Exception as e:
                console.print(f"  [red]ERR {q['id']}[/red]: {e}")

    save_results(pd.DataFrame(all_rows), "freshstack")
    return all_rows


def run_crumb():
    """CRUMB: 8 multi-task domains, HuggingFace."""
    from datasets import load_dataset

    tasks = [
        "clinical_trial",
        "code_retrieval",
        "legal_qa",
        "paper_retrieval",
        "set_operation_entity_retrieval",
        "stack_exchange",
        "theorem_retrieval",
        "tip_of_the_tongue",
    ]
    queries_all = load_dataset("jfkback/crumb", "evaluation_queries", split="train")
    corpus_all = load_dataset("jfkback/crumb", "passage_corpus", split="train")
    all_rows = []

    for task in tasks:
        console.print(f"\n[yellow]  ▶ {task}[/yellow]")

        tq = []
        for item in queries_all:
            if not isinstance(item, dict):
                continue
            if item.get("task") != task:
                continue
            qid = item.get("query_id", "")
            qrels_local = {}
            for qr in item.get("passage_qrels") or []:
                if isinstance(qr, dict):
                    qrels_local[(qid, qr.get("doc_id", ""))] = qr.get("relevance", 1)
            tq.append(
                {"id": qid, "text": item.get("query_content", ""), "qrels": qrels_local}
            )
            if len(tq) >= 20:
                break
        if not tq:
            continue

        relevant_ids = {did for q in tq for _, did in q["qrels"]}
        corpus, found = [], set()
        for item in corpus_all:
            if not isinstance(item, dict) or item.get("task") != task:
                continue
            did = item.get("document_id", "")
            corpus.append({"docno": did, "text": item.get("document_content", "")})
            if did in relevant_ids:
                found.add(did)
            if len(corpus) >= 5000 and len(found) == len(relevant_ids):
                break

        if not corpus:
            console.print(f"  [yellow]SKIP[/yellow]")
            continue

        tmp = os.path.join(tempfile.mkdtemp(), "idx")
        bm25 = pt.terrier.Retriever(
            pt.IterDictIndexer(
                tmp, overwrite=True, meta={"docno": 64, "text": 4096}, fields=["text"]
            ).index(iter(corpus)),
            wmodel="BM25",
            metadata=["docno", "text"],
            num_results=100,
        )

        for q in tq:
            try:
                res = bm25.transform(
                    pd.DataFrame([{"qid": q["id"], "query": q["text"]}])
                )
                t10 = res.sort_values("score", ascending=False)[
                    "docno"
                ].values.tolist()[:10]
                ndcg = ndcg_at_k_from_ids(t10, q["qrels"], q["id"], k=10)
                rel = {
                    did
                    for (qid, did), r in q["qrels"].items()
                    if qid == q["id"] and r > 0
                }
                recall = (
                    min(sum(1 for d in t10 if d in rel) / len(rel), 1.0) if rel else 0.0
                )
                all_rows.append(
                    {
                        "task": task,
                        "query_id": q["id"],
                        "ndcg@10": ndcg,
                        "recall@10": recall,
                    }
                )
            except Exception as e:
                console.print(f"  [red]ERR {q['id']}[/red]: {e}")

    save_results(pd.DataFrame(all_rows), "crumb")
    return all_rows


def run_obliq():
    """OBLIQ-Bench: latent/tip-of-the-tongue queries. BM25 expected ~0."""
    from datasets import load_dataset

    console.print(
        "[dim]OBLIQ has oblique queries. BM25 NDCG ~0 is expected by design.[/dim]"
    )
    domains = ["congress", "math", "twitter", "wildchat", "writing"]
    all_rows = []

    for domain in domains:
        console.print(f"\n[yellow]  ▶ {domain}[/yellow]")
        qrels_ds = load_dataset(
            "mteb/OBLIQBenchRetrieval", f"{domain}-qrels", split="test"
        )
        cds = load_dataset("mteb/OBLIQBenchRetrieval", f"{domain}-corpus", split="test")
        qds = load_dataset(
            "mteb/OBLIQBenchRetrieval", f"{domain}-queries", split="test"
        )

        qrels, relevant_ids = {}, set()
        for item in qrels_ds:
            qid, did = item.get("query-id", ""), item.get("corpus-id", "")
            qrels[(qid, did)] = item.get("score", 1)
            relevant_ids.add(did)

        corpus, found = [], set()
        for item in cds:
            did = item.get("id", "")
            corpus.append({"docno": did, "text": item.get("text", "")})
            if did in relevant_ids:
                found.add(did)
            if len(corpus) >= 10000 and len(found) == len(relevant_ids):
                break

        queries = [
            {"id": item.get("query-id", f"q{i}"), "text": item.get("text", "")}
            for i, item in enumerate(qds)
            if i < 20
        ]

        if not corpus:
            continue

        tmp = os.path.join(tempfile.mkdtemp(), "idx")
        bm25 = pt.terrier.Retriever(
            pt.IterDictIndexer(
                tmp, overwrite=True, meta={"docno": 128, "text": 4096}, fields=["text"]
            ).index(iter(corpus)),
            wmodel="BM25",
            metadata=["docno", "text"],
            num_results=100,
        )

        for q in queries:
            if not q["text"]:
                continue
            try:
                res = bm25.transform(
                    pd.DataFrame([{"qid": q["id"], "query": q["text"]}])
                )
                t10 = res.sort_values("score", ascending=False)[
                    "docno"
                ].values.tolist()[:10]
                ndcg = ndcg_at_k_from_ids(t10, qrels, q["id"], k=10)
                rel = {
                    did for (qid, did), r in qrels.items() if qid == q["id"] and r > 0
                }
                recall = (
                    min(sum(1 for d in t10 if d in rel) / len(rel), 1.0) if rel else 0.0
                )
                all_rows.append(
                    {
                        "domain": domain,
                        "query_id": q["id"],
                        "ndcg@10": ndcg,
                        "recall@10": recall,
                    }
                )
            except Exception as e:
                console.print(f"  [red]ERR[/red]: {e}")

    save_results(pd.DataFrame(all_rows), "obliq")
    return all_rows


def run_crag():
    """CRAG: end-to-end QA benchmark. Reports dataset statistics."""
    from datasets import load_dataset

    configs = {
        "qapairs_open": "open",
        "qapairs_finance": "finance",
        "qapairs_sports": "sports",
        "qapairs_movie": "movie",
        "qapairs_music": "music",
    }
    t = Table(box=box.SIMPLE_HEAVY)
    t.add_column("Domain", style="cyan")
    t.add_column("Queries", justify="right")
    t.add_column("With Search Results", justify="right")
    t.add_column("With Answers", justify="right")

    all_rows = []
    for config, label in configs.items():
        ds = load_dataset("DataRobot-Research/crag", config, split="test")
        queries, sr, ans = [], 0, 0
        for item in ds:
            queries.append(item.get("query", ""))
            if item.get("search_results"):
                sr += 1
            if item.get("answer"):
                ans += 1
        t.add_row(
            label, str(len(queries)), f"{sr}/{len(queries)}", f"{ans}/{len(queries)}"
        )
        all_rows.append(
            {
                "config": config,
                "label": label,
                "queries": len(queries),
                "search_results": sr,
                "answers": ans,
            }
        )

    console.print(t)
    console.print("[dim]CRAG evaluates end-to-end QA, not retrieval NDCG.[/dim]")
    console.print("[dim]Use CRAG's own evaluation suite for full metrics.[/dim]")
    save_results(pd.DataFrame(all_rows), "crag")
    return all_rows


# ── Main ──────────────────────────────────────────────────────────────────────

DATASETS = {
    "fever": ("FEVER Full Corpus", run_fever),
    "cqadupstack": ("CQADupStack", run_cqadupstack),
    "freshstack": ("FreshStack", run_freshstack),
    "crumb": ("CRUMB", run_crumb),
    "obliq": ("OBLIQ-Bench", run_obliq),
    "crag": ("CRAG", run_crag),
}


def main():
    parser = argparse.ArgumentParser(description="RAGtune Specialized Benchmark Runner")
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()), help="Dataset to benchmark"
    )
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    args = parser.parse_args()

    if args.all:
        names = list(DATASETS.keys())
    elif args.dataset:
        names = [args.dataset]
    else:
        parser.print_help()
        return

    if DEVICE == "cuda":
        console.print(f"[bold green]GPU: {torch.cuda.get_device_name(0)}[/bold green]")

    for name in names:
        label, runner_fn = DATASETS[name]
        console.print(f"\n[bold blue]═══ {label} ═══[/bold blue]")
        runner_fn()


if __name__ == "__main__":
    main()
