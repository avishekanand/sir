"""
OBLIQ-Bench benchmark for RAGtune.

Evaluates RAGtune against raw retrieval and static one-shot reranking on
OBLIQ-Bench — a suite of five oblique-query retrieval tasks where relevance
is latent (descriptive, analogue, and tip-of-tongue queries).

Tasks (HF config names):
    congress  — tip-of-tongue (214k docs, 254 queries)
    math      — analogue (3.7k docs, 151 queries)
    writing   — analogue (10.9k docs, 512 queries)
    twitter   — descriptive (72k docs, 281 queries)
    wildchat  — descriptive (508k docs, 40 queries)

Metrics: NDCG@10, Recall@10, Recall@50

Usage:
    python scripts/benchmark_obliq.py
    OBLIQ_TASKS=congress,math,twitter python scripts/benchmark_obliq.py
    OBLIQ_TASKS=congress OBLIQ_QUERIES=50 python scripts/benchmark_obliq.py
"""

import csv
import json
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console

from ragtune.adapters.langchain import LangChainRetriever
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.estimators import BaselineEstimator, SimilarityEstimator
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.rerankers import CrossEncoderReranker
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.core.budget import CostBudget
from ragtune.core.controller import RAGtuneController

_console = Console()
def print_header(msg): _console.print(f"[bold blue]{msg}[/bold blue]")
def print_step(msg):   _console.print(f"[dim]{msg}[/dim]")
def print_success(msg): _console.print(f"[bold green]{msg}[/bold green]")

# --- Configuration ---

DATASET_ID = "dianetc/OBLIQ-Bench"

# Maps HF config name → paths inside the repo for qrels and optional excluded_ids
TASK_META: Dict[str, Dict] = {
    "congress": {
        "qrels_path": "tip-of-tongue/congress/queries+qrels/qrels.tsv",
    },
    "math": {
        "qrels_path": "analogues/math/queries+qrels/qrels.tsv",
        "excluded_ids_path": "analogues/math/queries+qrels/per_query_excluded_ids.json",
    },
    "writing": {
        "qrels_path": "analogues/writing/queries+qrels/qrels.tsv",
        "excluded_ids_path": "analogues/writing/queries+qrels/per_query_excluded_ids.json",
    },
    "twitter": {
        "qrels_path": "descriptive/twitter/queries+qrels/qrels.tsv",
    },
    "wildchat": {
        "qrels_path": "descriptive/wildchat/queries+qrels/qrels.tsv",
    },
}

TASKS: List[str] = os.environ.get("OBLIQ_TASKS", "congress,math").split(",")
QUERIES_PER_TASK: int = int(os.environ.get("OBLIQ_QUERIES", "20"))
CANDIDATES_TOP_K: int = 50
MAX_CORPUS_DOCS: int = 5_000
EMBED_MODEL: str = "all-MiniLM-L6-v2"
CROSS_ENCODER: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --- Data Loading ---

def _parse_qrels(path: str, query_filter: Set[str]) -> Dict[str, Dict[str, int]]:
    """Parses a TREC-format qrels.tsv, keeping only queries in query_filter."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3 or row[0] == "query_id":
                continue
            qid, did, score = row[0], row[1], int(row[2])
            if qid in query_filter and score > 0:
                qrels.setdefault(qid, {})[did] = score
    return qrels


def load_task(task: str) -> Tuple[
    Dict[str, str],                 # corpus: {doc_id: text}
    Dict[str, str],                 # queries: {query_id: text}
    Dict[str, Dict[str, int]],      # qrels:   {query_id: {doc_id: score}}
    Optional[Dict[str, List[str]]], # excluded_ids per query, or None
]:
    """Loads corpus, queries, qrels, and optional excluded_ids for a task."""
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "Required packages missing. Install with:\n"
            "  pip install datasets huggingface_hub\n"
            "Or: pip install -e '.[benchmarks]'"
        )

    meta = TASK_META[task]

    # 1. Queries (small — load fully, then slice)
    print_step(f"Loading queries [{task}]...")
    queries_rows = list(load_dataset(DATASET_ID, task, split="queries"))[:QUERIES_PER_TASK]
    queries: Dict[str, str] = {row["_id"]: row["text"] for row in queries_rows}

    # 2. Qrels (file download — parse only for our query subset)
    print_step(f"Loading qrels [{task}]...")
    qrels_file = hf_hub_download(
        repo_id=DATASET_ID, repo_type="dataset", filename=meta["qrels_path"]
    )
    qrels = _parse_qrels(qrels_file, set(queries.keys()))

    # 3. Corpus (streamed — cap at MAX_CORPUS_DOCS but always include gold docs)
    gold_ids: Set[str] = {did for doc_scores in qrels.values() for did in doc_scores}
    print_step(f"Streaming corpus [{task}] (cap={MAX_CORPUS_DOCS}, gold={len(gold_ids)})...")
    corpus: Dict[str, str] = {}
    non_gold_count = 0
    corpus_ds = load_dataset(DATASET_ID, task, split="corpus", streaming=True)
    for row in corpus_ds:
        doc_id = row["_id"]
        is_gold = doc_id in gold_ids
        if is_gold or non_gold_count < MAX_CORPUS_DOCS:
            corpus[doc_id] = row["text"]
            if not is_gold:
                non_gold_count += 1
        # Stop once we've collected enough non-gold docs and all gold docs are found
        if non_gold_count >= MAX_CORPUS_DOCS and gold_ids.issubset(corpus):
            break

    # 4. Excluded IDs (math & writing only — mask these at eval time)
    excluded_ids: Optional[Dict[str, List[str]]] = None
    if "excluded_ids_path" in meta:
        excl_file = hf_hub_download(
            repo_id=DATASET_ID, repo_type="dataset", filename=meta["excluded_ids_path"]
        )
        with open(excl_file) as f:
            excluded_ids = json.load(f)

    return corpus, queries, qrels, excluded_ids


# --- Index Building ---

def build_retriever(
    corpus: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
) -> Tuple[LangChainRetriever, FAISS]:
    """Builds a FAISS index over the corpus."""
    lc_docs = [
        Document(page_content=text, metadata={"id": doc_id})
        for doc_id, text in corpus.items()
    ]
    print_step(f"Indexing {len(lc_docs)} documents...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        show_progress=True,
        encode_kwargs={"batch_size": 128},
    )
    vectorstore = FAISS.from_documents(lc_docs, embeddings)
    retriever = LangChainRetriever(
        vectorstore.as_retriever(search_kwargs={"k": CANDIDATES_TOP_K})
    )
    return retriever, vectorstore


# --- Evaluation ---

def _ndcg_at_k(ranked_ids: List[str], gold: Set[str], k: int) -> float:
    rel = [1 if did in gold else 0 for did in ranked_ids[:k]]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _recall_at_k(ranked_ids: List[str], gold: Set[str], k: int) -> float:
    return sum(1 for did in ranked_ids[:k] if did in gold) / len(gold) if gold else 0.0


def score_results(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    excluded_ids: Optional[Dict[str, List[str]]],
) -> Dict[str, float]:
    """
    Computes macro-averaged NDCG@10, Recall@10, Recall@50.
    Excluded docs (per-query mask for math/writing tasks) are removed before ranking.
    """
    ndcg10, rec10, rec50 = [], [], []
    for qid, doc_scores in results.items():
        gold = set((qrels.get(qid) or {}).keys())
        if not gold:
            continue
        masked = set(excluded_ids.get(qid, [])) if excluded_ids else set()
        ranked_ids = [
            did for did, _ in sorted(doc_scores.items(), key=lambda x: -x[1])
            if did not in masked
        ]
        ndcg10.append(_ndcg_at_k(ranked_ids, gold, 10))
        rec10.append(_recall_at_k(ranked_ids, gold, 10))
        rec50.append(_recall_at_k(ranked_ids, gold, 50))

    def _mean(vals): return round(sum(vals) / len(vals), 4) if vals else 0.0
    return {"NDCG@10": _mean(ndcg10), "Recall@10": _mean(rec10), "Recall@50": _mean(rec50)}


# --- Scenario Execution ---

def run_controller_scenario(
    name: str,
    controller: RAGtuneController,
    queries: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """Runs a controller over all queries. Returns (results, avg_reranked, avg_latency_ms)."""
    print_step(f"  Running [{name}]...")
    results: Dict[str, Dict[str, float]] = {}
    latencies: List[float] = []
    docs_reranked: List[float] = []

    for qid, qtext in queries.items():
        t0 = time.time()
        output = controller.run(qtext)
        latencies.append((time.time() - t0) * 1000)
        docs_reranked.append(output.final_budget_state.get("rerank_docs", 0))
        results[qid] = {doc.id: doc.score for doc in output.documents}

    return results, float(pd.Series(docs_reranked).mean()), float(pd.Series(latencies).mean())


def run_faiss_baseline(
    vectorstore: FAISS,
    queries: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Pure retrieval baseline — no reranking, raw FAISS cosine scores."""
    print_step("  Running [No-Rerank Baseline (FAISS)]...")
    results: Dict[str, Dict[str, float]] = {}
    for qid, qtext in queries.items():
        pairs = vectorstore.similarity_search_with_score(qtext, k=CANDIDATES_TOP_K)
        results[qid] = {
            doc.metadata["id"]: float(1.0 / (1.0 + score))
            for doc, score in pairs
        }
    return results


def build_scenarios(retriever: LangChainRetriever) -> List[Tuple[str, RAGtuneController]]:
    reranker = CrossEncoderReranker(CROSS_ENCODER)
    return [
        (
            "Static Rerank (budget=20)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=reranker,
                assembler=GreedyAssembler(),
                scheduler=ActiveLearningScheduler(batch_size=20),
                estimator=BaselineEstimator(),
                budget=CostBudget(max_reranker_docs=20),
            ),
        ),
        (
            "RAGtune (budget=10)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=reranker,
                assembler=GreedyAssembler(),
                scheduler=ActiveLearningScheduler(batch_size=2),
                estimator=SimilarityEstimator(),
                budget=CostBudget(max_reranker_docs=10),
            ),
        ),
        (
            "RAGtune (budget=20)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=reranker,
                assembler=GreedyAssembler(),
                scheduler=ActiveLearningScheduler(batch_size=5),
                estimator=SimilarityEstimator(),
                budget=CostBudget(max_reranker_docs=20),
            ),
        ),
    ]


# --- Main ---

def main():
    print_header("RAGtune × OBLIQ-Bench")
    print_step(
        f"Tasks: {TASKS}  |  Queries/task: {QUERIES_PER_TASK}"
        f"  |  Candidates: {CANDIDATES_TOP_K}  |  Corpus cap: {MAX_CORPUS_DOCS}"
    )

    all_rows: List[Dict] = []

    for task in TASKS:
        if task not in TASK_META:
            _console.print(
                f"[yellow]Unknown task '{task}', skipping. "
                f"Valid: {list(TASK_META)}[/yellow]"
            )
            continue

        print_header(f"\n── Task: {task} ──")
        corpus, queries, qrels, excluded_ids = load_task(task)
        n_qrels = sum(len(v) for v in qrels.values())
        print_step(f"Loaded {len(corpus)} corpus docs, {len(queries)} queries, {n_qrels} qrel pairs")

        retriever, vectorstore = build_retriever(corpus, qrels)

        def _record(scenario_name: str, results: Dict, avg_reranked: float = 0, avg_latency: float = 0):
            metrics = score_results(results, qrels, excluded_ids)
            all_rows.append({
                "task": task,
                "scenario": scenario_name,
                **metrics,
                "Avg Rerank Docs": round(avg_reranked, 1),
                "Avg Latency (ms)": round(avg_latency, 1),
            })

        faiss_results = run_faiss_baseline(vectorstore, queries)
        _record("No-Rerank (FAISS)", faiss_results)

        for name, controller in build_scenarios(retriever):
            ctrl_results, avg_reranked, avg_latency = run_controller_scenario(name, controller, queries)
            _record(name, ctrl_results, avg_reranked, avg_latency)

    print_header("\nFINAL RESULTS")
    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))
    print_success("\nOBLIQ-Bench benchmark complete.")


if __name__ == "__main__":
    main()
