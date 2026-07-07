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

import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console

from ragtune.adapters.langchain import LangChainRetriever
from ragtune.data.loaders.OBLIQLoader import OBLIQLoader, OBLIQ_TASKS
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.estimators import BaselineEstimator, SimilarityEstimator
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.core.budget import CostBudget
from ragtune.core.controller import RAGtuneController
from ragtune.evaluation.RetrievalEvaluator import RetrievalEvaluator
from ragtune.utils.config import config

_evaluator = RetrievalEvaluator(k_values=[10, 50])

_console = Console()
def print_header(msg): _console.print(f"[bold blue]{msg}[/bold blue]")
def print_step(msg):   _console.print(f"[dim]{msg}[/dim]")
def print_success(msg): _console.print(f"[bold green]{msg}[/bold green]")

# --- Configuration ---

TASKS: List[str] = os.environ.get("OBLIQ_TASKS", "congress,math").split(",")
QUERIES_PER_TASK: int = int(os.environ.get("OBLIQ_QUERIES", "20"))
CANDIDATES_TOP_K: int = 50
MAX_CORPUS_DOCS: int = 5_000
EMBED_MODEL: str = "all-MiniLM-L6-v2"


class _OracleReranker(SimulatedReranker):
    """Gold-aware oracle reranker for smoke testing. No model download required."""
    def __init__(self):
        self._gold: set = set()

    def set_gold(self, qid: str, qrels: Dict[str, Dict[str, int]]):
        self._gold = set(qrels.get(qid, {}).keys())

    def rerank(self, documents, context, strategy=None):
        return {doc.doc_id: (0.95 if doc.doc_id in self._gold else 0.3) for doc in documents}


_reranker = _OracleReranker()


# --- Data Loading ---

def load_task(task: str) -> Tuple[
    Dict[str, str],                 # corpus: {doc_id: {"text": str, "title": str}}
    Dict[str, str],                 # queries: {query_id: text}
    Dict[str, Dict[str, int]],      # qrels:   {query_id: {doc_id: score}}
    Optional[Dict[str, List[str]]], # excluded_ids per query, or None
]:
    """Loads corpus, queries, qrels, and excluded_ids via OBLIQLoader."""
    print_step(f"Loading [{task}] via OBLIQLoader...")
    loader = OBLIQLoader(
        task=task,
        max_queries=QUERIES_PER_TASK,
        max_corpus_docs=MAX_CORPUS_DOCS,
    )
    return loader.get_corpus(), loader.get_queries(), loader.get_qrels(), loader.get_excluded_ids()


# --- Index Building ---

def build_retriever(
    corpus: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
) -> Tuple[LangChainRetriever, FAISS]:
    """Builds a FAISS index over the corpus."""
    lc_docs = [
        Document(page_content=text if isinstance(text, str) else text.get("text", ""),
                 metadata={"id": doc_id})
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

def score_results(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    excluded_ids: Optional[Dict[str, List[str]]],
) -> Dict[str, float]:
    """
    Computes macro-averaged NDCG@10, Recall@10, Recall@50 via RetrievalEvaluator.
    Excluded docs (per-query mask for math/writing tasks) are removed before evaluation.
    """
    if excluded_ids:
        results = {
            qid: {did: s for did, s in doc_scores.items()
                  if did not in excluded_ids.get(qid, [])}
            for qid, doc_scores in results.items()
        }
    metrics = _evaluator.evaluate(qrels, results)
    return {
        "NDCG@10":   round(metrics["ndcg"].get("NDCG@10",    0.0), 4),
        "Recall@10": round(metrics["recall"].get("Recall@10", 0.0), 4),
        "Recall@50": round(metrics["recall"].get("Recall@50", 0.0), 4),
    }


# --- Scenario Execution ---

def run_controller_scenario(
    name: str,
    controller: RAGtuneController,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """Runs a controller over all queries. Returns (results, avg_reranked, avg_latency_ms)."""
    print_step(f"  Running [{name}]...")
    results: Dict[str, Dict[str, float]] = {}
    latencies: List[float] = []
    docs_reranked: List[float] = []

    for qid, qtext in queries.items():
        _reranker.set_gold(qid, qrels)
        t0 = time.time()
        output = controller.run(qtext)
        latencies.append((time.time() - t0) * 1000)
        docs_reranked.append(output.final_budget_state.get("rerank_docs", 0))
        results[qid] = {
            doc.id: 1.0 / (rank + 1)
            for rank, doc in enumerate(output.documents)
        }

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
            doc.metadata["id"]: 1.0 / (rank + 1)
            for rank, (doc, _) in enumerate(pairs)
        }
    return results


def build_scenarios(retriever: LangChainRetriever) -> List[Tuple[str, RAGtuneController]]:
    reranker = _reranker
    return [
        (
            "Static Rerank (budget=20)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=reranker,
                assembler=GreedyAssembler(max_docs=CANDIDATES_TOP_K),
                scheduler=ActiveLearningScheduler(batch_size=20),
                estimator=BaselineEstimator(),
                budget=CostBudget.simple(docs=20, tokens=100_000, latency=600_000),
            ),
        ),
        (
            "RAGtune (budget=10)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=reranker,
                assembler=GreedyAssembler(max_docs=CANDIDATES_TOP_K),
                scheduler=ActiveLearningScheduler(batch_size=2),
                estimator=SimilarityEstimator(),
                budget=CostBudget.simple(docs=10, tokens=100_000, latency=600_000),
            ),
        ),
        (
            "RAGtune (budget=20)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=reranker,
                assembler=GreedyAssembler(max_docs=CANDIDATES_TOP_K),
                scheduler=ActiveLearningScheduler(batch_size=5),
                estimator=SimilarityEstimator(),
                budget=CostBudget.simple(docs=20, tokens=100_000, latency=600_000),
            ),
        ),
    ]


# --- Main ---

def main():
    config.set("retrieval.original_query_depth", CANDIDATES_TOP_K)
    print_header("RAGtune × OBLIQ-Bench")
    print_step(
        f"Tasks: {TASKS}  |  Queries/task: {QUERIES_PER_TASK}"
        f"  |  Candidates: {CANDIDATES_TOP_K}  |  Corpus cap: {MAX_CORPUS_DOCS}"
    )

    all_rows: List[Dict] = []

    for task in TASKS:
        if task not in OBLIQ_TASKS:
            _console.print(
                f"[yellow]Unknown task '{task}', skipping. "
                f"Valid: {OBLIQ_TASKS}[/yellow]"
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
            ctrl_results, avg_reranked, avg_latency = run_controller_scenario(name, controller, queries, qrels)
            _record(name, ctrl_results, avg_reranked, avg_latency)

    print_header("\nFINAL RESULTS")
    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))
    print_success("\nOBLIQ-Bench benchmark complete.")


if __name__ == "__main__":
    main()
