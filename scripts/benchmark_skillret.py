"""
SKILLRET benchmark for RAGtune.

Evaluates RAGtune against raw retrieval and static one-shot reranking on
SKILLRET — a skill-retrieval benchmark that matches natural-language user
requests to agent skills sourced from GitHub.

The corpus is 6,660 skills (test split) represented as their name +
description. The full skill_md Markdown body is intentionally omitted
to stay within embedding model token limits while preserving semantics.

Dataset: ThakiCloud/SKILLRET
  skills/test   — 6,660 skills (the documents to retrieve)
  queries/test  — 4,997 evaluation queries
  qrels/test    — 8,347 binary relevance pairs

Metrics: NDCG@10, Recall@10, Recall@50

Usage:
    python scripts/benchmark_skillret.py
    SKILLRET_QUERIES=100 python scripts/benchmark_skillret.py
"""

import os
import time
from typing import Dict, List, Tuple

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console

from ragtune.adapters.langchain import LangChainRetriever
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.estimators import BaselineEstimator, SimilarityEstimator
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.core.budget import CostBudget
from ragtune.core.controller import RAGtuneController
from ragtune.evaluation.RetrievalEvaluator import RetrievalEvaluator

_console = Console()
def print_header(msg): _console.print(f"[bold blue]{msg}[/bold blue]")
def print_step(msg):   _console.print(f"[dim]{msg}[/dim]")
def print_success(msg): _console.print(f"[bold green]{msg}[/bold green]")

_evaluator = RetrievalEvaluator(k_values=[10, 50])

# --- Configuration ---

DATASET_ID = "ThakiCloud/SKILLRET"
QUERIES_PER_RUN: int = int(os.environ.get("SKILLRET_QUERIES", "20"))
CANDIDATES_TOP_K: int = 50
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

def load_data() -> Tuple[
    Dict[str, str],             # corpus:  {skill_id: text}
    Dict[str, str],             # queries: {query_id: query_text}
    Dict[str, Dict[str, int]],  # qrels:   {query_id: {skill_id: relevance}}
]:
    """
    Loads skills, evaluation queries, and qrels from SKILLRET (test splits).
    The full corpus (6,660 skills) is loaded without streaming since it fits
    comfortably in memory.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Required package missing. Install with:\n"
            "  pip install datasets\n"
            "Or: pip install -e '.[benchmarks]'"
        )

    # 1. Skills — full corpus, indexed as "name + description"
    print_step("Loading skills corpus (test split)...")
    skills_ds = load_dataset(DATASET_ID, "skills", split="test")
    corpus: Dict[str, str] = {
        row["id"]: f"{row['name']}\n\n{row['description']}"
        for row in skills_ds
    }

    # 2. Evaluation queries — capped at QUERIES_PER_RUN
    # streaming=True avoids a schema-cast error caused by train.jsonl having an
    # extra 'original_query' column absent from test.jsonl; streaming reads the
    # target split lazily without unifying schemas across all files in the config.
    print_step(f"Loading evaluation queries (first {QUERIES_PER_RUN})...")
    query_rows = []
    for row in load_dataset(DATASET_ID, "queries", split="test", streaming=True):
        query_rows.append(row)
        if len(query_rows) >= QUERIES_PER_RUN:
            break
    queries: Dict[str, str] = {row["id"]: row["query"] for row in query_rows}

    # 3. Qrels — load full test split, filter to our query subset
    print_step("Loading qrels (test split)...")
    query_ids = set(queries.keys())
    qrels: Dict[str, Dict[str, int]] = {}
    for row in load_dataset(DATASET_ID, "qrels", split="test"):
        qid = row["query_id"]
        if qid in query_ids and row["relevance"] > 0:
            qrels.setdefault(qid, {})[row["skill_id"]] = int(row["relevance"])

    return corpus, queries, qrels


# --- Index Building ---

def build_retriever(corpus: Dict[str, str]) -> Tuple[LangChainRetriever, FAISS]:
    """Builds a FAISS index over all skills."""
    lc_docs = [
        Document(page_content=text, metadata={"id": skill_id})
        for skill_id, text in corpus.items()
    ]
    print_step(f"Indexing {len(lc_docs)} skills...")
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
) -> Dict[str, float]:
    """Computes macro-averaged NDCG@10, Recall@10, Recall@50 via RetrievalEvaluator."""
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
    print_header("RAGtune × SKILLRET")
    print_step(f"Queries: {QUERIES_PER_RUN}  |  Candidates: {CANDIDATES_TOP_K}")

    corpus, queries, qrels = load_data()
    n_qrels = sum(len(v) for v in qrels.values())
    print_step(f"Loaded {len(corpus)} skills, {len(queries)} queries, {n_qrels} qrel pairs")

    retriever, vectorstore = build_retriever(corpus)

    all_rows: List[Dict] = []

    def _record(scenario_name: str, results: Dict, avg_reranked: float = 0, avg_latency: float = 0):
        metrics = score_results(results, qrels)
        all_rows.append({
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
    print_success("\nSKILLRET benchmark complete.")


if __name__ == "__main__":
    main()
