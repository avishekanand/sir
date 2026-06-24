"""
CRUMB passage-retrieval benchmark for RAGtune.

Evaluates RAGtune against raw retrieval and static one-shot reranking on
CRUMB (Complex Retrieval Unified Multi-task Benchmark), passage-retrieval
mode only. Document-level retrieval tasks are excluded by design.

Tasks (HF split names within the passage_corpus / evaluation_queries configs):
    paper_retrieval                  (~72 queries)
    theorem_retrieval                (~69 queries)
    tip_of_the_tongue                (~135 queries)
    stack_exchange                   (~107 queries)
    clinical_trial                   (~113 queries)
    set_operation_entity_retrieval   (~423 queries)
    code_retrieval                   (~3.7k queries)
    legal_qa                         (~6.75k queries)

Metrics: NDCG@10, Recall@10, Recall@50 (passage-level, using passage_qrels)

Usage:
    python scripts/benchmark_crumb.py
    CRUMB_TASKS=paper_retrieval,theorem_retrieval python scripts/benchmark_crumb.py
    CRUMB_TASKS=clinical_trial CRUMB_QUERIES=30 python scripts/benchmark_crumb.py
"""

import os
import time
from typing import Dict, List, Set, Tuple

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console

from ragtune.adapters.langchain import LangChainRetriever
from ragtune.evaluation.RetrievalEvaluator import RetrievalEvaluator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.estimators import BaselineEstimator, SimilarityEstimator
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.core.budget import CostBudget
from ragtune.core.controller import RAGtuneController
from ragtune.utils.config import config

_console = Console()
def print_header(msg): _console.print(f"[bold blue]{msg}[/bold blue]")
def print_step(msg):   _console.print(f"[dim]{msg}[/dim]")
def print_success(msg): _console.print(f"[bold green]{msg}[/bold green]")

_evaluator = RetrievalEvaluator(k_values=[10, 50])

# --- Configuration ---

DATASET_ID = "jfkback/crumb"

ALL_TASKS = [
    "paper_retrieval",
    "theorem_retrieval",
    "tip_of_the_tongue",
    "stack_exchange",
    "clinical_trial",
    "set_operation_entity_retrieval",
    "code_retrieval",
    "legal_qa",
]

TASKS: List[str] = os.environ.get("CRUMB_TASKS", "paper_retrieval,theorem_retrieval,stack_exchange").split(",")
QUERIES_PER_TASK: int = int(os.environ.get("CRUMB_QUERIES", "20"))
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
    Dict[str, str],             # corpus: {passage_id: text}
    Dict[str, str],             # queries: {query_id: query_text}
    Dict[str, Dict[str, int]],  # qrels:   {query_id: {passage_id: label}}
]:
    """
    Loads passage corpus, queries, and passage-level qrels for a CRUMB task.
    Qrels are embedded in the query rows (passage_qrels field) — no separate file needed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Required package missing. Install with:\n"
            "  pip install datasets\n"
            "Or: pip install -e '.[benchmarks]'"
        )

    # 1. Queries + embedded passage qrels
    print_step(f"Loading queries [{task}]...")
    query_rows = list(load_dataset(DATASET_ID, "evaluation_queries", split=task))[:QUERIES_PER_TASK]

    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}
    for row in query_rows:
        qid = row["query_id"]
        queries[qid] = row["query_content"]
        pqrels = row.get("passage_qrels") or []
        qrels[qid] = {
            entry["id"]: int(entry["label"])
            for entry in pqrels
            if int(entry["label"]) > 0
        }

    # 2. Passage corpus — streamed, capped at MAX_CORPUS_DOCS but gold passages always included
    gold_ids: Set[str] = {pid for doc_scores in qrels.values() for pid in doc_scores}
    print_step(f"Streaming passage corpus [{task}] (cap={MAX_CORPUS_DOCS}, gold={len(gold_ids)})...")

    corpus: Dict[str, str] = {}
    non_gold_count = 0
    corpus_ds = load_dataset(DATASET_ID, "passage_corpus", split=task, streaming=True)
    for row in corpus_ds:
        pid = row["document_id"]
        is_gold = pid in gold_ids
        if is_gold or non_gold_count < MAX_CORPUS_DOCS:
            corpus[pid] = row["document_content"]
            if not is_gold:
                non_gold_count += 1
        if non_gold_count >= MAX_CORPUS_DOCS and gold_ids.issubset(corpus):
            break

    return corpus, queries, qrels


# --- Index Building ---

def build_retriever(
    corpus: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
) -> Tuple[LangChainRetriever, FAISS]:
    """Builds a FAISS index over the passage corpus."""
    lc_docs = [
        Document(page_content=text, metadata={"id": pid})
        for pid, text in corpus.items()
    ]
    print_step(f"Indexing {len(lc_docs)} passages...")
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
    """Computes macro-averaged NDCG@10, Recall@10, Recall@50 at passage level."""
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
    print_header("RAGtune × CRUMB (Passage Retrieval)")
    print_step(
        f"Tasks: {TASKS}  |  Queries/task: {QUERIES_PER_TASK}"
        f"  |  Candidates: {CANDIDATES_TOP_K}  |  Corpus cap: {MAX_CORPUS_DOCS}"
    )

    all_rows: List[Dict] = []

    for task in TASKS:
        if task not in ALL_TASKS:
            _console.print(
                f"[yellow]Unknown task '{task}', skipping. "
                f"Valid: {ALL_TASKS}[/yellow]"
            )
            continue

        print_header(f"\n── Task: {task} ──")
        corpus, queries, qrels = load_task(task)
        n_qrels = sum(len(v) for v in qrels.values())
        print_step(f"Loaded {len(corpus)} passages, {len(queries)} queries, {n_qrels} qrel pairs")

        retriever, vectorstore = build_retriever(corpus, qrels)

        def _record(scenario_name: str, results: Dict, avg_reranked: float = 0, avg_latency: float = 0):
            metrics = score_results(results, qrels)
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
    print_success("\nCRUMB passage-retrieval benchmark complete.")


if __name__ == "__main__":
    main()
