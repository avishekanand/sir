"""
FreshStack benchmark for RAGtune.

Evaluates RAGtune against raw retrieval and static one-shot reranking
on the FreshStack technical IR benchmark (GitHub + StackOverflow corpus).

Metrics: α-NDCG@10, Coverage@20, Recall@50
Domains: configurable via DOMAINS constant (subset of langchain, yolo, angular, laravel, godot)

Usage:
    python scripts/benchmark_freshstack.py
    FRESHSTACK_DOMAINS=angular,laravel python scripts/benchmark_freshstack.py
"""

import os
import time
import pandas as pd
from typing import Dict, List, Tuple, Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rich.console import Console

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.adapters.langchain import LangChainRetriever
from ragtune.components.rerankers import CrossEncoderReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import BaselineEstimator, SimilarityEstimator

_console = Console()
def print_header(msg): _console.print(f"[bold blue]{msg}[/bold blue]")
def print_step(msg):   _console.print(f"[dim]{msg}[/dim]")
def print_success(msg): _console.print(f"[bold green]{msg}[/bold green]")

# --- Configuration ---

DOMAINS: List[str] = os.environ.get("FRESHSTACK_DOMAINS", "langchain,yolo").split(",")
QUERIES_PER_DOMAIN: int = 10
CANDIDATES_TOP_K: int = 50
MAX_CORPUS_DOCS: int = 5000
EMBED_MODEL: str = "all-MiniLM-L6-v2"
CROSS_ENCODER: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --- Data Loading ---

def load_domain(topic: str):
    """
    Returns (corpus, queries, nuggets, qrels_nuggets, qrels_query, query_to_nuggets).
    corpus follows BEIR format: {doc_id: {"text": str, "title": str}}.
    queries: {query_id: str}
    """
    try:
        from freshstack.datasets import DataLoader
    except ImportError:
        raise ImportError(
            "FreshStack is not installed. Run: pip install freshstack\n"
            "Or: pip install -e '.[benchmarks]'"
        )

    print_step(f"Loading FreshStack [{topic}] (test split)...")
    loader = DataLoader(
        queries_repo="freshstack/queries-oct-2024",
        corpus_repo="freshstack/corpus-oct-2024",
        topic=topic,
    )
    corpus, queries, nuggets = loader.load(split="test")
    qrels_nuggets, qrels_query, query_to_nuggets = loader.load_qrels(split="test")
    return corpus, queries, nuggets, qrels_nuggets, qrels_query, query_to_nuggets


def sample_queries(queries: Dict[str, Any], n: int) -> Dict[str, str]:
    """Returns up to n queries as {query_id: query_text}."""
    result = {}
    for qid, val in list(queries.items())[:n]:
        result[qid] = val["text"] if isinstance(val, dict) else val
    return result


def build_retriever(
    corpus: Dict[str, Any],
    qrels_query: Dict[str, Dict[str, int]],
    embed_model: str,
    top_k: int,
    max_docs: int,
) -> Tuple[LangChainRetriever, FAISS]:
    """
    Builds a FAISS index from the corpus, ensuring all gold docs are included.
    Returns the RAGtune-compatible retriever and the raw vectorstore.
    """
    # Collect gold doc IDs so they are never dropped by the corpus cap
    gold_ids: set = set()
    for doc_scores in qrels_query.values():
        gold_ids.update(doc_scores.keys())

    lc_docs: List[Document] = []
    included = 0
    for doc_id, doc_data in corpus.items():
        if included >= max_docs and doc_id not in gold_ids:
            continue
        text = doc_data.get("text", "")
        title = doc_data.get("title", "")
        page_content = f"{title}\n{text}".strip() if title else text
        lc_docs.append(
            Document(page_content=page_content, metadata={"id": doc_id})
        )
        included += 1

    print_step(f"Indexing {len(lc_docs)} documents (gold pool: {len(gold_ids)})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        encode_kwargs={"batch_size": 128, "show_progress_bar": True},
    )
    vectorstore = FAISS.from_documents(lc_docs, embeddings)
    retriever = LangChainRetriever(
        vectorstore.as_retriever(search_kwargs={"k": top_k})
    )
    return retriever, vectorstore


# --- Scenario Execution ---

def run_controller_scenario(
    name: str,
    controller: RAGtuneController,
    queries: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """
    Runs a controller over all queries.
    Returns (results_dict, avg_docs_reranked, avg_latency_ms).
    results_dict format: {query_id: {doc_id: score}}
    """
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
    top_k: int,
) -> Dict[str, Dict[str, float]]:
    """Pure retrieval baseline — no reranking, raw FAISS cosine scores."""
    print_step("  Running [No-Rerank Baseline (FAISS)]...")
    results: Dict[str, Dict[str, float]] = {}
    for qid, qtext in queries.items():
        pairs = vectorstore.similarity_search_with_score(qtext, k=top_k)
        results[qid] = {
            doc.metadata["id"]: float(1.0 / (1.0 + score))
            for doc, score in pairs
        }
    return results


# --- Evaluation ---

def evaluate(
    results: Dict[str, Dict[str, float]],
    qrels_nuggets: Dict,
    qrels_query: Dict,
    query_to_nuggets: Dict,
) -> Tuple[float, float, float]:
    """
    Evaluates results with FreshStack metrics.
    Returns (alpha_ndcg_at_10, coverage_at_20, recall_at_50).
    """
    from freshstack.retrieval.evaluation import EvaluateRetrieval

    evaluator = EvaluateRetrieval(k_values=[10, 20, 50])
    alpha_ndcg, coverage, recall = evaluator.evaluate(
        qrels_nuggets=qrels_nuggets,
        query_to_nuggets=query_to_nuggets,
        qrels_query=qrels_query,
        results=results,
    )

    def _extract(d, key):
        if isinstance(d, dict):
            for k, v in d.items():
                if str(key) in str(k):
                    return float(v)
            return float(next(iter(d.values())))
        return float(d)

    return _extract(alpha_ndcg, 10), _extract(coverage, 20), _extract(recall, 50)


# --- Main ---

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


def main():
    print_header("RAGtune × FreshStack Benchmark")
    print_step(f"Domains: {DOMAINS}  |  Queries/domain: {QUERIES_PER_DOMAIN}  |  Candidates: {CANDIDATES_TOP_K}")

    all_rows: List[Dict] = []

    for topic in DOMAINS:
        print_header(f"\n── Domain: {topic} ──")

        corpus, queries_raw, nuggets, qrels_nuggets, qrels_query, query_to_nuggets = load_domain(topic)
        queries = sample_queries(queries_raw, QUERIES_PER_DOMAIN)
        print_step(f"Loaded {len(corpus)} corpus docs, {len(queries)} queries")

        retriever, vectorstore = build_retriever(
            corpus, qrels_query, EMBED_MODEL, CANDIDATES_TOP_K, MAX_CORPUS_DOCS
        )

        # No-rerank baseline (pure FAISS)
        faiss_results = run_faiss_baseline(vectorstore, queries, CANDIDATES_TOP_K)
        ndcg, cov, rec = evaluate(faiss_results, qrels_nuggets, qrels_query, query_to_nuggets)
        all_rows.append({
            "domain": topic,
            "scenario": "No-Rerank (FAISS)",
            "α-NDCG@10": round(ndcg, 4),
            "Coverage@20": round(cov, 4),
            "Recall@50": round(rec, 4),
            "Avg Rerank Docs": 0,
            "Avg Latency (ms)": 0,
        })

        # Controller scenarios
        for name, controller in build_scenarios(retriever):
            ctrl_results, avg_reranked, avg_latency = run_controller_scenario(
                name, controller, queries
            )
            ndcg, cov, rec = evaluate(
                ctrl_results, qrels_nuggets, qrels_query, query_to_nuggets
            )
            all_rows.append({
                "domain": topic,
                "scenario": name,
                "α-NDCG@10": round(ndcg, 4),
                "Coverage@20": round(cov, 4),
                "Recall@50": round(rec, 4),
                "Avg Rerank Docs": round(avg_reranked, 1),
                "Avg Latency (ms)": round(avg_latency, 1),
            })

    print_header("\nFINAL RESULTS")
    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))
    print_success("\nFreshStack benchmark complete.")


if __name__ == "__main__":
    main()
