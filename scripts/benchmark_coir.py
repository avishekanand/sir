"""
CoIR (Code Information Retrieval) benchmark for RAGtune.

Evaluates RAGtune against raw retrieval and static one-shot reranking on
five code-domain retrieval tasks from the CoIR-Retrieval collection on
HuggingFace. All datasets follow the BEIR format (corpus / queries / qrels
configs).

Datasets:
    stackoverflow-qa       — Stack Overflow question → answer retrieval
    codefeedback-st        — Single-turn code feedback matching
    apps                   — APPS algorithmic problem → solution retrieval
    cosqa                  — Natural-language code search (CoSQA)
    synthetic-text2sql     — Natural language → SQL retrieval

Metrics: NDCG@10, Recall@10, Recall@50

Usage:
    python scripts/benchmark_coir.py
    COIR_DATASETS=stackoverflow-qa,cosqa python scripts/benchmark_coir.py
    COIR_DATASETS=apps COIR_QUERIES=20 python scripts/benchmark_coir.py
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
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.estimators import BaselineEstimator, SimilarityEstimator
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.rerankers import CrossEncoderReranker
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.core.budget import CostBudget
from ragtune.core.controller import RAGtuneController
from ragtune.data.loaders.HuggingFaceLoader import fetch_hf_split, populate_corpus, populate_queries, populate_qrels
from ragtune.evaluation.RetrievalEvaluator import RetrievalEvaluator
from ragtune.utils.config import config

_console = Console()
def print_header(msg): _console.print(f"[bold blue]{msg}[/bold blue]")
def print_step(msg):   _console.print(f"[dim]{msg}[/dim]")
def print_success(msg): _console.print(f"[bold green]{msg}[/bold green]")

# --- Configuration ---

HF_ORG = "CoIR-Retrieval"

ALL_DATASETS = [
    "stackoverflow-qa",
    "codefeedback-st",
    "apps",
    "cosqa",
    "synthetic-text2sql",
]

DATASETS: List[str] = os.environ.get(
    "COIR_DATASETS", "stackoverflow-qa,cosqa"
).split(",")
QUERIES_PER_DATASET: int = int(os.environ.get("COIR_QUERIES", "20"))
CANDIDATES_TOP_K: int = 50
MAX_CORPUS_DOCS: int = 5_000
EMBED_MODEL: str = "all-MiniLM-L6-v2"
CROSS_ENCODER: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_evaluator = RetrievalEvaluator(k_values=[10, 50])


# --- Data Loading ---

def load_task(name: str) -> Tuple[
    Dict[str, Dict[str, str]],      # corpus: {doc_id: {"text": str, "title": str}}
    Dict[str, str],                  # queries: {query_id: query_text}
    Dict[str, Dict[str, int]],       # qrels: {query_id: {doc_id: score}}
]:
    """
    Loads corpus, queries, and qrels for a CoIR dataset in BEIR format using
    the HuggingFaceLoader module-level helpers.
    Corpus is not capped here — capping happens in build_retriever so gold docs survive.
    Queries are capped to QUERIES_PER_DATASET, keeping only those that have qrels.
    """
    dataset_id = f"{HF_ORG}/{name}"

    # CoIR datasets follow BEIR format: corpus/queries/qrels are dataset configs,
    # not splits. Corpus lives in config="corpus", split="train".
    print_step(f"Loading corpus [{name}]...")
    corpus_rows = fetch_hf_split(dataset_id, config="corpus", split="train")
    corpus: Dict[str, Dict[str, str]] = {}
    populate_corpus(corpus, corpus_rows, id_col="_id", text_col="text", title_col="title")

    # Qrels — config="qrels", try test split first, fall back to train
    print_step(f"Loading qrels [{name}]...")
    qrels: Dict[str, Dict[str, int]] = {}
    try:
        qrels_rows = fetch_hf_split(dataset_id, config="qrels", split="test")
    except Exception:
        qrels_rows = fetch_hf_split(dataset_id, config="qrels", split="train")
    populate_qrels(qrels, qrels_rows, qid_col="query-id", did_col="corpus-id", score_col="score")
    # Keep only positive relevance judgements
    qrels = {qid: {did: s for did, s in rels.items() if s > 0} for qid, rels in qrels.items()}
    qrels = {qid: rels for qid, rels in qrels.items() if rels}

    # Queries — config="queries"; split name varies across datasets ("queries" or "test")
    print_step(f"Loading queries [{name}]...")
    try:
        queries_rows = fetch_hf_split(dataset_id, config="queries", split="queries")
    except Exception:
        queries_rows = fetch_hf_split(dataset_id, config="queries", split="test")
    queries: Dict[str, str] = {}
    for row in queries_rows:
        qid = str(row["_id"])
        if qid in qrels:
            queries[qid] = str(row.get("text", ""))
        if len(queries) >= QUERIES_PER_DATASET:
            break

    qrels = {qid: qrels[qid] for qid in queries}
    return corpus, queries, qrels


# --- Index Building ---

def build_retriever(
    corpus: Dict[str, Dict[str, str]],
    qrels: Dict[str, Dict[str, int]],
) -> Tuple[LangChainRetriever, FAISS]:
    """Builds a FAISS index over the corpus, preserving all gold documents."""
    gold_ids: Set[str] = {did for doc_scores in qrels.values() for did in doc_scores}

    lc_docs: List[Document] = []
    included = 0
    for doc_id, doc_data in corpus.items():
        if included >= MAX_CORPUS_DOCS and doc_id not in gold_ids:
            continue
        text = doc_data.get("text", "")
        title = doc_data.get("title", "")
        page_content = f"{title}\n{text}".strip() if title else text
        lc_docs.append(Document(page_content=page_content, metadata={"id": doc_id}))
        included += 1

    print_step(f"Indexing {len(lc_docs)} documents (gold pool: {len(gold_ids)})...")
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
        "NDCG@10": round(metrics["ndcg"].get("NDCG@10", 0.0), 4),
        "Recall@10": round(metrics["recall"].get("Recall@10", 0.0), 4),
        "Recall@50": round(metrics["recall"].get("Recall@50", 0.0), 4),
    }


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
    assembler = GreedyAssembler(max_docs=CANDIDATES_TOP_K)
    return [
        (
            "Static Rerank (budget=20)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=reranker,
                assembler=assembler,
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
                assembler=assembler,
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
                assembler=assembler,
                scheduler=ActiveLearningScheduler(batch_size=5),
                estimator=SimilarityEstimator(),
                budget=CostBudget.simple(docs=20, tokens=100_000, latency=600_000),
            ),
        ),
    ]


# --- Main ---

def main():
    config.set("retrieval.original_query_depth", CANDIDATES_TOP_K)

    print_header("RAGtune × CoIR Benchmark")
    print_step(
        f"Datasets: {DATASETS}  |  Queries/dataset: {QUERIES_PER_DATASET}"
        f"  |  Candidates: {CANDIDATES_TOP_K}  |  Corpus cap: {MAX_CORPUS_DOCS}"
    )

    all_rows: List[Dict] = []

    for dataset_name in DATASETS:
        if dataset_name not in ALL_DATASETS:
            _console.print(
                f"[yellow]Unknown dataset '{dataset_name}', skipping. "
                f"Valid: {ALL_DATASETS}[/yellow]"
            )
            continue

        print_header(f"\n── Dataset: {dataset_name} ──")
        corpus, queries, qrels = load_task(dataset_name)
        n_qrels = sum(len(v) for v in qrels.values())
        print_step(f"Loaded {len(corpus)} corpus docs, {len(queries)} queries, {n_qrels} qrel pairs")

        retriever, vectorstore = build_retriever(corpus, qrels)

        def _record(scenario_name, results, avg_reranked=0.0, avg_latency=0.0, *, _ds=dataset_name, _qr=qrels):
            metrics = score_results(results, _qr)
            all_rows.append({
                "dataset": _ds,
                "scenario": scenario_name,
                **metrics,
                "Avg Rerank Docs": round(avg_reranked, 1),
                "Avg Latency (ms)": round(avg_latency, 1),
            })

        faiss_results = run_faiss_baseline(vectorstore, queries)
        _record("No-Rerank (FAISS)", faiss_results)

        for name, controller in build_scenarios(retriever):
            ctrl_results, avg_reranked, avg_latency = run_controller_scenario(
                name, controller, queries
            )
            _record(name, ctrl_results, avg_reranked, avg_latency)

    print_header("\nFINAL RESULTS")
    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))
    print_success("\nCoIR benchmark complete.")


if __name__ == "__main__":
    main()
