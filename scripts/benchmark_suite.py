import os
import time
import pandas as pd
import ir_datasets
from datasets import load_dataset
from typing import List, Dict, Any, Optional

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.adapters.pyterrier import PyTerrierRetriever
from ragtune.components.rerankers import CrossEncoderReranker, OllamaListwiseReranker, SimulatedReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import SimilarityEstimator, UtilityEstimator
from rich.console import Console as _Console
_console = _Console()
def print_header(msg): _console.print(f"[bold blue]{msg}[/bold blue]")
def print_step(msg):   _console.print(f"[dim]{msg}[/dim]")
def print_success(msg): _console.print(f"[bold green]{msg}[/bold green]")
from benchmark_utils import get_found_rank, summarize_metrics, calculate_ndcg, calculate_mrr

import pyterrier as pt
if not pt.started():
    pt.init()

# --- Configuration ---
SAMPLES_PER_DATASET = 20
CANDIDATES_K = 50

# --- Data Loaders ---

def load_trec_covid():
    print_step("Loading TREC-COVID via ir_datasets...")
    dataset = ir_datasets.load("beir/trec-covid")
    queries = []
    for i, q in enumerate(dataset.queries_iter()):
        if i >= SAMPLES_PER_DATASET: break
        # Get qrels for this query
        qrels = [qr.doc_id for qr in dataset.qrels_iter() if qr.query_id == q.query_id and qr.relevance > 0]
        queries.append({"id": q.query_id, "text": q.text, "gold_ids": set(qrels)})
    
    # Corpus sample for indexing
    docs = []
    for i, d in enumerate(dataset.docs_iter()):
        if i >= 5000: break
        docs.append({"docno": d.doc_id, "text": f"{d.title} {d.text}"})
    return queries, docs

def load_bright(domain="biology"):
    print_step(f"Loading BRIGHT [{domain}] via HF...")
    queries_ds = load_dataset('xlangai/BRIGHT', 'examples', split=domain, streaming=True)
    queries = []
    gold_ids_all = set()
    for i, q in enumerate(queries_ds):
        if i >= SAMPLES_PER_DATASET: break
        queries.append({"id": f"q{i}", "text": q["query"], "gold_ids": set(q["gold_ids"])})
        gold_ids_all.update(q["gold_ids"])
    
    corpus_ds = load_dataset('xlangai/BRIGHT', 'documents', split=domain, streaming=True)
    docs = []
    for i, d in enumerate(corpus_ds):
        if i >= 5000 and d["id"] not in gold_ids_all: continue
        if i >= 10000: break # Hard limit
        docs.append({"docno": d["id"], "text": d["content"]})
    return queries, docs

# --- Scenario Runner ---

def run_scenario(name: str, controller: RAGtuneController, queries: List[Dict]):
    print_step(f"Running Scenario: {name}")
    results = []
    for q in queries:
        start_time = time.time()
        output = controller.run(q["text"])
        latency = (time.time() - start_time) * 1000
        
        found_rank = get_found_rank(output.documents, q["gold_ids"])
        ndcg_v = calculate_ndcg(output.documents, q["gold_ids"], k=5)
        
        results.append({
            "scenario": name,
            "query_id": q["id"],
            "found_at_rank": found_rank,
            "ndcg": ndcg_v,
            "rerank_docs": output.final_budget_state.get("rerank_docs", 0),
            "latency": latency
        })
    
    df = pd.DataFrame(results)
    return {
        "Scenario": name,
        "nDCG@5": df["ndcg"].mean(),
        "MRR": calculate_mrr(results),
        "Avg Rerank Count": df["rerank_docs"].mean(),
        "Avg Latency (ms)": df["latency"].mean()
    }

def main():
    print_header("RAGtune Unified Benchmark Suite")
    
    # 1. Setup Data (Using TREC-COVID for overall suite test)
    queries, docs = load_trec_covid()
    
    print_step("Indexing collection...")
    tmp_dir = tempfile.mkdtemp()
    index_path = os.path.join(tmp_dir, "index")
    indexer = pt.IterDictIndexer(index_path, overwrite=True, meta={'docno': 64})
    index_ref = indexer.index(docs)
    
    # 2. Setup Base Retriever
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    retriever = PyTerrierRetriever(bm25)
    
    # 3. Define Scenarios
    scenarios = [
        # Baseline: Rerank top 20 docs in one go
        (
            "Baseline (MiniLM)", 
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
                assembler=GreedyAssembler(),
                scheduler=ActiveLearningScheduler(batch_size=20),
                budget=CostBudget(max_reranker_docs=20)
            )
        ),
        # RAGtune: Iterative (Batch size 2, total budget 10)
        (
            "RAGtune (MiniLM - Budget 10)",
            RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
                assembler=GreedyAssembler(),
                scheduler=ActiveLearningScheduler(batch_size=2, estimator=SimilarityEstimator()),
                budget=CostBudget(max_reranker_docs=10)
            )
        ),
        # RAGtune: Higher Quality (MiniLM - Budget 20)
        (
            "RAGtune (MiniLM - Budget 20)",
             RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
                assembler=GreedyAssembler(),
                scheduler=ActiveLearningScheduler(batch_size=5, estimator=SimilarityEstimator()),
                budget=CostBudget(max_reranker_docs=20)
            )
        ),
        # RAGtune Generative: Ollama
        (
            "RAGtune (Ollama/DeepSeek - Budget 4)",
             RAGtuneController(
                retriever=retriever,
                reformulator=IdentityReformulator(),
                reranker=OllamaListwiseReranker(model_name="deepseek-r1:8b"),
                assembler=GreedyAssembler(),
                scheduler=ActiveLearningScheduler(batch_size=2, estimator=SimilarityEstimator()),
                budget=CostBudget(max_reranker_docs=4)
            )
        )
    ]
    
    # 4. Run Benchmarks
    all_summary = []
    for name, controller in scenarios:
        metrics = run_scenario(name, controller, queries)
        all_summary.append(metrics)
        
    # 5. Report
    print_header("FINAL BENCHMARK SUMMARY")
    summary_df = pd.DataFrame(all_summary)
    print(summary_df.to_string(index=False))
    
    print_success("Benchmark Suite Completed!")

import tempfile
if __name__ == "__main__":
    main()
