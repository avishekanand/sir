import os
import tempfile
import pandas as pd
import ir_datasets
import pyterrier as pt
from typing import Dict, List

# Initialize PyTerrier
if not pt.started():
    pt.init()

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.core.types import RerankStrategy
from ragtune.adapters.pyterrier import PyTerrierRetriever
from ragtune.components.rerankers import (
    MultiStrategyReranker, 
    LLMReranker, 
    CrossEncoderReranker, 
    NoOpReranker,
    SimulatedReranker
)
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import GracefulDegradationScheduler
from ragtune.components.estimators import BaselineEstimator
from ragtune.utils.console import print_header, print_step, print_success, print_budget
from ragtune.utils.config import config
import numpy as np

def calculate_metrics(documents: List, qrels: Dict, query_id: str, k: int = 5):
    """Calculate NDCG@K, Precision@K, and MRR."""
    relevances = []
    for doc in documents[:k]:
        rel_key = f"{query_id}-{doc.id}"
        # TREC-COVID: 2=highly relevant, 1=relevant, 0=not relevant
        rel = qrels.get(rel_key, 0)
        relevances.append(rel)
    
    # Precise calculation for NDCG
    def get_dcg(rels):
        return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
    
    dcg = get_dcg(relevances)
    
    # Ideal DCG: Sort all known relevant docs for this query
    # Simple approximation: sort the relevances we found (not perfect but common for local demo)
    ideal_rels = sorted(relevances, reverse=True)
    idcg = get_dcg(ideal_rels)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    # Precision@K (rel > 0)
    precision = sum(1 for r in relevances if r > 0) / k
    
    # MRR
    mrr = 0.0
    for i, r in enumerate(relevances):
        if r > 0:
            mrr = 1.0 / (i + 1)
            break
            
    return {"ndcg": ndcg, "precision": precision, "mrr": mrr}

def run_comparison():
    dataset_id = "beir/trec-covid"
    dataset = ir_datasets.load(dataset_id)
    
    print_header(f"RAGtune: TREC-COVID Budget Comparison Demo")
    
    # 1. Indexing (Sample 10k)
    print_step("Indexing 50,000 documents with PyTerrier...")
    tmp_dir = tempfile.mkdtemp()
    index_path = os.path.join(tmp_dir, "index")
    
    def get_docs():
        for i, doc in enumerate(dataset.docs_iter()):
            if i >= 50000: break
            yield {"docno": doc.doc_id, "text": f"{doc.title} {doc.text}", "title": doc.title}
            
    indexer = pt.IterDictIndexer(index_path, overwrite=True, meta={'docno': 64, 'text': 2048, 'title': 256})
    index_ref = indexer.index(get_docs())
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", metadata=['docno', 'text', 'title'])
    
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console()

    # Pre-load Qrels for relevance reporting
    print_step("Loading Qrels for relevance mapping...")
    qrels = {}
    for qrel in dataset.qrels_iter():
        qrels[f"{qrel.query_id}-{qrel.doc_id}"] = qrel.relevance
    
    console.print(f"[dim]Loaded {len(qrels)} qrels. Sample keys: {list(qrels.keys())[:3]}[/dim]")
    
    # 2. Setup Multi-Strategy Reranker
    from ragtune.core.types import RerankStrategy
    strategies = {
        "llm": SimulatedReranker(), 
        "cross_encoder": CrossEncoderReranker(),
        "identity": NoOpReranker()
    }
    
    reranker = MultiStrategyReranker(strategies=strategies)
    retriever = PyTerrierRetriever(bm25)
    
    # 3. Scenarios
    scenarios = [
        {
            "name": "Scenario 1: High Budget (3 LLM, 10 CE)",
            "llm_limit": 3,
            "ce_limit": 10,
            "budget": CostBudget.simple(tokens=5000, docs=13)
        },
        {
            "name": "Scenario 2: Low Budget (5 CE only)",
            "llm_limit": 0,
            "ce_limit": 5,
            "budget": CostBudget.simple(tokens=5000, docs=5)
        }
    ]
    
    # Get first few queries for a better average
    num_eval_queries = 3
    queries = list(dataset.queries_iter())[:num_eval_queries]
    
    results_summary = []

    for scenario in scenarios:
        console.print(f"\n[bold yellow]>>> Evaluating {scenario['name']}...[/bold yellow]")
        
        scenario_metrics = {"ndcg": [], "precision": [], "mrr": []}
        
        controller = RAGtuneController(
            retriever=retriever,
            reformulator=IdentityReformulator(),
            reranker=reranker,
            assembler=GreedyAssembler(),
            scheduler=GracefulDegradationScheduler(
                llm_limit=scenario["llm_limit"],
                cross_encoder_limit=scenario["ce_limit"],
                batch_size=5
            ),
            estimator=BaselineEstimator(),
            budget=scenario["budget"]
        )

        for q_idx, query in enumerate(queries):
            console.print(f"[dim]Query {q_idx+1}/{num_eval_queries}: {query.text[:50]}...[/dim]")
            output = controller.run(query.text)
            
            # Calculate metrics
            m = calculate_metrics(output.documents, qrels, query.query_id, k=5)
            for k_met in scenario_metrics:
                scenario_metrics[k_met].append(m[k_met])
                
            # For the VERY first query, show verbose document breakdown
            if q_idx == 0:
                print_budget(output.final_budget_state)
                console.print(f"\n[bold underline]Detailed Breakdown (Query 1): {scenario['name']}[/bold underline]")
                for i, doc in enumerate(output.documents[:5]):
                    rel_key = f"{query.query_id}-{doc.id}"
                    gt_relevance = str(qrels.get(rel_key, "N/A"))
                    initial_rank = doc.metadata.get("initial_rank", "N/A")
                    
                    provenance = "BM25"
                    if i < scenario["llm_limit"]: provenance = "LLM"
                    elif i < (scenario["llm_limit"] + scenario["ce_limit"]): provenance = "Cross_Encoder"
                    
                    title = doc.metadata.get("title", "No Title")
                    console.print(f"[bold]Rank {i+1}:[/bold] {doc.id} | [green]{provenance}[/green] | [cyan]InitRank: {initial_rank}[/cyan] | [bold magenta]GT Rel: {gt_relevance}[/bold magenta]")
                    console.print(f"   [yellow]Title:[/yellow] {title}\n")

        # Aggregate metrics
        avg_ndcg = np.mean(scenario_metrics["ndcg"])
        avg_prec = np.mean(scenario_metrics["precision"])
        avg_mrr = np.mean(scenario_metrics["mrr"])
        
        results_summary.append({
            "Scenario": scenario["name"],
            "nDCG@5": f"{avg_ndcg:.4f}",
            "P@5": f"{avg_prec:.4f}",
            "MRR": f"{avg_mrr:.4f}"
        })

    # Final Comparison Table
    console.print("\n[bold cyan]Overall Evaluation Summary[/bold cyan]")
    summary_table = Table(box=box.DOUBLE)
    summary_table.add_column("Scenario", style="yellow")
    summary_table.add_column("nDCG@5", justify="right")
    summary_table.add_column("P@5", justify="right")
    summary_table.add_column("MRR", justify="right")
    
    for row in results_summary:
        summary_table.add_row(row["Scenario"], row["nDCG@5"], row["P@5"], row["MRR"])
    
    console.print(summary_table)
    print_success("Comparison Demo with Metrics Completed!")

if __name__ == "__main__":
    run_comparison()
