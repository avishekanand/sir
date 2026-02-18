import pyterrier as pt
import pandas as pd
import os
import tempfile
import ir_datasets
import numpy as np
from typing import List, Dict, Optional
from ragtune.adapters.pyterrier import PyTerrierRetriever
from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.components.rerankers import OllamaListwiseReranker
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import GracefulDegradationScheduler
from ragtune.components.estimators import BaselineEstimator
from ragtune.components.reformulators import LLMReformulator
from rich.console import Console
from rich.table import Table
from rich import box

def calculate_metrics(documents: List, qrels: Dict, query_id: str, k: int = 5):
    """Calculate NDCG@K, Precision@K, and MRR."""
    if not documents:
        return {"ndcg": 0.0, "precision": 0.0, "mrr": 0.0}
        
    relevances = []
    for doc in documents[:k]:
        rel_key = f"{query_id}-{doc.id}"
        # TREC-COVID: 2=highly relevant, 1=relevant, 0=not relevant
        rel = qrels.get(rel_key, 0)
        relevances.append(rel)
    
    def get_dcg(rels):
        return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
    
    dcg = get_dcg(relevances)
    ideal_rels = sorted(relevances, reverse=True)
    idcg = get_dcg(ideal_rels)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    precision = sum(1 for r in relevances if r > 0) / k
    
    mrr = 0.0
    for i, r in enumerate(relevances):
        if r > 0:
            mrr = 1.0 / (i + 1)
            break
            
    return {"ndcg": ndcg, "precision": precision, "mrr": mrr}

def run_demo():
    console = Console()
    console.print("[bold blue]PyTerrier + RAGtune: Local Ollama NDCG Evaluation[/bold blue]\n")

    if not pt.started():
        pt.init()

    dataset_id = "beir/trec-covid"
    console.print(f"[dim]Loading dataset: {dataset_id}...[/dim]")
    dataset = ir_datasets.load(dataset_id)
    
    # 1. Setup Sample Index (50k docs)
    tmp_dir = tempfile.mkdtemp()
    index_path = os.path.join(tmp_dir, "index")
    
    console.print(f"[dim]Indexing 50,000 docs to {index_path}...[/dim]")
    def get_docs():
        for i, doc in enumerate(dataset.docs_iter()):
            if i >= 50000: break
            yield {"docno": doc.doc_id, "text": f"{doc.title} {doc.text}", "title": doc.title}
            
    indexer = pt.IterDictIndexer(index_path, overwrite=True, meta={'docno': 64, 'text': 2048, 'title': 256})
    index_ref = indexer.index(get_docs())
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", metadata=['docno', 'text', 'title'])

    # Pre-load Qrels
    qrels = {}
    for qrel in dataset.qrels_iter():
        qrels[f"{qrel.query_id}-{qrel.doc_id}"] = qrel.relevance

    # 2. Setup RAGtune Components
    retriever = PyTerrierRetriever(bm25)
    
    def create_controller(tier_config):
        # Reformulator
        from ragtune.components.reformulators import IdentityReformulator
        if tier_config["model"] == "identity":
            reformulator = IdentityReformulator()
        else:
            reformulator = LLMReformulator(
                model_name=tier_config.get("model", "ollama/deepseek-r1:8b"),
                api_base=tier_config.get("api_base", "http://localhost:11434")
            )
        
        # Reranker (Ollama)
        reranker = OllamaListwiseReranker(
            model_name="deepseek-r1:8b", 
            base_url="http://localhost:11434"
        )
        
        return RAGtuneController(
            retriever=retriever,
            reformulator=reformulator,
            reranker=reranker,
            assembler=GreedyAssembler(),
            scheduler=GracefulDegradationScheduler(batch_size=tier_config.get("batch_size", 3)),
            estimator=BaselineEstimator(),
            budget=CostBudget(limits=tier_config["budget"])
        )

    # 3. Define Budget Tiers
    tiers = {
        "Cheap (BM25 only)": {
            "budget": {"tokens": 2000, "retrieval_calls": 1, "rerank_docs": 0, "latency_ms": 30000},
            "model": "identity",
            "batch_size": 1
        },
        "Balanced (Ollama Rewrites+Rerank)": {
            "budget": {"tokens": 5000, "retrieval_calls": 2, "rerank_docs": 5, "latency_ms": 300000},
            "model": "ollama/deepseek-r1:8b",
            "batch_size": 3
        },
        "Premium (Deep Rewrites+Rerank)": {
            "budget": {"tokens": 20000, "retrieval_calls": 5, "rerank_docs": 10, "latency_ms": 600000},
            "model": "ollama/deepseek-r1:8b",
            "batch_size": 5
        }
    }

    # Find queries with results in 50k
    eval_queries = []
    for q in dataset.queries_iter():
        relevant_in_top = sum(1 for qrel in dataset.qrels_iter() if qrel.query_id == q.query_id and qrel.relevance > 0)
        if relevant_in_top > 10:
            eval_queries.append(q)
            if len(eval_queries) >= 2: break
    
    if not eval_queries:
        eval_queries = list(dataset.queries_iter())[:2]

    results = []

    for name, tier_config in tiers.items():
        console.print(f"\n[bold yellow]>>> Evaluating {name}...[/bold yellow]")
        
        # Explicit Pipeline Specification
        pipe_table = Table(box=box.SIMPLE, show_header=False)
        pipe_table.add_row("[magenta]Retriever:[/magenta]", "PyTerrier (BM25)")
        
        reform_model = tier_config["model"]
        pipe_table.add_row("[magenta]Reformulator:[/magenta]", f"LLM ({reform_model})" if reform_model != "identity" else "Identity (No-Op)")
        
        rerank_limit = tier_config["budget"].get("rerank_docs", 0)
        pipe_table.add_row("[magenta]Reranker:[/magenta]", f"Ollama Listwise (Docs: {rerank_limit})" if rerank_limit > 0 else "None (BM25 scores only)")
        
        pipe_table.add_row("[magenta]Estimator:[/magenta]", "Baseline (Reformulate everything)")
        pipe_table.add_row("[magenta]Budget:[/magenta]", str(tier_config["budget"]))
        
        console.print(pipe_table)
        
        controller = create_controller(tier_config)
        
        m_ndcg, m_mrr, m_prec = [], [], []
        t_spend, t_calls, t_docs = [], [], []
        
        for q in eval_queries:
            console.print(f"[dim]Query: {q.text[:60]}...[/dim]")
            output = controller.run(q.text)
            
            # Forensic Print
            if not output.documents:
                 console.print(f"[bold red]WARNING: 0 documents returned![/bold red]")
                 # Print a few trace events to see if latency was exceeded
                 for event in output.trace.events[-10:]:
                     console.print(f"  [dim]Trace: {event.action} - {event.details}[/dim]")

            metrics = calculate_metrics(output.documents, qrels, q.query_id, k=5)
            
            m_ndcg.append(metrics["ndcg"])
            m_mrr.append(metrics["mrr"])
            m_prec.append(metrics["precision"])
            t_spend.append(output.final_budget_state.get("tokens", 0))
            t_calls.append(output.final_budget_state.get("retrieval_calls", 0))
            t_docs.append(len(output.documents))

        results.append({
            "Tier": name,
            "NDCG@5": np.mean(m_ndcg),
            "MRR": np.mean(m_mrr),
            "Tokens": np.mean(t_spend),
            "Avg Yield": np.mean(t_docs)
        })

    # 4. Results Table
    res_table = Table(title="RAGtune v0.55 Evaluation (TREC-COVID 50k Sample)", box=box.DOUBLE)
    res_table.add_column("Tier", style="cyan")
    res_table.add_column("NDCG@5", justify="right", style="bold green")
    res_table.add_column("MRR", justify="right")
    res_table.add_column("Avg Tokens", justify="right")
    res_table.add_column("Avg Yield", justify="right")

    for r in results:
        res_table.add_row(
            r["Tier"],
            f"{r['NDCG@5']:.4f}",
            f"{r['MRR']:.4f}",
            f"{r['Tokens']:.0f}",
            f"{r['Avg Yield']:.1f}"
        )

    console.print(res_table)
    console.print("\n[dim]Note: Metrics are averages over sample queries. 'Yield' is number of docs returned within token budget.[/dim]")

if __name__ == "__main__":
    run_demo()
