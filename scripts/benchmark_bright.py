import os
import json
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.adapters.langchain import LangChainRetriever
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import SimilarityEstimator, UtilityEstimator
from ragtune.utils.console import print_header, print_step

# Configuration
DOMAINS = ["biology", "coding", "mathematics"]
QUERIES_PER_DOMAIN = 5
CANDIDATES_TOP_K = 50
RERANK_BUDGET = 10  # Max docs we are allowed to rerank

def load_bright_data(domain: str):
    """Loads queries and corpus for a domain."""
    print_step(f"Loading BRIGHT [{domain}]...")
    # Map domain to the correct split names in 'examples' and 'documents'
    # For BRIGHT, configs are 'examples' and 'documents', splits are domain names
    
    # Load queries
    ds = load_dataset('xlangai/BRIGHT', 'examples', split=domain, streaming=True)
    queries = []
    for i, q in enumerate(ds):
        if i >= QUERIES_PER_DOMAIN: break
        # Ensure standard keys for evaluate function
        queries.append({
            "query": q["query"],
            "gold_ids": q["gold_ids"]
        })
    
    # Load corpus
    corpus_ds = load_dataset('xlangai/BRIGHT', 'documents', split=domain, streaming=True)
    
    # We need to ensure gold documents are in our corpus
    gold_ids = set()
    for q in queries:
        gold_ids.update(q["gold_ids"])
    
    # Load sample corpus (first 1000 docs) + ensure gold docs are present
    corpus = []
    found_gold = set()
    
    print_step(f"Sampling corpus for relevance...")
    for i, c in enumerate(corpus_ds):
        doc_id = c["id"]
        is_gold = doc_id in gold_ids
        
        if i < 1000 or is_gold:
            corpus.append({"id": doc_id, "content": c["content"]})
            if is_gold:
                found_gold.add(doc_id)
        
        # Optimization: exit early if we have enough docs and all gold docs
        if i >= 1000 and len(found_gold) == len(gold_ids):
            break
        if i > 5000: break # Safety limit
            
    return queries, corpus


def evaluate(controller, queries):
    """Runs evaluation and returns metrics."""
    results = []
    for q in queries:
        query_str = q["query"]
        gold_ids = set(q["gold_ids"])
        
        start = time.time()
        output = controller.run(query_str)
        elapsed = time.time() - start
        
        found = any(doc.id in gold_ids for doc in output.documents)
        docs_reranked = output.final_budget_state.get("rerank_docs", 0)
        
        results.append({
            "found": found,
            "docs_reranked": docs_reranked,
            "latency": elapsed
        })
    
    df = pd.DataFrame(results)
    return {
        "accuracy": df["found"].mean(),
        "avg_docs_reranked": df["docs_reranked"].mean(),
        "avg_latency": df["latency"].mean()
    }

def run_benchmark():
    print_header("RAGtune Advanced Benchmarking: The BRIGHT Test")
    
    all_metrics = []
    
    for domain in DOMAINS:
        queries, corpus = load_bright_data(domain)
        
        # Indexing
        print_step(f"Indexing {len(corpus)} documents...")
        docs = [Document(page_content=c["content"], metadata={"id": c["id"]}) for c in corpus]
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = LangChainRetriever(vectorstore.as_retriever(search_kwargs={"k": CANDIDATES_TOP_K}))
        
        # 1. Baseline (Static Reranking)
        # In baseline, we rerank all docs in one batch (or as many as allowed)
        baseline_controller = RAGtuneController(
            retriever=retriever,
            reformulator=IdentityReformulator(),
            reranker=SimulatedReranker(),
            assembler=GreedyAssembler(),
            scheduler=ActiveLearningScheduler(batch_size=CANDIDATES_TOP_K), # One big batch
            budget=CostBudget(max_reranker_docs=CANDIDATES_TOP_K) 
        )
        
        # 2. RAGtune (Iterative + Similarity)
        ragtune_controller = RAGtuneController(
            retriever=retriever,
            reformulator=IdentityReformulator(),
            reranker=SimulatedReranker(),
            assembler=GreedyAssembler(),
            scheduler=ActiveLearningScheduler(
                batch_size=5, 
                estimator=SimilarityEstimator()
            ),
            budget=CostBudget(max_reranker_docs=RERANK_BUDGET)
        )
        
        print_step(f"Evaluating Baseline...")
        m_baseline = evaluate(baseline_controller, queries)
        m_baseline["domain"] = domain
        m_baseline["method"] = "Baseline (Static-Rerank-All)"
        
        print_step(f"Evaluating RAGtune...")
        m_ragtune = evaluate(ragtune_controller, queries)
        m_ragtune["domain"] = domain
        m_ragtune["method"] = "RAGtune (Iterative-Budget-10)"
        
        all_metrics.extend([m_baseline, m_ragtune])

    # Final Summary Table
    final_df = pd.DataFrame(all_metrics)
    print("\n" + "="*50)
    print("FINAL BENCHMARK SUMMARY")
    print("="*50)
    print(final_df[["domain", "method", "accuracy", "avg_docs_reranked", "avg_latency"]].to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
