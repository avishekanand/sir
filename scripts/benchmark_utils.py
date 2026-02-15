import numpy as np
import pandas as pd
from typing import List, Set, Dict, Any

def calculate_dcg(rel: List[int], k: int) -> float:
    """Calculates Discounted Cumulative Gain at k."""
    rel = np.asarray(rel, dtype=float)[:k]
    if rel.size:
        return np.sum(rel / np.log2(np.arange(2, rel.size + 2)))
    return 0.0

def calculate_ndcg(output_docs: List[Any], gold_ids: Set[str], k: int = 5) -> float:
    """Calculates Normalized Discounted Cumulative Gain at k (binary relevance)."""
    # 1. Calculate DCG
    relevance = [1 if str(doc.id) in gold_ids else 0 for doc in output_docs]
    dcg = calculate_dcg(relevance, k)
    
    # 2. Calculate IDCG
    # Ideal relevance is all gold docs at the top
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = calculate_dcg(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def calculate_recall_at_k(results: List[Dict[str, Any]], k: int = 1) -> float:
    """
    Calculates Recall@k across a list of query results.
    Each result should have 'found_at_rank' (integer or None).
    """
    if not results:
        return 0.0
    
    hits = 0
    for res in results:
        rank = res.get("found_at_rank")
        if rank is not None and rank <= k:
            hits += 1
            
    return hits / len(results)

def calculate_mrr(results: List[Dict[str, Any]]) -> float:
    """Calculates Mean Reciprocal Rank."""
    if not results:
        return 0.0
    
    rr_sum = 0.0
    for res in results:
        rank = res.get("found_at_rank")
        if rank is not None:
            rr_sum += 1.0 / rank
            
    return rr_sum / len(results)

def get_found_rank(output_docs: List[Any], gold_ids: Set[str]) -> int | None:
    """Finds the 1-indexed rank of the first gold document in the output."""
    for i, doc in enumerate(output_docs):
        if str(doc.id) in gold_ids:
            return i + 1
    return None

def summarize_metrics(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Summarizes a list of metrics dictionaries into a formatted DataFrame."""
    df = pd.DataFrame(metrics_list)
    return df
