import numpy as np
import pandas as pd
from typing import List, Set, Dict, Any, Optional, Tuple, Callable


# ── Metric Calculations ──────────────────────────────────────────────────────


def ndcg_at_k(
    documents: List[Any], qrels: Dict[Tuple[str, str], int], query_id: str, k: int = 5
) -> float:
    """NDCG@k supporting graded relevance (0-3) from qrels."""
    rels = [qrels.get((query_id, doc.id), 0) for doc in documents[:k]]
    return _ndcg_from_rels(rels)


def ndcg_at_k_from_ids(
    doc_ids: List[str], qrels: Dict[Tuple[str, str], int], query_id: str, k: int = 5
) -> float:
    """NDCG@k from a list of document IDs."""
    rels = [qrels.get((query_id, did), 0) for did in doc_ids[:k]]
    return _ndcg_from_rels(rels)


def _ndcg_from_rels(rels: List[int]) -> float:
    """NDCG from a list of relevance values."""
    if not rels:
        return 0.0

    def dcg(r):
        return sum((2**v - 1) / np.log2(i + 2) for i, v in enumerate(r))

    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal)
    return dcg(rels) / idcg if idcg > 0 else 0.0


def recall_at_k(
    documents: List[Any], qrels: Dict[Tuple[str, str], int], query_id: str, k: int = 5
) -> float:
    """Recall@k using qrels."""
    relevant = {did for (qid, did), rel in qrels.items() if qid == query_id and rel > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for doc in documents[:k] if doc.id in relevant)
    return min(hits / len(relevant), 1.0)


def recall_at_k_from_ids(
    doc_ids: List[str], qrels: Dict[Tuple[str, str], int], query_id: str, k: int = 5
) -> float:
    """Recall@k from a list of document IDs."""
    relevant = {did for (qid, did), rel in qrels.items() if qid == query_id and rel > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for did in doc_ids[:k] if did in relevant)
    return min(hits / len(relevant), 1.0)


def mrr(
    documents: List[Any], qrels: Dict[Tuple[str, str], int], query_id: str
) -> float:
    """Mean Reciprocal Rank."""
    relevant = {did for (qid, did), rel in qrels.items() if qid == query_id and rel > 0}
    for i, doc in enumerate(documents):
        if doc.id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mrr_from_ids(
    doc_ids: List[str], qrels: Dict[Tuple[str, str], int], query_id: str
) -> float:
    """Mean Reciprocal Rank from a list of document IDs."""
    relevant = {did for (qid, did), rel in qrels.items() if qid == query_id and rel > 0}
    for i, did in enumerate(doc_ids):
        if did in relevant:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(output_docs: List[Any], gold_ids: Set[str], k: int = 5) -> float:
    """Binary NDCG@k (legacy, for compatibility)."""
    relevance = [1 if str(doc.id) in gold_ids else 0 for doc in output_docs]

    def dcg(r):
        return np.sum(r / np.log2(np.arange(2, len(r) + 2)))

    dcg_val = dcg(relevance[:k])
    ideal = sorted(relevance, reverse=True)
    idcg_val = dcg(ideal[:k])
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def calculate_mrr(results: List[Dict[str, Any]]) -> float:
    """Mean Reciprocal Rank (legacy, for compatibility)."""
    if not results:
        return 0.0
    rr_sum = 0.0
    for res in results:
        rank = res.get("found_at_rank")
        if rank is not None:
            rr_sum += 1.0 / rank
    return rr_sum / len(results)


def get_found_rank(output_docs: List[Any], gold_ids: Set[str]) -> Optional[int]:
    """1-indexed rank of the first gold document (legacy, for compatibility)."""
    for i, doc in enumerate(output_docs):
        if str(doc.id) in gold_ids:
            return i + 1
    return None


# ── Indexing ──────────────────────────────────────────────────────────────────


def build_pyterrier_index(
    corpus: List[Dict], index_path: str = None, meta: Dict = None, fields: bool = True
):
    """Build a PyTerrier BM25 index from a list of documents.

    Args:
        corpus: List of dicts with 'docno' and 'text' keys
        index_path: Path for the index (uses tempdir if None)
        meta: Metadata config for the indexer
        fields: Whether to enable field-aware indexing
    Returns:
        (bm25_retriever, index_path)
    """
    import pyterrier as pt
    import tempfile
    import os

    if not pt.java.started():
        pt.java.init()

    if index_path is None:
        tmp_dir = tempfile.mkdtemp()
        index_path = os.path.join(tmp_dir, "index")
    else:
        os.makedirs(index_path, exist_ok=True)

    if meta is None:
        meta = {"docno": 128, "text": 4096}

    indexer_kwargs = {"overwrite": True, "meta": meta}
    if fields:
        indexer_kwargs["fields"] = True

    indexer = pt.IterDictIndexer(index_path, **indexer_kwargs)
    index_ref = indexer.index(iter(corpus))
    bm25 = pt.terrier.Retriever(
        index_ref, wmodel="BM25", metadata=["docno", "text"], num_results=100
    )
    return bm25, index_path


# ── Data Loading Helpers ──────────────────────────────────────────────────────


def load_beir_dataset(dataset_id: str, n_queries: int = 50, doc_cap: int = 0):
    """Load a BEIR dataset via ir_datasets.

    Args:
        dataset_id: ir_datasets identifier (e.g., "beir/trec-covid")
        n_queries: Max queries to load
        doc_cap: Max docs to load (0 = all)
    Returns:
        (doc_iter_fn, queries, qrels)
    """
    import ir_datasets

    ds = ir_datasets.load(dataset_id)

    qrels = {}
    for qr in ds.qrels_iter():
        qrels[(qr.query_id, qr.doc_id)] = qr.relevance

    relevant_qids = {qid for (qid, _), r in qrels.items() if r > 0}
    queries = []
    for q in ds.queries_iter():
        if q.query_id in relevant_qids:
            queries.append({"id": q.query_id, "text": q.text})
        if len(queries) >= n_queries:
            break

    def doc_iter():
        for i, doc in enumerate(ds.docs_iter()):
            if doc_cap and i >= doc_cap:
                break
            text = (
                getattr(doc, "title", "")
                + " "
                + getattr(doc, "text", getattr(doc, "abstract", ""))
            ).strip()
            yield {"docno": doc.doc_id, "text": text}

    return doc_iter, queries, qrels


def summarize_results(
    results: pd.DataFrame, group_col: str = "config", metric_cols: List[str] = None
) -> pd.DataFrame:
    """Summarize benchmark results by group."""
    if metric_cols is None:
        metric_cols = ["ndcg@5", "recall@5", "mrr"]

    available = [c for c in metric_cols if c in results.columns]
    if not available:
        return results.groupby(group_col).mean()

    agg = {c: "mean" for c in available}
    if "latency_ms" in results.columns:
        agg["latency_ms"] = "mean"
    if "rerank_docs" in results.columns:
        agg["rerank_docs"] = "mean"
    if "n_iterations" in results.columns:
        agg["n_iterations"] = "mean"

    return results.groupby(group_col).agg(agg).reset_index()
