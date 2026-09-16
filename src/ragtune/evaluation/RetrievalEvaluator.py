"""
Retrieval Evaluation Utilities
================================
Computes standard IR metrics from retrieval results and qrels.

Provides:
  - RetrievalEvaluator  : NDCG, MAP, Recall, Precision, MRR
  - evaluate_run()      : one-shot convenience function

Compatible with the BEIR EvaluateRetrieval interface so SIR's existing
evaluation pipeline can use it without modification.

Requires: pytrec_eval-terrier  (pip install pytrec_eval-terrier)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default k values matching BEIR / BRIGHT evaluation protocol
DEFAULT_K_VALUES = [1, 3, 5, 10, 50, 100]


def _check_pytrec():
    try:
        import pytrec_eval
        return pytrec_eval
    except ImportError:
        raise ImportError(
            "pytrec_eval is required for evaluation. "
            "Install with: pip install pytrec_eval-terrier"
        )


class RetrievalEvaluator:
    """
    Evaluates retrieval runs against ground-truth qrels.

    Parameters
    ----------
    k_values : List[int]
        Cut-off values for metrics.
    ignore_identical_ids : bool
        If True, skip retrieved docs where doc_id == query_id.
    """

    def __init__(
        self,
        k_values: Optional[List[int]] = None,
        ignore_identical_ids: bool = True,
    ):
        self.k_values = k_values or DEFAULT_K_VALUES
        self.ignore_identical_ids = ignore_identical_ids

    def evaluate(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute NDCG@k, MAP@k, Recall@k, Precision@k, MRR@k.

        Parameters
        ----------
        qrels : Dict[str, Dict[str, int]]
            Ground truth: {query_id: {doc_id: relevance}}.
        results : Dict[str, Dict[str, float]]
            Retrieval run: {query_id: {doc_id: score}}.
        k_values : List[int] | None
            Override self.k_values.

        Returns
        -------
        Dict with keys 'ndcg', 'map', 'recall', 'precision', 'mrr',
        each mapping to {f'@{k}': avg_score, ...}.
        """
        pytrec_eval = _check_pytrec()
        k_vals = k_values or self.k_values

        if self.ignore_identical_ids:
            results = self._filter_identical_ids(results)

        # Build per-metric evaluators
        ndcg_measures = {f"ndcg_cut.{k}" for k in k_vals}
        map_measures = {f"map_cut.{k}" for k in k_vals}
        recall_measures = {f"recall.{k}" for k in k_vals}
        prec_measures = {f"P.{k}" for k in k_vals}
        mrr_measures = {f"recip_rank"}

        all_measures = (
            ndcg_measures | map_measures | recall_measures | prec_measures | mrr_measures
        )

        evaluator = pytrec_eval.RelevanceEvaluator(
            {qid: {did: int(rel) for did, rel in rels.items()} for qid, rels in qrels.items()},
            all_measures,
        )

        # Filter results to only queries in qrels
        filtered_results = {qid: results[qid] for qid in qrels if qid in results}
        scores = evaluator.evaluate(filtered_results)

        # Aggregate across queries
        ndcg, _map, recall, precision, mrr = {}, {}, {}, {}, {}
        num_queries = len(scores)

        if num_queries == 0:
            logger.warning("No overlapping query IDs between qrels and results.")
            return {"ndcg": {}, "map": {}, "recall": {}, "precision": {}, "mrr": {}}

        for k in k_vals:
            ndcg[f"NDCG@{k}"] = (
                sum(s.get(f"ndcg_cut_{k}", 0.0) for s in scores.values()) / num_queries
            )
            _map[f"MAP@{k}"] = (
                sum(s.get(f"map_cut_{k}", 0.0) for s in scores.values()) / num_queries
            )
            recall[f"Recall@{k}"] = (
                sum(s.get(f"recall_{k}", 0.0) for s in scores.values()) / num_queries
            )
            precision[f"Precision@{k}"] = (
                sum(s.get(f"P_{k}", 0.0) for s in scores.values()) / num_queries
            )

        mrr["MRR"] = (
            sum(s.get("recip_rank", 0.0) for s in scores.values()) / num_queries
        )

        return {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
        }

    def evaluate_custom(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: Optional[List[int]] = None,
        metric: str = "mrr",
    ) -> Dict[str, float]:
        """
        Evaluate a single additional metric.

        Supported: 'mrr', 'holes', 'top_k_acc', 'judged_k'
        """
        all_metrics = self.evaluate(qrels, results, k_values)
        metric_lower = metric.lower()
        if metric_lower in all_metrics:
            return all_metrics[metric_lower]
        raise ValueError(f"Unsupported metric: {metric!r}")

    @staticmethod
    def _filter_identical_ids(
        results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Remove retrieved docs whose ID matches the query ID."""
        return {
            qid: {did: score for did, score in docs.items() if did != qid}
            for qid, docs in results.items()
        }

    @staticmethod
    def print_results(metrics: Dict[str, Dict[str, float]], title: str = "") -> None:
        """Pretty-print evaluation results."""
        if title:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")
        for metric_name, metric_dict in metrics.items():
            for k_label, score in sorted(metric_dict.items()):
                print(f"  {metric_name:12s}  {k_label:15s}  {score:.4f}")
        print()


def evaluate_run(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: Optional[List[int]] = None,
    title: str = "",
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    One-shot evaluation function.

    Parameters
    ----------
    qrels : Dict[str, Dict[str, int]]
    results : Dict[str, Dict[str, float]]
    k_values : List[int] | None
    title : str
        Label for printed output.
    verbose : bool
        Print results if True.

    Returns
    -------
    Dict with 'ndcg', 'map', 'recall', 'precision', 'mrr' sub-dicts.
    """
    evaluator = RetrievalEvaluator(k_values=k_values)
    metrics = evaluator.evaluate(qrels, results)
    if verbose:
        RetrievalEvaluator.print_results(metrics, title=title)
    return metrics
