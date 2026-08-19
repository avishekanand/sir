from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from ragtune.tuning.pruners import TuningPruner


@dataclass
class EvalQuery:
    query_id: str
    query: str
    qrels: Dict[str, int]  # doc_id → relevance (integer, higher = more relevant)


@dataclass
class EvalDataset:
    name: str
    queries: List[EvalQuery]

    def iter_queries(self, limit: Optional[int] = None) -> Iterator[EvalQuery]:
        items = self.queries if limit is None else self.queries[:limit]
        yield from items

    @classmethod
    def from_dicts(
        cls,
        name: str,
        queries: List[Dict],
        qrels: Dict[str, Dict[str, int]],
    ) -> EvalDataset:
        """
        queries: [{"query_id": ..., "query": ...}, ...]
        qrels:   {query_id: {doc_id: relevance, ...}, ...}
        """
        eval_queries = [
            EvalQuery(
                query_id=q["query_id"],
                query=q["query"],
                qrels=qrels.get(q["query_id"], {}),
            )
            for q in queries
        ]
        return cls(name=name, queries=eval_queries)

    @classmethod
    def from_pyterrier_irds(
        cls,
        irds_id: str,
        n_queries: Optional[int] = None,
        seed: int = 42,
    ) -> EvalDataset:
        """
        Load queries and qrels from a PyTerrier IRDS dataset.

        irds_id examples:
          "irds:beir/trec-covid/test"
          "irds:beir/nfcorpus/test"
        """
        import pyterrier as pt  # type: ignore

        if not pt.started():
            pt.init()

        dataset = pt.get_dataset(irds_id)
        topics_df = dataset.get_topics()    # columns vary by dataset
        qrels_df = dataset.get_qrels()      # columns: qid, docno, label

        # Normalise to a 'query' column.  BEIR datasets expose different field
        # names: trec-covid has ('text','query','narrative'), nfcorpus has
        # ('text','url'), scifact/fiqa have just ('text',).  We prefer 'query'
        # (the short title used in BEIR benchmarks) then fall back to 'text'.
        if "query" not in topics_df.columns:
            for candidate in ("text", "title", "question"):
                if candidate in topics_df.columns:
                    topics_df = topics_df.rename(columns={candidate: "query"})
                    break
            else:
                raise ValueError(
                    f"Cannot find a query column in topics for {irds_id}. "
                    f"Available columns: {list(topics_df.columns)}"
                )

        # Build qrels dict
        qrels_map: Dict[str, Dict[str, int]] = {}
        for _, row in qrels_df.iterrows():
            qrels_map.setdefault(str(row["qid"]), {})[str(row["docno"])] = int(row["label"])

        rng = np.random.default_rng(seed)
        rows = topics_df.to_dict("records")
        if n_queries is not None and n_queries < len(rows):
            indices = rng.choice(len(rows), size=n_queries, replace=False)
            rows = [rows[i] for i in sorted(indices)]

        eval_queries = [
            EvalQuery(
                query_id=str(r["qid"]),
                query=str(r["query"]),
                qrels=qrels_map.get(str(r["qid"]), {}),
            )
            for r in rows
        ]
        return cls(name=irds_id, queries=eval_queries)


@dataclass
class TrialObjectives:
    ndcg_at_10: float        # objective 1 — maximize
    rerank_docs: float       # objective 2 — minimize (mean per query)
    latency_ms: float        # logged as user attribute
    queries_evaluated: int


def ndcg_at_k(ranked_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """
    Graded NDCG with binary-gain normalisation.

    ranked_ids — doc IDs in rank order (best first), from ControllerOutput.documents
    qrels      — {doc_id: relevance_integer} for this query
    """
    ranked_k = ranked_ids[:k]

    def gain(rel: int) -> float:
        return 2.0 ** rel - 1.0

    dcg = sum(
        gain(qrels.get(doc_id, 0)) / np.log2(i + 2)
        for i, doc_id in enumerate(ranked_k)
    )

    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(gain(rel) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return float(dcg / idcg) if idcg > 0 else 0.0


class TrialEvaluator:
    """
    Runs a RAGtuneController against an EvalDataset and returns
    (NDCG@10, mean_rerank_docs).

    Calls trial.report() after every query so that Optuna's built-in
    MedianPruner (if configured) can also act on intermediate values.
    Calls each TuningPruner after every query; raises optuna.TrialPruned
    if any pruner fires.
    """

    def __init__(
        self,
        dataset: EvalDataset,
        n_eval_queries: int = 200,
        pruners: Optional[List[TuningPruner]] = None,
        report_interval: int = 1,
    ):
        self.dataset = dataset
        self.n_eval_queries = n_eval_queries
        self.pruners: List[TuningPruner] = pruners or []
        self.report_interval = report_interval

    def evaluate(
        self,
        controller: object,
        trial: object,
        retrieval_overrides: Optional[Dict[str, object]] = None,
    ) -> TrialObjectives:
        """
        Parameters
        ----------
        controller
            A RAGtuneController instance built from trial params.
        trial
            The active optuna.Trial (used for trial.report()).
        retrieval_overrides
            Dict of config.set() calls to apply before each query, e.g.
            {"retrieval.original_query_depth": 20}.  These override the
            global config singleton for the duration of this evaluation.
        """
        import optuna

        for pruner in self.pruners:
            pruner.reset()

        # Apply retrieval config overrides once (they persist for this trial)
        if retrieval_overrides:
            from ragtune.utils.config import config as global_cfg
            for k, v in retrieval_overrides.items():
                global_cfg.set(k, v)

        ndcg_scores: List[float] = []
        rerank_costs: List[float] = []
        latencies_ms: List[float] = []
        n_total = min(self.n_eval_queries, len(self.dataset.queries))

        for step, eq in enumerate(self.dataset.iter_queries(limit=n_total), start=1):
            t0 = time.monotonic()
            try:
                output = controller.run(eq.query)  # type: ignore[union-attr]
            except Exception as exc:
                # Treat crashed queries as zero-NDCG to not abort the trial entirely
                ndcg_scores.append(0.0)
                rerank_costs.append(0.0)
                latencies_ms.append(0.0)
                trial.set_user_attr(f"error_step_{step}", str(exc))  # type: ignore
                continue

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            ranked_ids = [doc.id for doc in output.documents]
            ndcg = ndcg_at_k(ranked_ids, eq.qrels, k=10)
            cost = float(output.final_budget_state.get("rerank_docs", 0.0))

            ndcg_scores.append(ndcg)
            rerank_costs.append(cost)
            latencies_ms.append(elapsed_ms)

            if step % self.report_interval == 0:
                prune_kwargs = dict(
                    step=step,
                    n_total=n_total,
                    obj1=ndcg,          # per-query value for per-query trackers
                    obj2=cost,
                    elapsed_ms=sum(latencies_ms),
                )
                for pruner in self.pruners:
                    pruner.report(**prune_kwargs)
                    if pruner.should_prune():
                        raise optuna.TrialPruned(
                            f"Pruned by {type(pruner).__name__} at step {step}"
                        )

        return TrialObjectives(
            ndcg_at_10=float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            rerank_docs=float(np.mean(rerank_costs)) if rerank_costs else 0.0,
            latency_ms=float(np.mean(latencies_ms)) if latencies_ms else 0.0,
            queries_evaluated=len(ndcg_scores),
        )
