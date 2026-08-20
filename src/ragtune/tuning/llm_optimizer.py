"""
LLM-agent-based optimizer for RAGtune pipeline configuration.

Implements GEPA (Genetic-Pareto) from arxiv 2507.19457:
  - Instance-wise Pareto pool: candidates kept if best on ≥1 evaluation query
  - Pareto-weighted selection: sample parent ∝ queries where it's the best config
  - Two-stage evaluation: minibatch screening → full evaluation only if improved
  - Module-targeted mutation: round-robin focus on one component per iteration
  - System-aware merge: crossover combining best module from each lineage
"""

from __future__ import annotations

import itertools
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import yaml
from pydantic import BaseModel, Field

from ragtune.tuning.evaluator import EvalDataset, EvalQuery, TrialObjectives, ndcg_at_k
from ragtune.tuning.search_space import RAGtuneSearchSpace


# ── Trace diagnostics ─────────────────────────────────────────────────────────

@dataclass
class TraceAggregate:
    avg_pool_size: float
    avg_pct_pool_reranked: float
    feedback_stop_rate: float
    retrieval_skip_rate: float
    avg_rewrite_utility: float
    avg_rerank_errors: float


def _extract_query_trace(output: Any) -> Dict[str, Any]:
    pool_size = 0
    total_reranked = 0
    had_feedback_stop = False
    had_retrieval_skip = False
    rewrite_utility = 0.0
    n_rerank_errors = 0

    for ev in output.trace.events:
        action = ev.action
        details = ev.details
        if action == "pool_init":
            pool_size = details.get("count", 0)
            metrics = details.get("metrics", {})
            rewrite_utility = metrics.get("rewrite_utility_ratio", 0.0)
        elif action == "rerank_batch":
            total_reranked += details.get("count", 0)
        elif action == "feedback_stop":
            had_feedback_stop = True
        elif action == "retrieval_skipped":
            had_retrieval_skip = True
        elif action == "rerank_error":
            n_rerank_errors += 1

    return {
        "pool_size": pool_size,
        "pct_pool_reranked": total_reranked / pool_size if pool_size > 0 else 0.0,
        "feedback_stop": had_feedback_stop,
        "retrieval_skip": had_retrieval_skip,
        "rewrite_utility": rewrite_utility,
        "rerank_errors": n_rerank_errors,
    }


def _aggregate_traces(per_query: List[Dict[str, Any]]) -> TraceAggregate:
    if not per_query:
        return TraceAggregate(0, 0, 0, 0, 0, 0)

    def _mean(key: str) -> float:
        return sum(q[key] for q in per_query) / len(per_query)

    def _rate(key: str) -> float:
        return sum(1 for q in per_query if q[key]) / len(per_query)

    return TraceAggregate(
        avg_pool_size=_mean("pool_size"),
        avg_pct_pool_reranked=_mean("pct_pool_reranked"),
        feedback_stop_rate=_rate("feedback_stop"),
        retrieval_skip_rate=_rate("retrieval_skip"),
        avg_rewrite_utility=_mean("rewrite_utility"),
        avg_rerank_errors=_mean("rerank_errors"),
    )


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    objectives: TrialObjectives
    trace: TraceAggregate
    query_ndcg: Dict[str, float]   # query_id → NDCG@10 (GEPA instance-wise tracking)


@dataclass
class LLMCandidate:
    iteration: int
    params: Dict[str, Any]
    ndcg_at_10: float
    mean_rerank_docs: float
    rationale: str
    trace: Optional[TraceAggregate] = None
    error: Optional[str] = None
    query_ndcg: Dict[str, float] = field(default_factory=dict)
    mutated_module: Optional[str] = None   # which module was targeted this iteration

    def dominates(self, other: LLMCandidate) -> bool:
        if self.error or other.error:
            return False
        return (
            self.ndcg_at_10 >= other.ndcg_at_10
            and self.mean_rerank_docs <= other.mean_rerank_docs
            and (
                self.ndcg_at_10 > other.ndcg_at_10
                or self.mean_rerank_docs < other.mean_rerank_docs
            )
        )


# ── GEPA: Instance-wise Pareto candidate pool ─────────────────────────────────

class GEPACandidatePool:
    """
    GEPA Algorithm 2: maintain candidates that are best on ≥1 evaluation query,
    sample parents proportionally to query coverage.

    This is distinct from objective-space Pareto (NDCG vs cost).  A candidate
    survives here if it achieves the highest NDCG on at least one query — even
    if another candidate dominates it on both aggregate objectives.  This
    preserves diversity and prevents lock-in on a local optimum.
    """

    def __init__(self) -> None:
        self._candidates: List[LLMCandidate] = []
        self._best_per_query: Dict[str, float] = {}

    def add(self, candidate: LLMCandidate) -> None:
        if candidate.error or not candidate.query_ndcg:
            return

        self._candidates.append(candidate)
        for qid, ndcg in candidate.query_ndcg.items():
            if ndcg > self._best_per_query.get(qid, 0.0):
                self._best_per_query[qid] = ndcg

        # Prune: drop candidates never best on any query
        self._candidates = [c for c in self._candidates if self._wins_any(c)]

    def _wins_any(self, c: LLMCandidate) -> bool:
        return any(
            ndcg >= self._best_per_query.get(qid, 0.0) - 1e-6
            for qid, ndcg in c.query_ndcg.items()
        )

    def _coverage(self, c: LLMCandidate) -> int:
        """Number of queries where c achieves the per-query best score."""
        return sum(
            1 for qid, ndcg in c.query_ndcg.items()
            if ndcg >= self._best_per_query.get(qid, 0.0) - 1e-6
        )

    def sample_for_mutation(self, rng: random.Random) -> LLMCandidate:
        """Sample proportionally to query coverage (f[Φ] in GEPA)."""
        if not self._candidates:
            raise ValueError("Pool is empty")
        if len(self._candidates) == 1:
            return self._candidates[0]
        weights = [max(self._coverage(c), 1) for c in self._candidates]
        total = sum(weights)
        r = rng.random() * total
        cumsum = 0.0
        for c, w in zip(self._candidates, weights):
            cumsum += w
            if r <= cumsum:
                return c
        return self._candidates[-1]

    def best(self) -> Optional[LLMCandidate]:
        if not self._candidates:
            return None
        return max(self._candidates, key=lambda c: c.ndcg_at_10)

    def pareto_front(self) -> List[LLMCandidate]:
        """Objective-space Pareto front (NDCG vs cost) over pool candidates."""
        return compute_pareto_front(self._candidates)

    def __len__(self) -> int:
        return len(self._candidates)

    def __iter__(self) -> Iterator[LLMCandidate]:
        return iter(self._candidates)


# ── Objective-space Pareto utility ────────────────────────────────────────────

def compute_pareto_front(candidates: List[LLMCandidate]) -> List[LLMCandidate]:
    """Return the non-dominated subset of candidates (ignoring errored entries)."""
    valid = [c for c in candidates if not c.error]
    pareto = []
    for c in valid:
        if not any(other.dominates(c) for other in valid if other is not c):
            pareto.append(c)
    return pareto


# ── Evaluation: full and minibatch ────────────────────────────────────────────

def _apply_retrieval_overrides(overrides: Optional[Dict[str, Any]]) -> None:
    if overrides:
        from ragtune.utils.config import config as global_cfg
        for k, v in overrides.items():
            global_cfg.set(k, v)


def evaluate_controller_full(
    controller: Any,
    dataset: EvalDataset,
    n_eval_queries: int,
    retrieval_overrides: Optional[Dict[str, Any]] = None,
) -> EvalResult:
    """
    Full evaluation: run controller on n_eval_queries, return per-query NDCG.
    Used for pool updates and final Pareto construction.
    """
    _apply_retrieval_overrides(retrieval_overrides)

    ndcg_scores: List[float] = []
    rerank_costs: List[float] = []
    latencies_ms: List[float] = []
    per_query_traces: List[Dict[str, Any]] = []
    query_ndcg: Dict[str, float] = {}
    n_total = min(n_eval_queries, len(dataset.queries))

    for eq in dataset.iter_queries(limit=n_total):
        t0 = time.monotonic()
        try:
            output = controller.run(eq.query)
        except Exception:
            ndcg_scores.append(0.0)
            rerank_costs.append(0.0)
            latencies_ms.append(0.0)
            query_ndcg[eq.query_id] = 0.0
            continue

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        ranked_ids = [doc.id for doc in output.documents]
        ndcg = ndcg_at_k(ranked_ids, eq.qrels, k=10)
        cost = float(output.final_budget_state.get("rerank_docs", 0.0))

        ndcg_scores.append(ndcg)
        rerank_costs.append(cost)
        latencies_ms.append(elapsed_ms)
        query_ndcg[eq.query_id] = ndcg
        per_query_traces.append(_extract_query_trace(output))

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return EvalResult(
        objectives=TrialObjectives(
            ndcg_at_10=_mean(ndcg_scores),
            rerank_docs=_mean(rerank_costs),
            latency_ms=_mean(latencies_ms),
            queries_evaluated=len(ndcg_scores),
        ),
        trace=_aggregate_traces(per_query_traces),
        query_ndcg=query_ndcg,
    )


# Backward-compat alias used by tests / older callers
evaluate_controller = evaluate_controller_full


def evaluate_controller_minibatch(
    controller: Any,
    dataset: EvalDataset,
    minibatch_qids: Set[str],
    retrieval_overrides: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """
    Minibatch screening (GEPA stage 1): evaluate on a small fixed query subset.
    Returns mean NDCG or None on fatal failure.
    """
    _apply_retrieval_overrides(retrieval_overrides)
    ndcg_scores: List[float] = []
    for eq in dataset.iter_queries():
        if eq.query_id not in minibatch_qids:
            continue
        try:
            output = controller.run(eq.query)
            ranked_ids = [doc.id for doc in output.documents]
            ndcg_scores.append(ndcg_at_k(ranked_ids, eq.qrels, k=10))
        except Exception:
            ndcg_scores.append(0.0)
    if not ndcg_scores:
        return None
    return sum(ndcg_scores) / len(ndcg_scores)


# ── Parameter correlation helper ──────────────────────────────────────────────

_CATEGORICAL_PARAMS = [
    "reranker_type", "reformulator_type", "estimator_type",
    "scheduler_type", "feedback_type",
]


def _compute_param_correlations(
    history: List[LLMCandidate],
) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    valid = [c for c in history if not c.error]
    result: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for param in _CATEGORICAL_PARAMS:
        groups: Dict[str, List[Tuple[float, float]]] = {}
        for c in valid:
            val = str(c.params.get(param, "?"))
            groups.setdefault(val, []).append((c.ndcg_at_10, c.mean_rerank_docs))
        result[param] = {
            val: (
                sum(x[0] for x in pairs) / len(pairs),
                sum(x[1] for x in pairs) / len(pairs),
                len(pairs),
            )
            for val, pairs in groups.items()
            if len(pairs) >= 2
        }
    return result


# ── Module targeting (GEPA round-robin) ──────────────────────────────────────

_MODULE_CYCLE = ["reranker", "scheduler", "estimator", "feedback", "retrieval"]

_MODULE_PARAM_KEYS: Dict[str, List[str]] = {
    "reranker":    ["reranker_type", "ce_model", "monot5_model", "monot5_batch_size"],
    "scheduler":   ["scheduler_type", "scheduler_batch_size", "gd_llm_limit", "gd_ce_limit"],
    "estimator":   ["estimator_type", "similarity_model", "min_reranked_for_regression"],
    "feedback":    ["feedback_type", "budget_stop_token_threshold"],
    "retrieval":   [
        "original_query_depth", "depth_per_reformulation", "max_pool_size",
        "near_duplicate_threshold", "budget_rerank_docs", "budget_reformulations",
        "assembler_max_docs",
    ],
}


# ── Config ────────────────────────────────────────────────────────────────────

class LLMOptimizerConfig(BaseModel):
    name: str = "llm-agent-optimizer"

    # LLM settings
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_history_in_prompt: int = 20

    # Iteration budget (includes startup)
    n_iterations: int = 30
    n_startup: int = 3       # random configs to seed the pool before GEPA loop
    n_minibatch: int = 10    # queries for fast minibatch screening (GEPA stage 1)
    merge_every: int = 10    # perform a merge crossover every N iterations (0 = disabled)

    # Evaluation settings
    n_eval_queries: int = 50
    seed: int = 42

    # Dataset (mirrors TuningStudyConfig.dataset structure)
    dataset: Dict[str, Any] = Field(default_factory=lambda: {"name": "trec-covid"})

    # Output
    output_dir: str = "tuning_results_llm"

    # Search space overrides
    search_space_overrides: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> LLMOptimizerConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ── Main optimizer class ──────────────────────────────────────────────────────

class LLMAgentOptimizer:
    """
    GEPA-style RAGtune configuration optimizer.

    Loop per iteration:
      1. Sample parent from instance-wise Pareto pool (weighted by query coverage).
      2. Select target module (round-robin).
      3. Ask LLM to mutate only the target module of the parent config.
      4. Stage 1 — minibatch screen: evaluate on n_minibatch queries.
         If the new config doesn't improve over the parent's score on those
         queries, skip the full evaluation (saves ~80% of eval cost).
      5. Stage 2 — full evaluation: record per-query NDCG, update pool.
      6. Periodic merge: combine best module from each of the top-2 candidates.
    """

    def __init__(
        self,
        config: LLMOptimizerConfig,
        search_space: Optional[RAGtuneSearchSpace] = None,
    ):
        self.config = config
        self.search_space = search_space or RAGtuneSearchSpace(**config.search_space_overrides)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        fixed_retriever: Any,
        eval_dataset: EvalDataset,
    ) -> List[LLMCandidate]:
        rng = random.Random(self.config.seed)
        pool = GEPACandidatePool()
        history: List[LLMCandidate] = []
        module_iter = itertools.cycle(_MODULE_CYCLE)
        minibatch_qids = self._make_minibatch_qids(eval_dataset)
        iteration = 0

        # ── Startup: seed pool with random configs ────────────────────────────
        n_startup = min(self.config.n_startup, self.config.n_iterations)
        for _ in range(n_startup):
            iteration += 1
            params = self._sample_random(rng)
            candidate = self._full_eval_candidate(
                iteration, params, fixed_retriever, eval_dataset,
                rationale="Random startup configuration.",
                mutated_module="(startup)",
            )
            history.append(candidate)
            pool.add(candidate)

        # ── GEPA loop ─────────────────────────────────────────────────────────
        remaining = self.config.n_iterations - n_startup
        for step in range(remaining):
            iteration += 1
            target_module = next(module_iter)

            # Select parent from pool (Pareto-weighted)
            if len(pool) > 0:
                parent = pool.sample_for_mutation(rng)
            else:
                valid = [c for c in history if not c.error]
                parent = max(valid, key=lambda c: c.ndcg_at_10) if valid else None

            # Propose mutation: only change target_module
            params, rationale = self._propose(history, pool, parent, target_module)

            # ── Stage 1: minibatch screening ──────────────────────────────────
            candidate = self._minibatch_screen_then_eval(
                iteration, params, fixed_retriever, eval_dataset,
                minibatch_qids, parent, rationale, target_module,
            )
            history.append(candidate)
            pool.add(candidate)

            # ── Periodic merge ────────────────────────────────────────────────
            if (
                self.config.merge_every > 0
                and (step + 1) % self.config.merge_every == 0
                and len(pool) >= 2
                and iteration < self.config.n_iterations
            ):
                iteration += 1
                merged_params, merge_rationale = self._merge(pool, rng)
                merged = self._full_eval_candidate(
                    iteration, merged_params, fixed_retriever, eval_dataset,
                    rationale=merge_rationale, mutated_module="(merge)",
                )
                history.append(merged)
                pool.add(merged)

        self._write_pareto_configs(pool.pareto_front())
        return history

    # ── Evaluation helpers ────────────────────────────────────────────────────

    def _make_minibatch_qids(self, dataset: EvalDataset) -> Set[str]:
        queries = list(dataset.iter_queries(limit=self.config.n_eval_queries))
        n = min(self.config.n_minibatch, len(queries))
        return {q.query_id for q in queries[:n]}

    def _full_eval_candidate(
        self,
        iteration: int,
        params: Dict[str, Any],
        fixed_retriever: Any,
        eval_dataset: EvalDataset,
        rationale: str,
        mutated_module: Optional[str],
    ) -> LLMCandidate:
        try:
            controller = self.search_space.build_controller(params, fixed_retriever)
            overrides = self.search_space.to_retrieval_overrides(params)
            result = evaluate_controller_full(
                controller, eval_dataset, self.config.n_eval_queries, overrides,
            )
            return LLMCandidate(
                iteration=iteration,
                params=params,
                ndcg_at_10=result.objectives.ndcg_at_10,
                mean_rerank_docs=result.objectives.rerank_docs,
                rationale=rationale,
                trace=result.trace,
                query_ndcg=result.query_ndcg,
                mutated_module=mutated_module,
            )
        except Exception as exc:
            return LLMCandidate(
                iteration=iteration,
                params=params,
                ndcg_at_10=0.0,
                mean_rerank_docs=float("inf"),
                rationale=rationale,
                error=str(exc),
                mutated_module=mutated_module,
            )

    def _minibatch_screen_then_eval(
        self,
        iteration: int,
        params: Dict[str, Any],
        fixed_retriever: Any,
        eval_dataset: EvalDataset,
        minibatch_qids: Set[str],
        parent: Optional[LLMCandidate],
        rationale: str,
        target_module: str,
    ) -> LLMCandidate:
        # Parent's score on the minibatch queries (from cached query_ndcg)
        parent_mini = 0.0
        if parent is not None and parent.query_ndcg:
            scores = [parent.query_ndcg[qid] for qid in minibatch_qids if qid in parent.query_ndcg]
            parent_mini = sum(scores) / len(scores) if scores else 0.0

        # Stage 1: minibatch eval
        try:
            controller = self.search_space.build_controller(params, fixed_retriever)
            overrides = self.search_space.to_retrieval_overrides(params)
            mini_ndcg = evaluate_controller_minibatch(
                controller, eval_dataset, minibatch_qids, overrides,
            )
        except Exception as exc:
            return LLMCandidate(
                iteration=iteration, params=params, ndcg_at_10=0.0,
                mean_rerank_docs=float("inf"), rationale=rationale,
                error=f"Minibatch build/eval failed: {exc}",
                mutated_module=target_module,
            )

        # Allow a small tolerance so near-ties proceed to full eval
        if mini_ndcg is None or mini_ndcg < parent_mini - 0.02:
            return LLMCandidate(
                iteration=iteration, params=params,
                ndcg_at_10=mini_ndcg or 0.0,
                mean_rerank_docs=float("inf"), rationale=rationale,
                error="Minibatch screen failed — skipped full evaluation",
                mutated_module=target_module,
            )

        # Stage 2: full evaluation
        return self._full_eval_candidate(
            iteration, params, fixed_retriever, eval_dataset,
            rationale=rationale, mutated_module=target_module,
        )

    # ── Proposal ──────────────────────────────────────────────────────────────

    def _propose(
        self,
        history: List[LLMCandidate],
        pool: GEPACandidatePool,
        parent: Optional[LLMCandidate],
        target_module: str,
    ) -> Tuple[Dict[str, Any], str]:
        import litellm  # type: ignore

        system = self._build_system_prompt(target_module)
        user = self._build_user_message(history, pool, parent, target_module)

        response = litellm.completion(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.config.temperature,
        )
        raw = response.choices[0].message.content or "{}"
        return self._parse_response(raw, parent)

    def _build_system_prompt(self, target_module: str) -> str:
        sp = self.search_space
        str_batch_sizes = [str(b) for b in sp.monot5_batch_sizes]
        module_keys = _MODULE_PARAM_KEYS.get(target_module, [])
        return (
            "You are an expert at optimizing RAG pipeline configurations.\n"
            "Goal: find configurations that Pareto-dominate existing ones on:\n"
            "  - MAXIMIZE ndcg_at_10  (retrieval quality, 0–1)\n"
            "  - MINIMIZE mean_rerank_docs  (computational cost)\n\n"
            f"## Your task this iteration: improve the [{target_module.upper()}] module only.\n"
            f"Module parameters to change: {module_keys}\n"
            "All other parameters must be copied exactly from the parent config.\n\n"
            "## Full search space (for reference):\n\n"
            "### Component types:\n"
            f"reranker_type:     {sp.reranker_types}\n"
            f"reformulator_type: {sp.reformulator_types}\n"
            f"estimator_type:    {sp.estimator_types}\n"
            f"scheduler_type:    {sp.scheduler_types}\n"
            f"feedback_type:     {sp.feedback_types}\n\n"
            "### Conditional sub-parameters:\n"
            f'- reranker_type=="cross-encoder" → ce_model ∈ {sp.ce_models}\n'
            f'- reranker_type=="monot5"        → monot5_model ∈ {sp.monot5_models}, '
            f'monot5_batch_size ∈ {str_batch_sizes}\n'
            f'- estimator_type=="similarity"   → similarity_model ∈ {sp.similarity_models}\n'
            f'- scheduler_type=="graceful-degradation" → '
            f'gd_llm_limit ∈ [{sp.gd_llm_limit_range[0]}–{sp.gd_llm_limit_range[1]}], '
            f'gd_ce_limit ∈ [{sp.gd_ce_limit_range[0]}–{sp.gd_ce_limit_range[1]}]\n'
            f'- feedback_type=="budget-stop"   → budget_stop_token_threshold ∈ '
            f'[{sp.budget_stop_token_threshold_range[0]}–{sp.budget_stop_token_threshold_range[1]}]\n\n'
            "### Always-required numerical parameters:\n"
            f"original_query_depth:       int   ∈ [{sp.original_query_depth_range[0]}–{sp.original_query_depth_range[1]}]\n"
            f"depth_per_reformulation:    int   ∈ [{sp.depth_per_reformulation_range[0]}–{sp.depth_per_reformulation_range[1]}]\n"
            f"max_pool_size:              int   ∈ [{sp.max_pool_size_range[0]}–{sp.max_pool_size_range[1]}]\n"
            f"near_duplicate_threshold:   float ∈ [{sp.near_duplicate_threshold_range[0]}–{sp.near_duplicate_threshold_range[1]}]\n"
            f"scheduler_batch_size:       int   ∈ [{sp.scheduler_batch_size_range[0]}–{sp.scheduler_batch_size_range[1]}]\n"
            f"assembler_max_docs:         int   ∈ [{sp.assembler_max_docs_range[0]}–{sp.assembler_max_docs_range[1]}]\n"
            f"budget_rerank_docs:         int   ∈ [{sp.budget_rerank_docs_range[0]}–{sp.budget_rerank_docs_range[1]}]\n"
            f"budget_reformulations:      int   ∈ [{sp.budget_reformulations_range[0]}–{sp.budget_reformulations_range[1]}]\n\n"
            "## Diagnostic signals in each result:\n"
            "- avg_pool_size: candidate docs in pool after retrieval\n"
            "- pct_pool_reranked: fraction of pool reranked (budget-limited)\n"
            "- feedback_stop_rate: fraction of queries with early stopping\n"
            "- retrieval_skip_rate: fraction where budget blocked supplemental retrieval\n"
            "- rewrite_utility: fraction of pool docs found only via reformulated queries\n\n"
            "## Output format (JSON only, no markdown):\n"
            '{"rationale": "<reasoning referencing the target module and diagnostics>", '
            '"params": {<all required params, most copied from parent>}}'
        )

    def _build_user_message(
        self,
        history: List[LLMCandidate],
        pool: GEPACandidatePool,
        parent: Optional[LLMCandidate],
        target_module: str,
    ) -> str:
        sections: List[str] = []

        # ── Parent config ─────────────────────────────────────────────────────
        if parent is not None:
            t = parent.trace
            diag = (
                f"pool={t.avg_pool_size:.0f} reranked={t.avg_pct_pool_reranked:.0%} "
                f"fb_stop={t.feedback_stop_rate:.0%} rewrite_util={t.avg_rewrite_utility:.0%}"
            ) if t else "no trace"
            sections.append(
                f"## Parent config (iter {parent.iteration}) — "
                f"NDCG={parent.ndcg_at_10:.3f}  cost={parent.mean_rerank_docs:.1f}\n"
                f"Diagnostics: {diag}\n"
                f"Parameters:\n{json.dumps(parent.params, indent=2)}"
            )
            sections.append(
                f"\n## Your task: mutate ONLY the [{target_module.upper()}] parameters.\n"
                f"Keys to change: {_MODULE_PARAM_KEYS.get(target_module, [])}\n"
                "Keep all other parameters identical to the parent."
            )
        else:
            sections.append("No history yet — this is the first iteration.")
            sections.append("Propose a balanced starting configuration.")

        # ── Pool coverage summary ─────────────────────────────────────────────
        if len(pool) > 0:
            pool_list = sorted(pool, key=lambda c: -c.ndcg_at_10)
            sections.append(f"\n## Instance-wise Pareto pool ({len(pool)} candidates):")
            for c in pool_list[:5]:
                t = c.trace
                mod_tag = f" [changed: {c.mutated_module}]" if c.mutated_module else ""
                diag = (
                    f"pool={t.avg_pool_size:.0f} reranked={t.avg_pct_pool_reranked:.0%}"
                ) if t else ""
                sections.append(
                    f"  iter={c.iteration:3d}{mod_tag}  ndcg={c.ndcg_at_10:.3f}"
                    f"  cost={c.mean_rerank_docs:5.1f}  {diag}"
                )

        # ── Parameter correlation table ────────────────────────────────────────
        correlations = _compute_param_correlations(history)
        if any(v for v in correlations.values()):
            sections.append(f"\n## Parameter trends ({len([c for c in history if not c.error])} successful runs):")
            for param, vals in correlations.items():
                if not vals:
                    continue
                sorted_vals = sorted(vals.items(), key=lambda x: -x[1][0])
                parts = [f"{v}→ndcg={nd:.3f}/cost={co:.1f}(n={n})"
                         for v, (nd, co, n) in sorted_vals]
                sections.append(f"  {param}: {',  '.join(parts)}")

        # ── Recent history with rationale ─────────────────────────────────────
        shown = history[-self.config.max_history_in_prompt:]
        sections.append(f"\n## Recent history (last {len(shown)}):")
        sections.append(
            "| iter | ndcg  | cost  | reranker     | sched                | pool | rnk% | module   | reasoning |"
        )
        sections.append(
            "|------|-------|-------|--------------|----------------------|------|------|----------|-----------|"
        )
        for c in shown:
            t = c.trace
            pool_s = f"{t.avg_pool_size:.0f}" if t else "—"
            rnk_s = f"{t.avg_pct_pool_reranked:.0%}" if t else "—"
            mod_s = (c.mutated_module or "?")[:8]
            rat_s = (c.rationale or "")[:45].replace("|", "/")
            err_tag = " [SKIP]" if c.error and "screen" in (c.error or "") else (
                " [ERR]" if c.error else ""
            )
            sections.append(
                f"| {c.iteration:4d} | {c.ndcg_at_10:.3f} | {c.mean_rerank_docs:5.1f} | "
                f"{c.params.get('reranker_type','?'):12s} | "
                f"{c.params.get('scheduler_type','?'):20s} | "
                f"{pool_s:4s} | {rnk_s:4s} | {mod_s:8s} | {rat_s}{err_tag} |"
            )

        # ── Diagnostic guide (target-module focused) ──────────────────────────
        sections.append(f"\n## Diagnostic hints for [{target_module.upper()}]:")
        if target_module == "reranker":
            sections.append("- pct_pool_reranked < 30%: budget_rerank_docs too tight; consider raising it in the retrieval turn")
            sections.append("- noop consistently wins: BM25 pool quality is already high; cross-encoder adds noise on small pools")
            sections.append("- cross-encoder wins at depth>100: neural reranker rescues BM25-misranked relevant docs")
        elif target_module == "scheduler":
            sections.append("- active-learning: ranks by estimator score; good when estimator is reliable")
            sections.append("- graceful-degradation: falls back gracefully when reranker quality is uncertain")
        elif target_module == "estimator":
            sections.append("- similarity: adds sentence-encoder cost per candidate but improves batch selection")
            sections.append("- baseline: cheapest; good when pool is already well-ordered by BM25")
        elif target_module == "feedback":
            sections.append("- budget-stop with low threshold: exits early when budget nearly spent")
            sections.append("- feedback_stop_rate>70%: feedback is cutting runs short before docs are reranked")
        elif target_module == "retrieval":
            sections.append("- original_query_depth controls how many BM25 docs enter the pool")
            sections.append("- raising depth helps cross-encoder reranking; noop is unaffected by depth >10")
            sections.append("- budget_rerank_docs must be ≤ max_pool_size to avoid wasted budget")

        sections.append("\nRespond with JSON only (rationale + params).")
        return "\n".join(sections)

    # ── Merge crossover ───────────────────────────────────────────────────────

    def _merge(
        self, pool: GEPACandidatePool, rng: random.Random
    ) -> Tuple[Dict[str, Any], str]:
        """
        GEPA system-aware merge: combine best module from each of the top-2
        pool candidates, prioritising whichever was specialized for that module.
        """
        sorted_pool = sorted(pool, key=lambda c: -c.ndcg_at_10)
        a, b = sorted_pool[0], sorted_pool[1]

        merged: Dict[str, Any] = {}
        sources: List[str] = []

        for module, param_keys in _MODULE_PARAM_KEYS.items():
            # Prefer the candidate that was last specialized for this module
            a_specialized = a.mutated_module == module
            b_specialized = b.mutated_module == module

            if a_specialized and not b_specialized:
                src = a
            elif b_specialized and not a_specialized:
                src = b
            else:
                # No specialization signal: use the candidate with higher NDCG
                src = a if a.ndcg_at_10 >= b.ndcg_at_10 else b

            for k in param_keys:
                if k in src.params:
                    merged[k] = src.params[k]
            sources.append(f"{module}←iter{src.iteration}")

        # Fill any remaining params from the best candidate
        for k in a.params:
            if k not in merged:
                merged[k] = a.params[k]

        return merged, f"Merge crossover: {', '.join(sources)}"

    # ── Random sampling for startup ───────────────────────────────────────────

    def _sample_random(self, rng: random.Random) -> Dict[str, Any]:
        sp = self.search_space
        str_batch = [str(b) for b in sp.monot5_batch_sizes]
        raw: Dict[str, Any] = {
            "reranker_type":     rng.choice(sp.reranker_types),
            "reformulator_type": rng.choice(sp.reformulator_types),
            "estimator_type":    rng.choice(sp.estimator_types),
            "scheduler_type":    rng.choice(sp.scheduler_types),
            "feedback_type":     rng.choice(sp.feedback_types),
            "original_query_depth":      rng.randint(*sp.original_query_depth_range),
            "depth_per_reformulation":   rng.randint(*sp.depth_per_reformulation_range),
            "max_pool_size":             rng.randint(*sp.max_pool_size_range),
            "near_duplicate_threshold":  rng.uniform(*sp.near_duplicate_threshold_range),
            "scheduler_batch_size":      rng.randint(*sp.scheduler_batch_size_range),
            "assembler_max_docs":        rng.randint(*sp.assembler_max_docs_range),
            "budget_rerank_docs":        rng.randint(*sp.budget_rerank_docs_range),
            "budget_reformulations":     rng.randint(*sp.budget_reformulations_range),
            "ce_model":                  rng.choice(sp.ce_models),
            "monot5_model":              rng.choice(sp.monot5_models),
            "monot5_batch_size":         rng.choice(str_batch),
            "reformulator_model":        rng.choice(sp.reformulator_models),
            "reformulator_n_variants":   rng.randint(*sp.reformulator_n_variants_range),
            "similarity_model":          rng.choice(sp.similarity_models),
            "min_reranked_for_regression": rng.randint(*sp.min_reranked_for_regression_range),
            "gd_llm_limit":              rng.randint(*sp.gd_llm_limit_range),
            "gd_ce_limit":               rng.randint(*sp.gd_ce_limit_range),
            "budget_stop_token_threshold": rng.uniform(*sp.budget_stop_token_threshold_range),
        }
        return self._validate_and_fix_params(raw)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_response(
        self, raw: str, parent: Optional[LLMCandidate]
    ) -> Tuple[Dict[str, Any], str]:
        text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        data: Dict[str, Any] = {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        rationale = data.get("rationale", "(no rationale provided)")
        raw_params = data.get("params", {})

        # Start from parent params so non-targeted fields are preserved exactly
        base = dict(parent.params) if parent is not None else {}
        base.update(raw_params)

        return self._validate_and_fix_params(base), rationale

    def _validate_and_fix_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        sp = self.search_space

        def clamp_int(key: str, lo: int, hi: int, default: int) -> int:
            try:
                return max(lo, min(hi, int(p.get(key, default))))
            except (TypeError, ValueError):
                return default

        def clamp_float(key: str, lo: float, hi: float, default: float) -> float:
            try:
                return max(lo, min(hi, float(p.get(key, default))))
            except (TypeError, ValueError):
                return default

        def pick(key: str, choices: List[Any], default: Any) -> Any:
            val = p.get(key, default)
            return val if val in choices else default

        str_batch_sizes = [str(b) for b in sp.monot5_batch_sizes]

        def fix_batch_size(val: Any) -> str:
            sval = str(val) if not isinstance(val, str) else val
            if sval in str_batch_sizes:
                return sval
            try:
                target = int(float(str(val)))
                nearest = min(sp.monot5_batch_sizes, key=lambda x: abs(x - target))
                return str(nearest)
            except (TypeError, ValueError):
                return str_batch_sizes[2]

        out: Dict[str, Any] = {}
        out["reranker_type"] = pick("reranker_type", sp.reranker_types, sp.reranker_types[0])
        out["reformulator_type"] = pick("reformulator_type", sp.reformulator_types, sp.reformulator_types[0])
        out["estimator_type"] = pick("estimator_type", sp.estimator_types, sp.estimator_types[0])
        out["scheduler_type"] = pick("scheduler_type", sp.scheduler_types, sp.scheduler_types[0])
        out["feedback_type"] = pick("feedback_type", sp.feedback_types, sp.feedback_types[0])
        out["ce_model"] = pick("ce_model", sp.ce_models, sp.ce_models[0])
        out["monot5_model"] = pick("monot5_model", sp.monot5_models, sp.monot5_models[0])
        out["monot5_batch_size"] = fix_batch_size(p.get("monot5_batch_size", str_batch_sizes[2]))
        out["reformulator_model"] = pick("reformulator_model", sp.reformulator_models, sp.reformulator_models[0])
        out["reformulator_n_variants"] = clamp_int("reformulator_n_variants", *sp.reformulator_n_variants_range, 3)
        out["similarity_model"] = pick("similarity_model", sp.similarity_models, sp.similarity_models[0])
        out["min_reranked_for_regression"] = clamp_int("min_reranked_for_regression", *sp.min_reranked_for_regression_range, 3)
        out["gd_llm_limit"] = clamp_int("gd_llm_limit", *sp.gd_llm_limit_range, 3)
        out["gd_ce_limit"] = clamp_int("gd_ce_limit", *sp.gd_ce_limit_range, 10)
        out["budget_stop_token_threshold"] = clamp_float(
            "budget_stop_token_threshold", *sp.budget_stop_token_threshold_range, 0.9
        )
        out["original_query_depth"] = clamp_int("original_query_depth", *sp.original_query_depth_range, 10)
        out["depth_per_reformulation"] = clamp_int("depth_per_reformulation", *sp.depth_per_reformulation_range, 5)
        out["max_pool_size"] = clamp_int("max_pool_size", *sp.max_pool_size_range, 50)
        out["near_duplicate_threshold"] = clamp_float(
            "near_duplicate_threshold", *sp.near_duplicate_threshold_range, 0.8
        )
        out["scheduler_batch_size"] = clamp_int("scheduler_batch_size", *sp.scheduler_batch_size_range, 5)
        out["assembler_max_docs"] = clamp_int("assembler_max_docs", *sp.assembler_max_docs_range, 10)
        out["budget_rerank_docs"] = clamp_int("budget_rerank_docs", *sp.budget_rerank_docs_range, 50)
        out["budget_reformulations"] = clamp_int("budget_reformulations", *sp.budget_reformulations_range, 1)
        return out

    # ── Output ────────────────────────────────────────────────────────────────

    def _write_pareto_configs(self, pareto_front: List[LLMCandidate]) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for c in pareto_front:
            pipeline_dict = self.search_space.to_pipeline_dict(c.params)
            fname = (
                f"pareto_iter_{c.iteration}"
                f"_ndcg{c.ndcg_at_10:.3f}"
                f"_cost{c.mean_rerank_docs:.0f}"
                ".yaml"
            )
            with open(out_dir / fname, "w") as f:
                yaml.dump({"pipeline": pipeline_dict}, f, sort_keys=False)
