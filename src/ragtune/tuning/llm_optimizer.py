"""
LLM-agent-based optimizer for RAGtune pipeline configuration.

Replaces Bayesian (Optuna TPE) search with an LLM reasoning loop inspired by:
  - GEPA (arxiv 2507.19457): trace-reflection → mutation proposals
  - TextGrad (arxiv 2406.07496): parameters as text variables, LLM as gradient

Each iteration the agent:
  1. Observes the full evaluation history and current Pareto front.
  2. Reflects on which parameter patterns drive quality vs. cost.
  3. Proposes a new pipeline configuration as structured JSON.
  4. Evaluates the proposed config and updates the Pareto front.

The agent calls are made via litellm, which already ships as a dependency.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

from ragtune.tuning.evaluator import EvalDataset, TrialObjectives, ndcg_at_k
from ragtune.tuning.search_space import RAGtuneSearchSpace


# ── Trace diagnostics ─────────────────────────────────────────────────────────

@dataclass
class TraceAggregate:
    """
    Per-trial execution diagnostics aggregated across all eval queries.

    These signals tell the LLM *why* a config performed the way it did —
    not just the aggregate NDCG/cost numbers.
    """
    avg_pool_size: float          # avg docs in candidate pool after retrieval
    avg_pct_pool_reranked: float  # avg fraction of pool that got reranked (0–1)
    feedback_stop_rate: float     # fraction of queries that triggered early stop
    retrieval_skip_rate: float    # fraction with supplemental retrieval blocked by budget
    avg_rewrite_utility: float    # avg fraction of pool docs found only by reformulated queries
    avg_rerank_errors: float      # avg rerank errors per query


def _extract_query_trace(output: Any) -> Dict[str, Any]:
    """Extract scalar signals from one ControllerOutput."""
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
        vals = [q[key] for q in per_query]
        return sum(vals) / len(vals)

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


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Return value of evaluate_controller — objectives + execution diagnostics."""
    objectives: TrialObjectives
    trace: TraceAggregate


# ── Domain objects ────────────────────────────────────────────────────────────

@dataclass
class LLMCandidate:
    """One evaluated pipeline configuration."""
    iteration: int
    params: Dict[str, Any]
    ndcg_at_10: float
    mean_rerank_docs: float
    rationale: str
    trace: Optional[TraceAggregate] = None
    error: Optional[str] = None

    def dominates(self, other: LLMCandidate) -> bool:
        """True if self Pareto-dominates other (higher NDCG, lower cost, strictly one)."""
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


class LLMOptimizerConfig(BaseModel):
    """Configuration for the LLM-agent-based optimizer."""
    name: str = "llm-agent-optimizer"

    # LLM settings
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_history_in_prompt: int = 20

    # Iteration budget
    n_iterations: int = 30

    # Evaluation settings (mirrors TuningStudyConfig fields used by evaluator)
    n_eval_queries: int = 50
    seed: int = 42

    # Dataset (mirrors TuningStudyConfig.dataset structure)
    dataset: Dict[str, Any] = Field(default_factory=lambda: {"name": "trec-covid"})

    # Output
    output_dir: str = "tuning_results_llm"

    # Search space overrides (same semantics as TuningStudyConfig)
    search_space_overrides: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> LLMOptimizerConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ── Pareto utilities ──────────────────────────────────────────────────────────

def compute_pareto_front(candidates: List[LLMCandidate]) -> List[LLMCandidate]:
    """Return the non-dominated subset of candidates (ignoring errored entries)."""
    valid = [c for c in candidates if not c.error]
    pareto = []
    for c in valid:
        if not any(other.dominates(c) for other in valid if other is not c):
            pareto.append(c)
    return pareto


# ── Standalone evaluator (no Optuna trial required) ───────────────────────────

def evaluate_controller(
    controller: Any,
    dataset: EvalDataset,
    n_eval_queries: int,
    retrieval_overrides: Optional[Dict[str, Any]] = None,
) -> EvalResult:
    """
    Evaluate a controller on an EvalDataset without Optuna (no pruning).

    Returns both performance objectives and per-trial execution diagnostics
    aggregated across queries (pool size, budget exhaustion, etc.).
    """
    if retrieval_overrides:
        from ragtune.utils.config import config as global_cfg
        for k, v in retrieval_overrides.items():
            global_cfg.set(k, v)

    ndcg_scores: List[float] = []
    rerank_costs: List[float] = []
    latencies_ms: List[float] = []
    per_query_traces: List[Dict[str, Any]] = []
    n_total = min(n_eval_queries, len(dataset.queries))

    for eq in dataset.iter_queries(limit=n_total):
        t0 = time.monotonic()
        try:
            output = controller.run(eq.query)
        except Exception:
            ndcg_scores.append(0.0)
            rerank_costs.append(0.0)
            latencies_ms.append(0.0)
            continue

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        ranked_ids = [doc.id for doc in output.documents]
        ndcg = ndcg_at_k(ranked_ids, eq.qrels, k=10)
        cost = float(output.final_budget_state.get("rerank_docs", 0.0))

        ndcg_scores.append(ndcg)
        rerank_costs.append(cost)
        latencies_ms.append(elapsed_ms)
        per_query_traces.append(_extract_query_trace(output))

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    objectives = TrialObjectives(
        ndcg_at_10=_mean(ndcg_scores),
        rerank_docs=_mean(rerank_costs),
        latency_ms=_mean(latencies_ms),
        queries_evaluated=len(ndcg_scores),
    )
    trace_agg = _aggregate_traces(per_query_traces)
    return EvalResult(objectives=objectives, trace=trace_agg)


# ── Parameter correlation helper ──────────────────────────────────────────────

_CATEGORICAL_PARAMS = [
    "reranker_type",
    "reformulator_type",
    "estimator_type",
    "scheduler_type",
    "feedback_type",
]


def _compute_param_correlations(
    history: List[LLMCandidate],
) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    """
    For each categorical param, return {value: (mean_ndcg, mean_cost, n)} from
    completed (non-errored) runs.  Only values with ≥ 2 observations are included.
    """
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


# ── Main optimizer class ──────────────────────────────────────────────────────

class LLMAgentOptimizer:
    """
    Reflection-based pipeline optimizer driven by an LLM agent.

    Context passed to the LLM each iteration:
      1. Search space definition (valid values, ranges) — system prompt, static.
      2. Top-3 by NDCG and top-3 by cost from all history.
      3. Parameter correlation table (mean NDCG/cost per categorical value).
      4. Current Pareto front with trace diagnostics.
      5. Chronological last-N history with per-run trace diagnostics and the
         agent's own prior rationale — enabling self-reflection.

    Parameters
    ----------
    config : LLMOptimizerConfig
    search_space : RAGtuneSearchSpace, optional
        If not provided, one is built from config.search_space_overrides.
    """

    def __init__(
        self,
        config: LLMOptimizerConfig,
        search_space: Optional[RAGtuneSearchSpace] = None,
    ):
        self.config = config
        self.search_space = search_space or RAGtuneSearchSpace(**config.search_space_overrides)

    def run(
        self,
        fixed_retriever: Any,
        eval_dataset: EvalDataset,
    ) -> List[LLMCandidate]:
        """
        Run the optimization loop.

        Returns the full history of evaluated candidates (not just the Pareto
        front) so callers can audit the agent's reasoning trajectory.
        """
        history: List[LLMCandidate] = []
        pareto_front: List[LLMCandidate] = []

        for i in range(1, self.config.n_iterations + 1):
            params, rationale = self._propose(history, pareto_front)

            try:
                controller = self.search_space.build_controller(params, fixed_retriever)
                retrieval_overrides = self.search_space.to_retrieval_overrides(params)
                result = evaluate_controller(
                    controller,
                    eval_dataset,
                    self.config.n_eval_queries,
                    retrieval_overrides,
                )
                candidate = LLMCandidate(
                    iteration=i,
                    params=params,
                    ndcg_at_10=result.objectives.ndcg_at_10,
                    mean_rerank_docs=result.objectives.rerank_docs,
                    rationale=rationale,
                    trace=result.trace,
                )
            except Exception as exc:
                candidate = LLMCandidate(
                    iteration=i,
                    params=params,
                    ndcg_at_10=0.0,
                    mean_rerank_docs=float("inf"),
                    rationale=rationale,
                    error=str(exc),
                )

            history.append(candidate)
            pareto_front = compute_pareto_front(history)

        self._write_pareto_configs(pareto_front)
        return history

    # ── Proposal ──────────────────────────────────────────────────────────────

    def _propose(
        self,
        history: List[LLMCandidate],
        pareto_front: List[LLMCandidate],
    ) -> Tuple[Dict[str, Any], str]:
        """Ask the LLM to propose the next configuration. Returns (params, rationale)."""
        import litellm  # type: ignore

        system = self._build_system_prompt()
        user = self._build_user_message(history, pareto_front)

        response = litellm.completion(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.config.temperature,
        )
        raw = response.choices[0].message.content or "{}"
        return self._parse_response(raw)

    def _build_system_prompt(self) -> str:
        sp = self.search_space
        str_batch_sizes = [str(b) for b in sp.monot5_batch_sizes]
        return (
            "You are an expert at optimizing RAG pipeline configurations.\n"
            "Your goal: find configurations that Pareto-dominate existing ones on two objectives:\n"
            "  - MAXIMIZE ndcg_at_10  (retrieval quality, 0–1 scale)\n"
            "  - MINIMIZE mean_rerank_docs  (computational cost)\n\n"
            "## Search Space\n\n"
            "### Main component types (choose exactly one from each):\n"
            f"reranker_type:     {sp.reranker_types}\n"
            f"reformulator_type: {sp.reformulator_types}\n"
            f"estimator_type:    {sp.estimator_types}\n"
            f"scheduler_type:    {sp.scheduler_types}\n"
            f"feedback_type:     {sp.feedback_types}\n\n"
            "### Conditional sub-parameters:\n"
            f'- reranker_type=="cross-encoder" → ce_model ∈ {sp.ce_models}\n'
            f'- reranker_type=="monot5"        → monot5_model ∈ {sp.monot5_models},\n'
            f'                                    monot5_batch_size ∈ {str_batch_sizes}\n'
            f'- reformulator_type in ["llm_rewrite","reformir"] → reformulator_model ∈ {sp.reformulator_models}\n'
            f'- reformulator_type=="reformir"  → reformulator_n_variants ∈ '
            f'[{sp.reformulator_n_variants_range[0]}–{sp.reformulator_n_variants_range[1]}]\n'
            f'- estimator_type=="similarity"   → similarity_model ∈ {sp.similarity_models}\n'
            f'- estimator_type=="reformir"     → min_reranked_for_regression ∈ '
            f'[{sp.min_reranked_for_regression_range[0]}–{sp.min_reranked_for_regression_range[1]}]\n'
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
            "## Diagnostic signals (present in each evaluation result):\n"
            "- avg_pool_size: how many candidate docs were in the pool after retrieval\n"
            "- pct_pool_reranked: fraction of pool that got processed by the reranker (budget limited)\n"
            "- feedback_stop_rate: fraction of queries where early-stopping fired\n"
            "- retrieval_skip_rate: fraction where budget ran out before supplemental retrieval\n"
            "- rewrite_utility: fraction of pool docs found only via reformulated queries (0=reformulation added nothing)\n\n"
            "## Output format (respond with valid JSON only, no markdown):\n"
            '{\n'
            '  "rationale": "<step-by-step reasoning referencing specific patterns you observe>",\n'
            '  "params": {\n'
            '    "reranker_type": "...",\n'
            '    ... (all required params) ...\n'
            '  }\n'
            '}'
        )

    def _build_user_message(
        self,
        history: List[LLMCandidate],
        pareto_front: List[LLMCandidate],
    ) -> str:
        if not history:
            return (
                "No history yet — this is iteration 1.\n"
                "Propose a well-balanced starting configuration that includes all required parameters.\n"
                "Respond with JSON only."
            )

        sections: List[str] = []

        # ── 1. Top configs ────────────────────────────────────────────────────
        valid = [c for c in history if not c.error]
        if valid:
            top_ndcg = sorted(valid, key=lambda c: -c.ndcg_at_10)[:3]
            top_cheap = sorted(valid, key=lambda c: c.mean_rerank_docs)[:3]

            def _cfg_row(c: LLMCandidate) -> str:
                t = c.trace
                diag = (
                    f"pool={t.avg_pool_size:.0f} reranked={t.avg_pct_pool_reranked:.0%} "
                    f"fb_stop={t.feedback_stop_rate:.0%} rewrite_util={t.avg_rewrite_utility:.0%}"
                ) if t else "no trace"
                return (
                    f"  iter={c.iteration:3d} ndcg={c.ndcg_at_10:.3f} cost={c.mean_rerank_docs:5.1f} "
                    f"reranker={c.params.get('reranker_type')} sched={c.params.get('scheduler_type')} "
                    f"| {diag}"
                )

            sections.append("## Top configurations by NDCG@10:")
            sections.extend(_cfg_row(c) for c in top_ndcg)
            sections.append("\n## Most efficient configurations (lowest cost):")
            sections.extend(_cfg_row(c) for c in top_cheap)

        # ── 2. Parameter correlation table ────────────────────────────────────
        correlations = _compute_param_correlations(history)
        has_corr = any(v for v in correlations.values())
        if has_corr:
            n_valid = len(valid)
            sections.append(f"\n## Parameter trends (from {n_valid} successful runs, ≥2 obs per value):")
            for param, vals in correlations.items():
                if not vals:
                    continue
                sorted_vals = sorted(vals.items(), key=lambda x: -x[1][0])
                parts = [f"{v}→ndcg={ndcg:.3f}/cost={cost:.1f}(n={n})"
                         for v, (ndcg, cost, n) in sorted_vals]
                sections.append(f"  {param}: {',  '.join(parts)}")

        # ── 3. Pareto front ───────────────────────────────────────────────────
        if pareto_front:
            sections.append("\n## Current Pareto front:")
            for c in sorted(pareto_front, key=lambda x: -x.ndcg_at_10):
                t = c.trace
                diag = (
                    f"pool={t.avg_pool_size:.0f} reranked={t.avg_pct_pool_reranked:.0%} "
                    f"fb_stop={t.feedback_stop_rate:.0%} rewrite_util={t.avg_rewrite_utility:.0%} "
                    f"skip={t.retrieval_skip_rate:.0%}"
                ) if t else ""
                sections.append(
                    f"  iter={c.iteration:3d}  ndcg={c.ndcg_at_10:.3f}  cost={c.mean_rerank_docs:5.1f}"
                    f"  budget_rerank_docs={c.params.get('budget_rerank_docs')}"
                    f"  {diag}"
                )
        else:
            sections.append("\nPareto front is empty (all runs errored).")

        # ── 4. Chronological history with rationale replay ────────────────────
        shown = history[-self.config.max_history_in_prompt:]
        sections.append(f"\n## Recent history (last {len(shown)} of {len(history)} iterations):")
        sections.append(
            "| iter | ndcg  | cost  | reranker     | sched                | pool | rnk% | fb%  | util% | your reasoning |"
        )
        sections.append(
            "|------|-------|-------|--------------|----------------------|------|------|------|-------|----------------|"
        )
        for c in shown:
            t = c.trace
            if t:
                pool_s = f"{t.avg_pool_size:.0f}"
                rnk_s = f"{t.avg_pct_pool_reranked:.0%}"
                fb_s = f"{t.feedback_stop_rate:.0%}"
                util_s = f"{t.avg_rewrite_utility:.0%}"
            else:
                pool_s = rnk_s = fb_s = util_s = "—"
            rationale_snip = (c.rationale or "")[:55].replace("|", "/")
            err_tag = " [ERR]" if c.error else ""
            sections.append(
                f"| {c.iteration:4d} | {c.ndcg_at_10:.3f} | {c.mean_rerank_docs:5.1f} | "
                f"{c.params.get('reranker_type','?'):12s} | "
                f"{c.params.get('scheduler_type','?'):20s} | "
                f"{pool_s:4s} | {rnk_s:4s} | {fb_s:4s} | {util_s:5s} | "
                f"{rationale_snip}{err_tag} |"
            )

        # ── 5. Diagnosis hints ────────────────────────────────────────────────
        sections.append("\n## Diagnostic interpretation guide:")
        sections.append(
            "- pct_pool_reranked < 50%: budget is too tight for pool size; raise budget_rerank_docs OR lower original_query_depth"
        )
        sections.append(
            "- feedback_stop_rate > 50%: early-stopping fires frequently; the feedback mechanism may be cutting runs short"
        )
        sections.append(
            "- rewrite_utility ≈ 0%: reformulation added no new docs; reformulator_type=identity is as effective but cheaper"
        )
        sections.append(
            "- rewrite_utility > 30%: reformulation is finding unique docs; reformulator with budget_reformulations ≥ 2 could help"
        )

        # ── 6. Task ───────────────────────────────────────────────────────────
        sections.append("\n## Your task:")
        sections.append("1. Read the diagnostic signals to understand WHY configs underperform.")
        sections.append("2. Note which parameter values your own rationale predicted would help — did they?")
        sections.append("3. Propose ONE new configuration that addresses a specific weakness you observe.")
        sections.append("\nRespond with JSON only (rationale + params).")

        return "\n".join(sections)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> Tuple[Dict[str, Any], str]:
        """Extract (params, rationale) from raw LLM text. Robust to markdown fences."""
        text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        data: Dict[str, Any] = {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        rationale = data.get("rationale", "(no rationale provided)")
        raw_params = data.get("params", {})
        params = self._validate_and_fix_params(raw_params)
        return params, rationale

    def _validate_and_fix_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp numerical params and default-fill missing or invalid categoricals."""
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

        # Component types
        out["reranker_type"] = pick("reranker_type", sp.reranker_types, sp.reranker_types[0])
        out["reformulator_type"] = pick("reformulator_type", sp.reformulator_types, sp.reformulator_types[0])
        out["estimator_type"] = pick("estimator_type", sp.estimator_types, sp.estimator_types[0])
        out["scheduler_type"] = pick("scheduler_type", sp.scheduler_types, sp.scheduler_types[0])
        out["feedback_type"] = pick("feedback_type", sp.feedback_types, sp.feedback_types[0])

        # Conditional sub-params
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

        # Always-required params
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
