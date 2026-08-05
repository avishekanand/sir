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


# ── Domain objects ────────────────────────────────────────────────────────────

@dataclass
class LLMCandidate:
    """One evaluated pipeline configuration."""
    iteration: int
    params: Dict[str, Any]
    ndcg_at_10: float
    mean_rerank_docs: float
    rationale: str
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
) -> TrialObjectives:
    """Evaluate a controller on an EvalDataset without Optuna (no pruning)."""
    if retrieval_overrides:
        from ragtune.utils.config import config as global_cfg
        for k, v in retrieval_overrides.items():
            global_cfg.set(k, v)

    ndcg_scores: List[float] = []
    rerank_costs: List[float] = []
    latencies_ms: List[float] = []
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

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return TrialObjectives(
        ndcg_at_10=_mean(ndcg_scores),
        rerank_docs=_mean(rerank_costs),
        latency_ms=_mean(latencies_ms),
        queries_evaluated=len(ndcg_scores),
    )


# ── Main optimizer class ──────────────────────────────────────────────────────

class LLMAgentOptimizer:
    """
    Reflection-based pipeline optimizer driven by an LLM agent.

    The optimizer maintains a history of evaluated configurations and their
    NDCG@10 / rerank_docs objectives.  At each iteration it builds a prompt
    containing the search space description, the history, and the current
    Pareto front, then asks the LLM to propose a new configuration to try.

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
                objectives = evaluate_controller(
                    controller,
                    eval_dataset,
                    self.config.n_eval_queries,
                    retrieval_overrides,
                )
                candidate = LLMCandidate(
                    iteration=i,
                    params=params,
                    ndcg_at_10=objectives.ndcg_at_10,
                    mean_rerank_docs=objectives.rerank_docs,
                    rationale=rationale,
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
            "## Output format (respond with valid JSON only, no markdown):\n"
            '{\n'
            '  "rationale": "<step-by-step reasoning>",\n'
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

        shown = history[-self.config.max_history_in_prompt:]
        rows = [
            "| iter | ndcg@10 | cost  | reranker     | scheduler            | reformulator | estimator |"
            "\n|------|---------|-------|--------------|----------------------|--------------|-----------|"
        ]
        for c in shown:
            err_note = f" [ERR: {c.error[:40]}]" if c.error else ""
            rows.append(
                f"| {c.iteration:4d} | {c.ndcg_at_10:.3f}   | {c.mean_rerank_docs:5.1f} | "
                f"{c.params.get('reranker_type','?'):12s} | "
                f"{c.params.get('scheduler_type','?'):20s} | "
                f"{c.params.get('reformulator_type','?'):12s} | "
                f"{c.params.get('estimator_type','?'):9s} |"
                f"{err_note}"
            )

        pareto_rows: List[str] = []
        if pareto_front:
            pareto_rows.append("Current Pareto front:")
            for c in sorted(pareto_front, key=lambda x: -x.ndcg_at_10):
                pareto_rows.append(
                    f"  iter={c.iteration:3d}  ndcg={c.ndcg_at_10:.3f}  cost={c.mean_rerank_docs:5.1f}"
                    f"  reranker={c.params.get('reranker_type')} "
                    f"scheduler={c.params.get('scheduler_type')} "
                    f"budget_rerank_docs={c.params.get('budget_rerank_docs')}"
                )

        sections = [
            f"## Evaluation History (last {len(shown)} of {len(history)} iterations)",
            "\n".join(rows),
            "",
            "\n".join(pareto_rows) if pareto_rows else "Pareto front is empty (all runs errored).",
            "",
            "## Your Task",
            "1. Reflect: which parameter combinations produce high NDCG? Low cost?",
            "2. Identify: unexplored regions that could extend the Pareto front.",
            "3. Propose: one new configuration likely to be non-dominated.",
            "",
            "Respond with JSON only (rationale + params, no markdown).",
        ]
        return "\n".join(sections)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> Tuple[Dict[str, Any], str]:
        """Extract (params, rationale) from raw LLM text. Robust to markdown fences."""
        # Strip markdown code fences
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

        # Conditional sub-params (always populated; inactive ones are ignored by build_controller)
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
