"""
Outcome-reflective hyper-parameter search.

The proposer is shown, for a sample of queries, the *ranked list the pipeline
actually returned* alongside the ground-truth relevance labels, and is asked to
diagnose why specific queries failed before changing a knob.

This is a different feedback channel from the other arms, not a different search
heuristic over the same one:

    bayes / random    no feedback beyond the objective tuple
    agenthpo          aggregate NDCG + cost per trial
    llm / gepafree    aggregate NDCG + cost + execution-trace summaries
                      (pool size, % reranked) + a per-query NDCG *scalar*
    reflect (this)    per-query ranked lists vs qrels, with first-stage ranks

The motivation is credit assignment. A single aggregate score per trial carries
almost no information about *which* of 15 knobs to move, which is the most
likely reason bayes, random and both GEPA variants are statistically
indistinguishable in this space. A relevant document sitting at first-stage rank
140 while original_query_depth is 100 points at one knob unambiguously.

First-stage ranks are computed ONCE per cell: the retriever is fixed across
trials and original_query_depth only truncates its ordering, so a relevant
document's BM25 rank does not change between trials.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ragtune.tuning.llm_optimizer import (
    LLMCandidate,
    compute_pareto_front,
    evaluate_controller_full,
    sample_random_params,
    validate_and_fix_params,
    write_pareto_configs,
)
from ragtune.tuning.query_evidence import (
    FIRST_STAGE_PROBE_DEPTH,
    build_relevance_map,
    choose_evidence_qids,
    render_evidence,
)
from ragtune.tuning.search_space import RAGtuneSearchSpace


@dataclass
class ReflectConfig:
    name: str
    llm_model: str = "anthropic/claude-opus-5"
    temperature: float = 1.0
    n_iterations: int = 30
    n_eval_queries: int = 50
    n_evidence_queries: int = 6      # queries shown per proposal
    n_contrast_queries: int = 2      # of those, best-performing (the rest worst)
    seed: int = 42
    output_dir: str = "reflect_results"
    search_space_overrides: Dict[str, Any] = field(default_factory=dict)
    dataset_info: str = ""
    model_info: str = ""
    baseline_ndcg: Optional[float] = None
    cost_label: str = "mean reranked documents per query"
    verbose: bool = True


class ReflectOptimizer:
    def __init__(self, config: ReflectConfig, cost_model: Any = None):
        self.config = config
        self.search_space = RAGtuneSearchSpace(**config.search_space_overrides)
        self.cost_model = cost_model
        # query_id -> {"query": str, "relevant": [(doc_id, rel, first_stage_rank)]}
        self.relevance_map: Dict[str, Dict[str, Any]] = {}

    # ── one-off per cell: where do the relevant documents sit in BM25? ────────

    def build_relevance_map(self, fixed_retriever: Any, eval_dataset: Any) -> None:
        self.relevance_map = build_relevance_map(
            fixed_retriever, eval_dataset, self.config.n_eval_queries
        )

    # ── evidence block ───────────────────────────────────────────────────────

    def choose_evidence_qids(self, query_ndcg: Dict[str, float]) -> List[str]:
        return choose_evidence_qids(
            query_ndcg, self.relevance_map,
            self.config.n_evidence_queries, self.config.n_contrast_queries,
        )

    def render_evidence(self, qids, per_query, query_ndcg, depth) -> str:
        return render_evidence(
            qids, per_query, query_ndcg, self.relevance_map, depth
        )

    # ── prompts ──────────────────────────────────────────────────────────────

    def _hp_block(self) -> str:
        sp = self.search_space
        lines = []
        for knob, attr in (
            ("reranker_type", "reranker_types"), ("ce_model", "ce_models"),
            ("estimator_type", "estimator_types"), ("scheduler_type", "scheduler_types"),
            ("feedback_type", "feedback_types"), ("similarity_model", "similarity_models"),
        ):
            v = list(getattr(sp, attr, []) or [])
            if len(v) > 1:
                lines.append(f"- {knob}: one of {v}")
        for knob, attr in (
            ("original_query_depth", "original_query_depth_range"),
            ("max_pool_size", "max_pool_size_range"),
            ("near_duplicate_threshold", "near_duplicate_threshold_range"),
            ("scheduler_batch_size", "scheduler_batch_size_range"),
            ("assembler_max_docs", "assembler_max_docs_range"),
            ("budget_rerank_docs", "budget_rerank_docs_range"),
            ("gd_llm_limit", "gd_llm_limit_range"),
            ("gd_ce_limit", "gd_ce_limit_range"),
            ("budget_stop_token_threshold", "budget_stop_token_threshold_range"),
        ):
            r = getattr(sp, attr, None)
            if r and r[0] != r[1]:
                lines.append(f"- {knob}: in [{r[0]}, {r[1]}]")
        return "\n".join(lines)

    def _system_prompt(self) -> str:
        goal = [
            "You tune a retrieval pipeline. Each round you are shown the current "
            "configuration and, for a sample of queries, the ranked list it "
            "returned against the ground-truth relevance labels.",
            "",
            "Diagnose the failures before changing anything. Typical causes:",
            "  - relevant doc beyond original_query_depth -> it never entered the "
            "pool; raise depth",
            "  - relevant doc entered the pool but was never reranked -> "
            "budget_rerank_docs too small, or scheduler_batch_size / limits too tight",
            "  - relevant doc was reranked but ranked below irrelevant ones -> the "
            "reranker is wrong for this corpus; try another ce_model, or rerank less",
            "  - relevant doc never retrieved at all -> the first stage cannot find "
            "it and no setting of these knobs will help; do not chase it",
            "",
            "Objectives:",
            "  - MAXIMIZE ndcg_at_10",
            f"  - MINIMIZE cost ({self.config.cost_label})",
        ]
        if self.config.baseline_ndcg is not None:
            goal.append(
                f"\nThe untuned pipeline scores {self.config.baseline_ndcg:.4f}; "
                "below that you have found nothing."
            )
        return (
            "\n".join(goal)
            + "\n\n## Model\n" + (self.config.model_info or "(not supplied)")
            + "\n\n## Dataset\n" + (self.config.dataset_info or "(not supplied)")
            + "\n\n## Hyperparameters you may set\n" + self._hp_block()
            + "\n\n## Rules\n"
            "- Propose a configuration distinct from every one already tried.\n"
            "- Tie each change to a specific failure you can point to in the "
            "evidence. Do not change knobs you cannot justify.\n\n"
            "Respond with JSON only:\n"
            '{"diagnosis": "<what failed and why>", '
            '"reasoning": "<which knobs you are changing and why>", '
            '"params": {"<knob>": <value>}}'
        )

    def _user_message(self, history: List[LLMCandidate], evidence: str) -> str:
        if not history:
            return (
                "## Completed trials\n(none yet — propose a sensible starting "
                "configuration.)"
            )
        front = {id(c) for c in compute_pareto_front([c for c in history if not c.error])}
        out = [f"## Completed trials ({len(history)})"]
        # Full history with the same rationale budget as the agenthpo arm. An
        # earlier version showed only the last 12 at 220 chars, which confounded
        # "does outcome evidence help" with "59% less history".
        for c in history:
            if c.error:
                out.append(f"- trial {c.iteration}: FAILED ({c.error})")
                continue
            tag = "  <-- ON PARETO FRONT" if id(c) in front else ""
            out.append(
                f"- trial {c.iteration}: ndcg={c.ndcg_at_10:.4f} "
                f"cost={c.mean_rerank_docs:.5f}{tag}\n"
                f"    params: {json.dumps(c.params)}\n"
                f"    reasoning: {(c.rationale or '').strip()[:300]}"
            )
        return "\n".join(out) + "\n\n" + evidence

    # ── proposal ─────────────────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str) -> str:
        import litellm

        r = litellm.completion(
            model=self.config.llm_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.config.temperature,
        )
        return r.choices[0].message.content or "{}"

    @staticmethod
    def _parse(raw: str) -> Tuple[Dict[str, Any], str]:
        text = re.sub(r"```(?:json)?\s*", "", raw or "").strip().rstrip("`").strip()
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            try:
                data = json.loads(m.group(0)) if m else {}
            except (json.JSONDecodeError, AttributeError):
                data = {}
        if not isinstance(data, dict):
            data = {}
        p = data.get("params")
        why = " | ".join(
            str(data.get(k, "")).strip() for k in ("diagnosis", "reasoning")
            if data.get(k)
        )
        return (p if isinstance(p, dict) else {}), why

    @staticmethod
    def _sig(params: Dict[str, Any]) -> str:
        return json.dumps(params, sort_keys=True, default=str)

    def propose(self, history, seen, rng, evidence) -> Tuple[Dict[str, Any], str, bool]:
        system, user = self._system_prompt(), self._user_message(history, evidence)
        for attempt in range(2):
            msg = user if attempt == 0 else (
                user + "\n\n## Correction\nThat configuration was already tried. "
                "Propose a different one."
            )
            try:
                params, why = self._parse(self._call_llm(system, msg))
            except Exception as exc:  # noqa: BLE001
                return (sample_random_params(self.search_space, rng),
                        f"random fallback: proposal call failed ({type(exc).__name__})", True)
            fixed = validate_and_fix_params(self.search_space, params)
            if self._sig(fixed) not in seen:
                return fixed, why, False
        return (sample_random_params(self.search_space, rng),
                "random fallback: duplicate proposed twice", True)

    # ── loop ─────────────────────────────────────────────────────────────────

    def run(self, fixed_retriever: Any, eval_dataset: Any) -> List[LLMCandidate]:
        rng = random.Random(self.config.seed)
        self.build_relevance_map(fixed_retriever, eval_dataset)
        history: List[LLMCandidate] = []
        seen: Set[str] = set()
        evidence = "## Per-query outcomes\n(no trial run yet)"
        query_ndcg: Dict[str, float] = {}

        for it in range(self.config.n_iterations):
            if self.config.verbose:
                print(f"    [reflect] trial {it+1}/{self.config.n_iterations}: "
                      f"proposing...", flush=True)
            t0 = time.time()
            params, why, fallback = self.propose(history, seen, rng, evidence)
            seen.add(self._sig(params))
            if self.config.verbose:
                print(f"    [reflect] trial {it+1}: proposed in {time.time()-t0:.0f}s"
                      f"{' (random fallback)' if fallback else ''}; evaluating...",
                      flush=True)

            cand = LLMCandidate(
                iteration=it, params=params, ndcg_at_10=0.0,
                mean_rerank_docs=float("inf"),
                rationale=why + ("  [random fallback]" if fallback else ""),
            )
            try:
                qids = set(self.choose_evidence_qids(query_ndcg))
                controller = self.search_space.build_controller(params, fixed_retriever)
                res = evaluate_controller_full(
                    controller, eval_dataset, self.config.n_eval_queries,
                    self.search_space.to_retrieval_overrides(params),
                    cost_model=self.cost_model, capture_qids=qids,
                )
                obj = res.objectives
                cand.ndcg_at_10 = obj.ndcg_at_10
                cand.mean_rerank_docs = (
                    obj.cost_usd if self.cost_model is not None else obj.rerank_docs
                )
                cand.trace = res.trace
                cand.query_ndcg = res.query_ndcg
                query_ndcg = res.query_ndcg
                evidence = self.render_evidence(
                    sorted(qids), res.per_query, res.query_ndcg,
                    int(params.get("original_query_depth", 0)) or None,
                )
            except Exception as exc:  # noqa: BLE001
                cand.error = f"{type(exc).__name__}: {exc}"
            history.append(cand)
            if self.config.verbose:
                st = (f"FAILED {cand.error}" if cand.error
                      else f"ndcg={cand.ndcg_at_10:.4f} cost={cand.mean_rerank_docs:.5f}")
                print(f"    [reflect] trial {it+1}: {st}", flush=True)

        front = compute_pareto_front([c for c in history if not c.error])
        write_pareto_configs(self.search_space, self.config.output_dir, front)
        return history
