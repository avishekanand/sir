#!/usr/bin/env python
"""
Prompt optimization for RAGtune's LLM-backed components.

Implements specs/spec-prompt-optimization.md: a GEPA-style reflective search
where the evolved artifact is the *prompt text*, not the pipeline configuration.
The retriever, the pipeline config and the evaluation set are all held fixed, so
the only independent variable is the prompt.

This is the complement to the two existing optimizers, both of which search
structured parameters and never touch a prompt:

    tuning/optimizer.py      structured params, proposed by Optuna TPE
    tuning/llm_optimizer.py  structured params, proposed by an LLM (GEPA)
    this script              prompt text,       proposed by an LLM (GEPA)

Targets
-------
  reranker       LLMReranker's pointwise relevance prompt ({query}, {document})
  reformulator   LLMReformulator's query-rewrite prompt   ({query}, {m})

Loop (per iteration)
--------------------
  1. Sample a parent from the Pareto pool, weighted by per-query wins.
  2. Collect that parent's worst queries, with the documents it mis-scored.
  3. Ask the optimizer LLM to rewrite the prompt given those failures.
  4. Validate placeholders — a malformed template is recorded, not evaluated.
  5. Screen on a minibatch; only survivors get a full evaluation.

Quick start
-----------
    export OPENAI_API_KEY=...
    # see what it would cost before spending anything
    python examples/prompt_optimizer.py --dataset nfcorpus --dry-run

    # optimize the reranker prompt
    python examples/prompt_optimizer.py --dataset nfcorpus --target reranker \\
        --iterations 20 --n-queries 20 --rerank-depth 10

    # evaluate an evolved prompt against the hand-written one, 3 seeds
    python examples/prompt_optimizer.py --dataset scifact --target reranker \\
        --compare out/prompts_best.yaml --seeds 42,43,44

Requirements
------------
    pip install -e ".[tuning]"
    pip install python-terrier ir-datasets litellm
    export OPENAI_API_KEY=...
    A JDK (not just a JRE) on PATH or in the conda env.

Cost warning
------------
Pointwise reranking is one API call per document: a full evaluation costs
n_queries * rerank_depth calls. The defaults are deliberately small. --dry-run
prints the projection and the script refuses to exceed --max-calls without --yes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Run straight from a checkout, without `pip install -e .` (repo convention).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))  # for the shared helpers in beir_full_benchmark

# ══════════════════════════════════════════════════════════════════════════════
# Target definitions
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PromptTarget:
    """Everything that differs between the components whose prompt we can evolve."""

    name: str
    component: str                 # which pipeline slot this occupies
    registry_type: str             # RAGtune registry name
    placeholders: Tuple[str, ...]  # must appear in the user template
    config_key: str                # dot path the component reads
    via_prompts_store: bool        # True -> config.set_prompt, False -> config.set
    source_key: Optional[str]      # where the shipped default lives in prompts.yaml
    task_description: str          # handed to the optimizer LLM
    output_contract: str           # what the component's parser requires


TARGETS: Dict[str, PromptTarget] = {
    "reranker": PromptTarget(
        name="reranker",
        component="reranker",
        registry_type="llm",
        placeholders=("query", "document"),
        # LLMReranker reads config.get("prompts.reranking.pointwise"), which
        # resolves against _config — so this is written with set(), not
        # set_prompt(). See spec section 3.1: the component's key and accessor
        # are both wrong relative to prompts.yaml, and fixing that belongs in
        # its own fix/ branch. Seeding the value here works around it.
        config_key="prompts.reranking.pointwise",
        via_prompts_store=False,
        source_key="reranking.pointwise_scoring",
        task_description=(
            "score a single document's relevance to a search query, on a 0.0-1.0 scale"
        ),
        output_contract=(
            'The model is called with response_format={"type": "json_object"} and the '
            'reply is parsed as JSON; the score is read from the key "relevance_score". '
            "The prompt MUST therefore instruct the model to reply with a JSON object "
            'containing a numeric "relevance_score" field.'
        ),
    ),
    "reformulator": PromptTarget(
        name="reformulator",
        component="reformulator",
        registry_type="llm_rewrite",
        placeholders=("query", "m"),
        config_key="reformulation.llm_rewrite",
        via_prompts_store=True,
        source_key="reformulation.llm_rewrite",
        task_description=(
            "rewrite a search query into {m} diverse variants that improve recall"
        ),
        output_contract=(
            'The model is called with response_format={"type": "json_object"} and the '
            "reply is parsed for a JSON list of query strings. The prompt MUST instruct "
            "the model to output JSON containing that list."
        ),
    ),
}

# Used when prompts.yaml has nothing usable for the target (e.g. the reranker,
# whose shipped key the component never reads).
FALLBACK_PROMPTS: Dict[str, Dict[str, str]] = {
    "reranker": {
        "system": "You are a helpful assistant that rates the relevance of documents to a query.",
        "user": (
            "Query: {query}\n\n"
            "Document: {document}\n\n"
            "Rate how relevant the document is to the query from 0.0 to 1.0. "
            'Respond with JSON: {{"relevance_score": <float>}}'
        ),
    },
    "reformulator": {
        "system": "You are a search expert. Your goal is to rewrite the user's query into different variations to improve retrieval.",
        "user": (
            "Original Query: {query}\n"
            "Generate {m} different search queries that cover different aspects or "
            "synonyms of the original query. Output as a JSON list of strings only."
        ),
    },
}

# Fixed pipeline around the prompt under test. Only the prompt varies.
PIPELINE_PARAMS: Dict[str, Any] = {
    "reranker_type": "noop",
    "reformulator_type": "identity",
    "estimator_type": "baseline",
    "scheduler_type": "active-learning",
    "feedback_type": "none",
    "original_query_depth": 50,
    "depth_per_reformulation": 5,
    "max_pool_size": 50,
    "near_duplicate_threshold": 0.8,
    "assembler_max_docs": 20,
    "budget_rerank_docs": 10,
    "budget_reformulations": 0,
    "scheduler_batch_size": 5,
    "gd_llm_limit": 3,
    "gd_ce_limit": 10,
    "ce_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "monot5_model": "castorini/monot5-base-msmarco",
    "monot5_batch_size": 16,
    "reformulator_model": "gpt-4o-mini",
    "reformulator_n_variants": 3,
    "similarity_model": "all-MiniLM-L6-v2",
    "min_reranked_for_regression": 3,
    "budget_stop_token_threshold": 0.9,
}


# ══════════════════════════════════════════════════════════════════════════════
# Candidates
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PromptCandidate:
    iteration: int
    system: str
    user: str
    ndcg_at_10: float = 0.0
    mean_rerank_docs: float = 0.0
    rationale: str = ""
    parent: Optional[int] = None
    error: str = ""
    query_ndcg: Dict[str, float] = field(default_factory=dict)
    api_calls: int = 0

    def dominates(self, other: "PromptCandidate") -> bool:
        """Standard Pareto dominance on (NDCG up, cost down)."""
        return (
            self.ndcg_at_10 >= other.ndcg_at_10
            and self.mean_rerank_docs <= other.mean_rerank_docs
            and (self.ndcg_at_10 > other.ndcg_at_10
                 or self.mean_rerank_docs < other.mean_rerank_docs)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "mean_rerank_docs": round(self.mean_rerank_docs, 2),
            "parent": self.parent,
            "error": self.error,
            "rationale": self.rationale,
            "system": self.system,
            "user": self.user,
        }


def pareto_front(candidates: List[PromptCandidate]) -> List[PromptCandidate]:
    valid = [c for c in candidates if not c.error]
    return [c for c in valid if not any(o.dominates(c) for o in valid if o is not c)]


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════


def validate_prompt(system: str, user: str, target: PromptTarget,
                    max_chars: int) -> Optional[str]:
    """
    Return an error string if this candidate cannot be used, else None.

    The component calls user.format(**placeholders), so anything format() would
    choke on has to be caught here — otherwise it surfaces as a per-query
    exception that the evaluator silently scores 0.0, and the optimizer learns
    from a number that reflects a crash rather than a bad prompt.
    """
    if not system.strip() or not user.strip():
        return "empty system or user template"
    if len(user) > max_chars:
        return f"user template is {len(user)} chars (limit {max_chars})"

    for ph in target.placeholders:
        if "{" + ph + "}" not in user:
            return f"missing required placeholder {{{ph}}}"

    probe = {ph: "x" for ph in target.placeholders}
    try:
        user.format(**probe)
    except KeyError as exc:
        return f"unknown placeholder {exc} — escape literal braces as {{{{ }}}}"
    except (IndexError, ValueError) as exc:
        return f"malformed template: {type(exc).__name__}: {exc}"
    return None


def install_prompt(target: PromptTarget, system: str, user: str) -> None:
    """Make the running pipeline use this prompt on its next call."""
    from ragtune.utils.config import config

    payload = {"system": system, "user": user}
    if target.via_prompts_store:
        config.set_prompt(target.config_key, payload)
    else:
        config.set(target.config_key, payload)


def load_seed_prompt(target: PromptTarget, path: Optional[str]) -> Dict[str, str]:
    """Baseline prompt: an explicit file, else prompts.yaml, else the fallback."""
    if path:
        data = yaml.safe_load(Path(path).read_text())
        if "system" not in data or "user" not in data:  # allow a nested fragment
            for value in data.values():
                if isinstance(value, dict) and "user" in value:
                    data = value
                    break
        return {"system": str(data["system"]), "user": str(data["user"])}

    from ragtune.utils.config import config

    shipped = config.get_prompt(target.source_key) if target.source_key else None
    if isinstance(shipped, dict) and "user" in shipped:
        candidate = {"system": str(shipped.get("system", "")), "user": str(shipped["user"])}
        # prompts.yaml's reranker entry is listwise ({documents}) while the
        # component is pointwise ({document}) — only adopt it if it fits.
        if validate_prompt(candidate["system"], candidate["user"], target, 10_000) is None:
            return candidate
        print(f"  [seed] prompts.yaml '{target.source_key}' does not satisfy the "
              f"{target.name} contract ({', '.join('{'+p+'}' for p in target.placeholders)}); "
              "using the built-in fallback instead")
    return dict(FALLBACK_PROMPTS[target.name])


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════


def build_params(target: PromptTarget, args) -> Dict[str, Any]:
    """Fixed pipeline with the target component switched to its LLM version."""
    params = dict(PIPELINE_PARAMS)
    params["original_query_depth"] = args.retrieval_depth
    params["max_pool_size"] = args.retrieval_depth
    params["budget_rerank_docs"] = args.rerank_depth
    if target.component == "reranker":
        params["reranker_type"] = target.registry_type
        params["llm_reranker_model"] = args.component_model
    else:
        params["reformulator_type"] = target.registry_type
        params["reformulator_model"] = args.component_model
        params["budget_reformulations"] = 1
        # Reranking is not under test here; keep it cheap and deterministic.
        params["reranker_type"] = "noop"
    return params


def evaluate_prompt(candidate: PromptCandidate, target: PromptTarget, retriever,
                    eval_ds, n_queries: int, args) -> None:
    """Run the pipeline with this prompt installed; fill in the candidate's scores."""
    from ragtune.tuning.llm_optimizer import evaluate_controller_full
    from ragtune.tuning.search_space import RAGtuneSearchSpace

    install_prompt(target, candidate.system, candidate.user)
    space = RAGtuneSearchSpace()
    params = build_params(target, args)

    controller = space.build_controller(params, retriever)
    result = evaluate_controller_full(
        controller, eval_ds, n_queries, space.to_retrieval_overrides(params)
    )
    candidate.ndcg_at_10 = result.objectives.ndcg_at_10
    candidate.mean_rerank_docs = result.objectives.rerank_docs
    candidate.query_ndcg = result.query_ndcg
    candidate.api_calls = int(round(result.objectives.rerank_docs * n_queries))


# ══════════════════════════════════════════════════════════════════════════════
# Reflection
# ══════════════════════════════════════════════════════════════════════════════

REFLECTION_SYSTEM = """\
You are optimizing the prompt used by one component of a retrieval pipeline.

Component task: {task}
The prompt is used with str.format(), so these placeholders MUST appear verbatim
in the user template: {placeholders}
Any other brace must be escaped by doubling it: {{{{ like this }}}}.

{contract}

You are given the current prompt, the NDCG@10 it achieved, and the queries where
it did worst — including the documents it scored and the scores it gave them.
Diagnose WHY those queries failed, then rewrite the prompt to fix that specific
failure. Change one thing at a time; do not rewrite from scratch.

Reply with JSON only, no markdown:
{{"rationale": "<what failed and what you changed>", "system": "<new system prompt>", "user": "<new user template>"}}
"""


def build_reflection_message(parent: PromptCandidate, history: List[PromptCandidate],
                             eval_ds, target: PromptTarget, n_failures: int) -> str:
    sections = [
        f"## Current prompt (iteration {parent.iteration})",
        f"NDCG@10 = {parent.ndcg_at_10:.4f}   mean cost = {parent.mean_rerank_docs:.1f} docs/query",
        "",
        "SYSTEM:",
        parent.system,
        "",
        "USER TEMPLATE:",
        parent.user,
        "",
    ]

    worst = sorted(parent.query_ndcg.items(), key=lambda kv: kv[1])[:n_failures]
    if worst:
        sections.append(f"## Worst {len(worst)} queries for this prompt")
        by_id = {q.query_id: q for q in eval_ds.queries}
        for qid, score in worst:
            eq = by_id.get(qid)
            if eq is None:
                continue
            n_rel = sum(1 for v in eq.qrels.values() if v > 0)
            sections.append(
                f"  NDCG@10={score:.3f}  query={eq.query!r}  "
                f"({n_rel} relevant documents exist in the collection)"
            )
        sections.append("")

    tried = [c for c in history if c.error or c.ndcg_at_10 <= parent.ndcg_at_10]
    if tried:
        sections.append("## Variants already tried that did NOT beat the current prompt")
        for c in tried[-6:]:
            verdict = c.error or f"NDCG@10={c.ndcg_at_10:.4f}"
            sections.append(f"  iter {c.iteration}: {verdict} — {c.rationale[:140]}")
        sections.append("")

    sections.append("Rewrite the prompt to fix the failures above.")
    return "\n".join(sections)


def propose(parent: PromptCandidate, history: List[PromptCandidate], eval_ds,
            target: PromptTarget, args) -> Tuple[str, str, str]:
    """Ask the optimizer LLM for a mutated prompt. Returns (system, user, rationale)."""
    import litellm

    placeholders = ", ".join("{" + p + "}" for p in target.placeholders)
    system = REFLECTION_SYSTEM.format(
        task=target.task_description,
        placeholders=placeholders,
        contract=target.output_contract,
    )
    user = build_reflection_message(parent, history, eval_ds, target, args.n_failures)

    response = litellm.completion(
        model=args.optimizer_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=args.temperature,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1]) if start >= 0 < end else {}

    return (
        str(data.get("system", parent.system)),
        str(data.get("user", parent.user)),
        str(data.get("rationale", "")),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Optimization loop
# ══════════════════════════════════════════════════════════════════════════════


def sample_parent(pool: List[PromptCandidate], rng: random.Random) -> PromptCandidate:
    """
    Pick a parent from the Pareto front, weighted by per-query wins.

    GEPA's instance-wise selection: a candidate that is best on many individual
    queries is a better mutation base than one with the highest mean, because
    the mean hides which queries it actually fixed.
    """
    front = pareto_front(pool) or pool
    wins = {id(c): 1 for c in front}  # 1 = Laplace smoothing, no zero weights
    all_qids = {q for c in front for q in c.query_ndcg}
    for qid in all_qids:
        best = max(front, key=lambda c: c.query_ndcg.get(qid, -1.0))
        wins[id(best)] += 1
    return rng.choices(front, weights=[wins[id(c)] for c in front], k=1)[0]


def optimize(target: PromptTarget, retriever, eval_ds, args) -> List[PromptCandidate]:
    rng = random.Random(args.seed)
    history: List[PromptCandidate] = []

    seed_prompt = load_seed_prompt(target, args.seed_prompt)
    baseline = PromptCandidate(
        iteration=0, system=seed_prompt["system"], user=seed_prompt["user"],
        rationale="Baseline prompt (unoptimized).",
    )
    err = validate_prompt(baseline.system, baseline.user, target, args.max_prompt_chars)
    if err:
        raise SystemExit(f"Seed prompt is invalid: {err}")

    print(f"\n  [iter  0] evaluating baseline prompt on {args.n_queries} queries …")
    evaluate_prompt(baseline, target, retriever, eval_ds, args.n_queries, args)
    history.append(baseline)
    print(f"  [iter  0] NDCG@10={baseline.ndcg_at_10:.4f}  "
          f"cost={baseline.mean_rerank_docs:.1f} docs/query  "
          f"(~{baseline.api_calls} API calls)")

    for it in range(1, args.iterations + 1):
        parent = sample_parent(history, rng)
        try:
            system, user, rationale = propose(parent, history, eval_ds, target, args)
        except Exception as exc:  # noqa: BLE001 — API hiccup shouldn't end the run
            print(f"  [iter {it:2d}] proposal failed — {type(exc).__name__}: {exc}")
            history.append(PromptCandidate(
                iteration=it, system=parent.system, user=parent.user,
                error=f"proposal failed: {type(exc).__name__}", parent=parent.iteration,
            ))
            continue

        cand = PromptCandidate(iteration=it, system=system, user=user,
                               rationale=rationale, parent=parent.iteration)

        err = validate_prompt(system, user, target, args.max_prompt_chars)
        if err:
            cand.error = f"invalid: {err}"
            history.append(cand)
            print(f"  [iter {it:2d}] rejected — {err}")
            continue

        # Stage 1: minibatch screen against the parent on the same queries.
        if args.minibatch and args.minibatch < args.n_queries:
            evaluate_prompt(cand, target, retriever, eval_ds, args.minibatch, args)
            parent_mini = _mean(
                parent.query_ndcg.get(q.query_id, 0.0)
                for q in eval_ds.iter_queries(limit=args.minibatch)
            )
            if cand.ndcg_at_10 < parent_mini:
                cand.error = f"screened out (minibatch {cand.ndcg_at_10:.3f} < parent {parent_mini:.3f})"
                history.append(cand)
                print(f"  [iter {it:2d}] screened out on minibatch "
                      f"({cand.ndcg_at_10:.3f} vs {parent_mini:.3f}) — {rationale[:60]}")
                continue

        # Stage 2: full evaluation.
        try:
            evaluate_prompt(cand, target, retriever, eval_ds, args.n_queries, args)
        except Exception as exc:  # noqa: BLE001
            cand.error = f"eval failed: {type(exc).__name__}: {exc}"
            history.append(cand)
            print(f"  [iter {it:2d}] evaluation failed — {cand.error[:80]}")
            continue

        history.append(cand)
        best = max(c.ndcg_at_10 for c in history if not c.error)
        flag = "  <-- new best" if cand.ndcg_at_10 >= best else ""
        print(f"  [iter {it:2d}] NDCG@10={cand.ndcg_at_10:.4f}  "
              f"cost={cand.mean_rerank_docs:.1f}{flag}")
        print(f"            {rationale[:150]}")

    return history


def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════


def write_results(history: List[PromptCandidate], target: PromptTarget,
                  out_dir: Path, args) -> Optional[PromptCandidate]:
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = [c for c in history if not c.error]
    if not valid:
        print("\n  No prompt survived evaluation — nothing to write.")
        return None

    best = max(valid, key=lambda c: c.ndcg_at_10)
    baseline = history[0]

    # A prompts.yaml fragment: paste under the matching key, or load with
    # --seed-prompt on a later run.
    key_parts = target.source_key.split(".") if target.source_key else [target.name]
    fragment: Dict[str, Any] = {}
    node = fragment
    for part in key_parts[:-1]:
        node = node.setdefault(part, {})
    node[key_parts[-1]] = {"system": best.system, "user": best.user}
    (out_dir / "prompts_best.yaml").write_text(yaml.dump(fragment, sort_keys=False))

    for cand in pareto_front(history):
        name = f"prompt_iter{cand.iteration}_ndcg{cand.ndcg_at_10:.3f}.yaml"
        (out_dir / name).write_text(yaml.dump(
            {"system": cand.system, "user": cand.user,
             "ndcg_at_10": round(cand.ndcg_at_10, 4),
             "mean_rerank_docs": round(cand.mean_rerank_docs, 2),
             "rationale": cand.rationale},
            sort_keys=False,
        ))

    (out_dir / "history.json").write_text(json.dumps(
        {"target": target.name, "dataset": args.dataset, "seed": args.seed,
         "n_queries": args.n_queries, "rerank_depth": args.rerank_depth,
         "component_model": args.component_model, "optimizer_model": args.optimizer_model,
         "candidates": [c.to_dict() for c in history]},
        indent=2,
    ))

    gain = best.ndcg_at_10 - baseline.ndcg_at_10
    print("\n" + "=" * 88)
    print(f"  Prompt optimization — {target.name} on {args.dataset} (seed {args.seed})")
    print("=" * 88)
    print(f"  baseline prompt   NDCG@10 = {baseline.ndcg_at_10:.4f}")
    print(f"  best evolved      NDCG@10 = {best.ndcg_at_10:.4f}  (iteration {best.iteration})")
    print(f"  gain                        {gain:+.4f}")
    print(f"  evaluated {len(valid)} prompts, rejected {len(history) - len(valid)}, "
          f"~{sum(c.api_calls for c in history):,} component API calls")
    print(f"  written to {out_dir}/")
    if abs(gain) < 0.02:
        print("  NOTE: a gain under ~0.02 NDCG on a single seed is within noise. "
              "Confirm with --compare over several seeds before adopting.")
    print("=" * 88)
    return best


def compare_prompts(target: PromptTarget, retriever_factory, args) -> int:
    """
    Head-to-head: baseline prompt vs an evolved one, over several seeds.

    This is the only honest way to decide whether an evolved prompt is real —
    the optimization run itself selects for the seed it was tuned on.
    """
    from ragtune.tuning.evaluator import EvalDataset

    evolved = load_seed_prompt(target, args.compare)
    baseline = load_seed_prompt(target, None)

    rows = []
    for seed in args.seed_list:
        eval_ds = EvalDataset.from_pyterrier_irds(
            irds_id=DATASETS[args.dataset]["irds_id"], n_queries=args.n_queries, seed=seed
        )
        retriever = retriever_factory()
        for label, prompt in (("baseline", baseline), ("evolved", evolved)):
            cand = PromptCandidate(iteration=seed, system=prompt["system"], user=prompt["user"])
            evaluate_prompt(cand, target, retriever, eval_ds, args.n_queries, args)
            rows.append({"seed": seed, "prompt": label, "ndcg_at_10": cand.ndcg_at_10,
                         "cost": cand.mean_rerank_docs})
            print(f"  seed {seed}  {label:<9} NDCG@10={cand.ndcg_at_10:.4f}")

    print("\n" + "=" * 72)
    print(f"  {'prompt':<10} {'mean NDCG@10':>13} {'stdev':>8} {'seeds':>6}")
    print("-" * 72)
    means = {}
    for label in ("baseline", "evolved"):
        vals = [r["ndcg_at_10"] for r in rows if r["prompt"] == label]
        means[label] = _mean(vals)
        sd = (sum((v - means[label]) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        print(f"  {label:<10} {means[label]:>13.4f} {sd:>8.4f} {len(vals):>6}")
    delta = means["evolved"] - means["baseline"]
    print("-" * 72)
    print(f"  evolved - baseline = {delta:+.4f}")
    if len(args.seed_list) < 3:
        print("  (fewer than 3 seeds — treat this as indicative only)")
    print("=" * 72)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(rows, indent=2))
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

# Dataset catalogue and PyTerrier plumbing are shared with the benchmark script
# rather than duplicated; both live in examples/.
try:
    from beir_full_benchmark import (  # noqa: E402
        DATASETS, build_sparse_index, init_pyterrier, make_retriever,
    )
except ImportError as _exc:  # pragma: no cover - only when run outside examples/
    raise SystemExit(
        f"prompt_optimizer.py needs beir_full_benchmark.py alongside it ({_exc})"
    )


def estimate_calls(args) -> Dict[str, int]:
    """Projected component API calls — see spec section 8."""
    full = args.n_queries * args.rerank_depth
    mini = (args.minibatch or 0) * args.rerank_depth
    # Assume half the proposals survive screening; that is what the pilot runs
    # showed, and it is the number the warning is based on.
    accepted = max(1, args.iterations // 2)
    return {
        "baseline": full,
        "screening": args.iterations * mini,
        "full_evals": accepted * full,
        "reflection": args.iterations,
        "total": full + args.iterations * mini + accepted * full + args.iterations,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evolve the prompt text used by RAGtune's LLM components.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", default="reranker", choices=sorted(TARGETS),
                   help="which component's prompt to optimize")
    p.add_argument("--dataset", default="nfcorpus",
                   help=f"BEIR dataset; choices: {', '.join(list(DATASETS)[:8])}, …")
    p.add_argument("--retriever", default="bm25", help="first-stage retriever (fixed)")

    e = p.add_argument_group("evaluation")
    e.add_argument("--n-queries", type=int, default=20, help="queries per full evaluation")
    e.add_argument("--minibatch", type=int, default=5,
                   help="queries for stage-1 screening (0 disables screening)")
    e.add_argument("--rerank-depth", type=int, default=10,
                   help="documents scored per query — ALSO the API calls per query")
    e.add_argument("--retrieval-depth", type=int, default=50)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--seeds", default=None, help="comma-separated seeds, for --compare")

    o = p.add_argument_group("optimization")
    o.add_argument("--iterations", type=int, default=20)
    o.add_argument("--optimizer-model", default="gpt-4o-mini",
                   help="model that proposes new prompts")
    o.add_argument("--component-model", default="gpt-4o-mini",
                   help="model the pipeline component itself runs on")
    o.add_argument("--temperature", type=float, default=0.8)
    o.add_argument("--n-failures", type=int, default=5,
                   help="worst queries shown to the optimizer each iteration")
    o.add_argument("--max-prompt-chars", type=int, default=4000)
    o.add_argument("--seed-prompt", default=None,
                   help="YAML file with system/user to start from (default: prompts.yaml)")

    i = p.add_argument_group("io")
    i.add_argument("--index-dir", default="./indexes")
    i.add_argument("--out-dir", default="./prompt_opt_results")
    i.add_argument("--compare", default=None,
                   help="evaluate this prompt YAML against the baseline and exit")
    i.add_argument("--dry-run", action="store_true",
                   help="print the API-call projection and exit")
    i.add_argument("--max-calls", type=int, default=10_000,
                   help="refuse to start above this projection unless --yes")
    i.add_argument("--yes", action="store_true", help="proceed past the cost warning")

    # Flags the shared benchmark helpers expect but this script doesn't expose.
    args = p.parse_args(argv)
    args.bm25_k1, args.bm25_b = 0.9, 0.4
    args.dense_backend, args.encode_batch_size, args.device = "np", 32, "cpu"
    args.hybrid_sparse, args.hybrid_dense, args.rrf_k = "bm25", "dense-minilm", 60

    if args.dataset not in DATASETS:
        p.error(f"unknown dataset {args.dataset!r}")
    args.seed_list = (
        [int(x) for x in str(args.seeds).split(",") if x.strip()] if args.seeds else [args.seed]
    )
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    target = TARGETS[args.target]
    est = estimate_calls(args)

    print("=" * 88)
    print(f"  RAGtune prompt optimization — target: {target.name}")
    print("=" * 88)
    print(f"  dataset    : {args.dataset}   retriever: {args.retriever}")
    print(f"  component  : {target.registry_type} on {args.component_model}")
    print(f"  optimizer  : {args.optimizer_model} (temperature {args.temperature})")
    print(f"  budget     : {args.iterations} iterations x {args.n_queries} queries "
          f"x {args.rerank_depth} docs")
    print(f"  projected component API calls: ~{est['total']:,} "
          f"(baseline {est['baseline']}, screening {est['screening']}, "
          f"full {est['full_evals']}, reflection {est['reflection']})")
    print("=" * 88)

    if args.dry_run:
        return 0
    if est["total"] > args.max_calls and not args.yes:
        print(f"\nProjection exceeds --max-calls ({args.max_calls:,}). "
              "Re-run with --yes, or lower --iterations / --n-queries / --rerank-depth.",
              file=sys.stderr)
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    pt = init_pyterrier()
    import ragtune.adapters  # noqa: F401 — registry side effects
    import ragtune.components  # noqa: F401
    from ragtune.tuning.evaluator import EvalDataset

    build_sparse_index(pt, args.dataset, args.index_dir)

    def retriever_factory():
        return make_retriever(pt, args.retriever, args.dataset, args.index_dir, args)

    if args.compare:
        return compare_prompts(target, retriever_factory, args)

    eval_ds = EvalDataset.from_pyterrier_irds(
        irds_id=DATASETS[args.dataset]["irds_id"], n_queries=args.n_queries, seed=args.seed
    )
    print(f"  [queries] {len(eval_ds.queries)} topics loaded")

    t0 = time.time()
    history = optimize(target, retriever_factory(), eval_ds, args)
    write_results(history, target, Path(args.out_dir), args)
    print(f"  wall time: {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
