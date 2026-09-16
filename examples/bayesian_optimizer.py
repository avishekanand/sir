#!/usr/bin/env python
"""
Bayesian (Optuna TPE) optimization of a RAGtune pipeline on BEIR.

This is the configuration-search counterpart to examples/prompt_optimizer.py.
The two scripts are deliberately built to the same protocol so their results can
be compared directly:

                        bayesian_optimizer.py        prompt_optimizer.py
    what varies         pipeline configuration       prompt text
    what is held fixed  prompts (shipped defaults)   pipeline configuration
    proposer            Optuna multi-objective TPE   LLM reflection (GEPA)
    baseline            BASELINE_PARAMS at iter 0    baseline prompt at iter 0
    metric              NDCG@10 (tuning/evaluator)   NDCG@10 (tuning/evaluator)
    cost axis           mean rerank docs/query       mean rerank docs/query
    headline number     gain over baseline           gain over baseline
    summary.json        identical schema             identical schema

Run both with the same --dataset, --retriever, --n-queries, --seeds and
--iterations and the two `gain` figures answer one question: for the same number
of pipeline evaluations, does tuning the configuration or tuning the prompt buy
more retrieval quality?

Objectives are (maximize NDCG@10, minimize mean rerank docs), so a run produces a
Pareto front rather than a single winner — the same front the notebook
examples/ragtune_benchmark.ipynb plots for its TPE arm.

Quick start
-----------
    # cost projection, evaluates nothing
    python examples/bayesian_optimizer.py --dataset nfcorpus --dry-run

    # optimize
    python examples/bayesian_optimizer.py --dataset nfcorpus \\
        --iterations 50 --n-queries 50

    # apples-to-apples against prompt optimization (same protocol both sides)
    python examples/bayesian_optimizer.py --dataset nfcorpus --rerankers llm \\
        --iterations 20 --n-queries 20 --rerank-depth 10 --seeds 42,43,44
    python examples/prompt_optimizer.py  --dataset nfcorpus --target reranker \\
        --iterations 20 --n-queries 20 --rerank-depth 10 --seeds 42,43,44

    # random-search control — same space, same budget, no surrogate model
    python examples/bayesian_optimizer.py --dataset nfcorpus --sampler random

    # verify a tuned config against the baseline across seeds
    python examples/bayesian_optimizer.py --dataset nfcorpus \\
        --compare bayes_opt_results/best_config.yaml --seeds 42,43,44

Requirements
------------
    pip install -e ".[tuning]"
    pip install python-terrier ir-datasets optuna
    A JDK (not just a JRE) on PATH or in the conda env.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Run straight from a checkout, without `pip install -e .` (repo convention).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))  # shared helpers live in beir_full_benchmark

# Catalogue and PyTerrier plumbing are shared with the sweep script rather than
# duplicated; BASELINE_PARAMS is the single definition of "the untuned
# pipeline", so both optimizers measure their gain from the same starting point.
try:
    from beir_full_benchmark import (  # noqa: E402
        BASELINE_PARAMS, DATASETS, RETRIEVERS, build_sparse_index, hypervolume_2d,
        init_pyterrier, make_retriever, search_space_overrides, write_csv,
        _mean, _stdev,
    )
except ImportError as _exc:  # pragma: no cover - only when run outside examples/
    raise SystemExit(
        f"bayesian_optimizer.py needs beir_full_benchmark.py alongside it ({_exc})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trial:
    """One evaluated configuration. Mirrors prompt_optimizer.PromptCandidate."""

    iteration: int
    params: Dict[str, Any]
    ndcg_at_10: float = 0.0
    mean_rerank_docs: float = 0.0
    state: str = "COMPLETE"
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "mean_rerank_docs": round(self.mean_rerank_docs, 2),
            "state": self.state,
            "error": self.error,
            "params": self.params,
        }


@dataclass
class SeedRun:
    """Everything one seed produced."""

    seed: int
    baseline: Trial
    trials: List[Trial] = field(default_factory=list)
    pareto: List[Trial] = field(default_factory=list)
    wall_seconds: float = 0.0
    config_paths: List[str] = field(default_factory=list)

    @property
    def completed(self) -> List[Trial]:
        return [t for t in self.trials if t.state == "COMPLETE" and not t.error]

    @property
    def best(self) -> Optional[Trial]:
        return max(self.completed, key=lambda t: t.ndcg_at_10, default=None)

    @property
    def gain(self) -> float:
        best = self.best
        return (best.ndcg_at_10 - self.baseline.ndcg_at_10) if best else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════


def baseline_params(args) -> Dict[str, Any]:
    """
    The untuned reference pipeline.

    Same dict prompt_optimizer.py starts from, with the depth/budget knobs this
    run's CLI controls applied — so "gain over baseline" means the same thing in
    both scripts.
    """
    params = dict(BASELINE_PARAMS)
    params.update(
        original_query_depth=args.retrieval_depth,
        max_pool_size=args.retrieval_depth,
        budget_rerank_docs=args.rerank_depth,
        reranker_type=args.rerankers[0],
        ce_model=args.ce_models[0],
        monot5_model=args.monot5_models[0],
        llm_reranker_model=args.llm_reranker_model,
    )
    return params


def evaluate_params(params: Dict[str, Any], retriever, eval_ds, n_queries: int) -> Trial:
    """Run one configuration over n_queries. Same evaluator the tuner uses."""
    from ragtune.tuning.llm_optimizer import evaluate_controller_full
    from ragtune.tuning.search_space import RAGtuneSearchSpace

    space = RAGtuneSearchSpace()
    controller = space.build_controller(params, retriever)
    result = evaluate_controller_full(
        controller, eval_ds, n_queries, space.to_retrieval_overrides(params)
    )
    return Trial(
        iteration=0,
        params=params,
        ndcg_at_10=result.objectives.ndcg_at_10,
        mean_rerank_docs=result.objectives.rerank_docs,
    )


def run_seed(seed: int, retriever, eval_ds, args, out_dir: Path) -> SeedRun:
    """Baseline evaluation, then `--iterations` TPE trials against it."""
    import optuna
    from optuna.samplers import RandomSampler, TPESampler

    from ragtune.tuning.evaluator import TrialEvaluator
    from ragtune.tuning.optimizer import extract_pareto_configs
    from ragtune.tuning.pruners import CostPruner, ParetoPruner, RuntimePruner
    from ragtune.tuning.search_space import RAGtuneSearchSpace

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n  [seed {seed}] evaluating baseline configuration …")
    base = evaluate_params(baseline_params(args), retriever, eval_ds, args.n_queries)
    print(f"  [seed {seed}] baseline NDCG@10={base.ndcg_at_10:.4f}  "
          f"cost={base.mean_rerank_docs:.1f} docs/query")

    overrides = search_space_overrides(args)
    space = RAGtuneSearchSpace(**overrides)

    if args.sampler == "tpe":
        sampler = TPESampler(
            multivariate=True, constant_liar=True,
            n_startup_trials=max(5, args.iterations // 8), seed=seed,
        )
    else:
        sampler = RandomSampler(seed=seed)

    study = optuna.create_study(
        study_name=f"{args.sampler}-{args.dataset}-{args.retriever}-s{seed}",
        directions=["maximize", "minimize"],
        sampler=sampler,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    pruners = [
        CostPruner(max_mean_rerank_docs=args.max_cost, warmup_steps=3),
        RuntimePruner(max_trial_seconds=args.max_trial_seconds, warmup_steps=3),
    ]
    if args.pareto_pruning:
        pruners.append(ParetoPruner(
            study=study, warmup_trials=max(5, args.iterations // 8), zscore=1.645,
        ))
    evaluator = TrialEvaluator(dataset=eval_ds, n_eval_queries=args.n_queries,
                               pruners=pruners)

    best_so_far = {"ndcg": base.ndcg_at_10}

    def objective(trial):
        params = space.sample(trial)
        try:
            controller = space.build_controller(params, retriever)
        except Exception as exc:  # noqa: BLE001 — checkpoint unavailable etc.
            trial.set_user_attr("build_error", str(exc))
            return 0.0, float("inf")

        obj = evaluator.evaluate(controller, trial, space.to_retrieval_overrides(params))
        flag = ""
        if obj.ndcg_at_10 > best_so_far["ndcg"]:
            best_so_far["ndcg"] = obj.ndcg_at_10
            flag = "  <-- new best"
        print(f"  [seed {seed}] trial {trial.number:3d}  "
              f"NDCG@10={obj.ndcg_at_10:.4f}  cost={obj.rerank_docs:6.1f}  "
              f"{params['reranker_type']:<14}{flag}")
        return obj.ndcg_at_10, obj.rerank_docs

    t0 = time.time()
    study.optimize(objective, n_trials=args.iterations, n_jobs=1, catch=(Exception,))
    elapsed = time.time() - t0

    pareto_numbers = {t.number for t in study.best_trials}
    trials: List[Trial] = []
    for t in study.trials:
        if t.values is None:
            trials.append(Trial(iteration=t.number, params=t.params,
                                state=str(t.state).split(".")[-1],
                                error=t.user_attrs.get("build_error", "")))
            continue
        trials.append(Trial(
            iteration=t.number, params=t.params,
            ndcg_at_10=t.values[0], mean_rerank_docs=t.values[1],
            state=str(t.state).split(".")[-1],
            error=t.user_attrs.get("build_error", ""),
        ))

    cfg_dir = out_dir / f"pareto_configs_s{seed}"
    paths = extract_pareto_configs(study, space, str(cfg_dir))

    run = SeedRun(
        seed=seed, baseline=base, trials=trials,
        pareto=[t for t in trials if t.iteration in pareto_numbers and t.state == "COMPLETE"],
        wall_seconds=elapsed, config_paths=[str(p) for p in paths],
    )
    best = run.best
    print(f"  [seed {seed}] done in {elapsed:.0f}s — "
          f"{len(run.completed)}/{args.iterations} trials scored, "
          f"best NDCG@10={best.ndcg_at_10 if best else 0.0:.4f} "
          f"({run.gain:+.4f} vs baseline), Pareto size {len(run.pareto)}")
    return run


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════


def write_results(runs: List[SeedRun], args, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for run in runs:
        rows.append({"seed": run.seed, "iteration": -1, "role": "baseline",
                     "ndcg_at_10": round(run.baseline.ndcg_at_10, 4),
                     "mean_rerank_docs": round(run.baseline.mean_rerank_docs, 2),
                     "on_pareto": False, "state": "COMPLETE"})
        pareto_iters = {t.iteration for t in run.pareto}
        for t in run.trials:
            row = {"seed": run.seed, "iteration": t.iteration, "role": "trial",
                   "ndcg_at_10": round(t.ndcg_at_10, 4),
                   "mean_rerank_docs": round(t.mean_rerank_docs, 2),
                   "on_pareto": t.iteration in pareto_iters, "state": t.state}
            row.update({f"p_{k}": v for k, v in t.params.items()})
            rows.append(row)
    write_csv(rows, out_dir / "trials.csv")

    best_overall = max(
        (r.best for r in runs if r.best), key=lambda t: t.ndcg_at_10, default=None
    )
    if best_overall is not None:
        from ragtune.tuning.search_space import RAGtuneSearchSpace

        space = RAGtuneSearchSpace(**search_space_overrides(args))
        (out_dir / "best_config.yaml").write_text(yaml.dump(
            {"pipeline": space.to_pipeline_dict(best_overall.params)}, sort_keys=False
        ))

    per_seed_best = [r.best.ndcg_at_10 for r in runs if r.best]
    per_seed_gain = [r.gain for r in runs if r.best]
    baselines = [r.baseline.ndcg_at_10 for r in runs]

    # Same schema prompt_optimizer.py writes, so the two runs can be diffed
    # field by field without reshaping either side.
    summary = {
        "run_type": f"bayesian-{args.sampler}",
        "searches": "pipeline_configuration",
        "dataset": args.dataset,
        "retriever": args.retriever,
        "n_queries": args.n_queries,
        "rerank_depth": args.rerank_depth,
        "retrieval_depth": args.retrieval_depth,
        "iterations": args.iterations,
        "seeds": [r.seed for r in runs],
        "baseline_ndcg_mean": round(_mean(baselines), 4),
        "best_ndcg_mean": round(_mean(per_seed_best), 4),
        "best_ndcg_stdev": round(_stdev(per_seed_best), 4),
        "gain_mean": round(_mean(per_seed_gain), 4),
        "gain_stdev": round(_stdev(per_seed_gain), 4),
        "evaluations_per_seed": args.iterations + 1,
        "wall_seconds_total": round(sum(r.wall_seconds for r in runs), 1),
        "hypervolume_mean": round(_mean(
            hypervolume_2d([(t.ndcg_at_10, t.mean_rerank_docs) for t in r.pareto],
                           args.max_cost) for r in runs
        ), 3),
        "pareto_size_mean": round(_mean(len(r.pareto) for r in runs), 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "history.json").write_text(json.dumps({
        "run_type": summary["run_type"],
        "dataset": args.dataset,
        "seeds": {str(r.seed): {
            "baseline": r.baseline.to_dict(),
            "wall_seconds": round(r.wall_seconds, 1),
            "trials": [t.to_dict() for t in r.trials],
        } for r in runs},
    }, indent=2))

    print("\n" + "=" * 88)
    print(f"  Bayesian optimization ({args.sampler.upper()}) — {args.dataset} / {args.retriever}")
    print("=" * 88)
    print(f"  {'seed':>6} {'baseline':>10} {'best':>10} {'gain':>9} {'cost':>8} "
          f"{'|front|':>8} {'wall s':>8}")
    print("  " + "-" * 70)
    for run in runs:
        best = run.best
        print(f"  {run.seed:>6} {run.baseline.ndcg_at_10:>10.4f} "
              f"{best.ndcg_at_10 if best else 0.0:>10.4f} {run.gain:>+9.4f} "
              f"{best.mean_rerank_docs if best else 0.0:>8.1f} "
              f"{len(run.pareto):>8} {run.wall_seconds:>8.0f}")
    print("  " + "-" * 70)
    print(f"  {'mean':>6} {summary['baseline_ndcg_mean']:>10.4f} "
          f"{summary['best_ndcg_mean']:>10.4f} {summary['gain_mean']:>+9.4f}"
          f"   +/- {summary['gain_stdev']:.4f} over {len(runs)} seed(s)")
    print("=" * 88)
    print(f"  hypervolume {summary['hypervolume_mean']:.2f} "
          f"(ref cost {args.max_cost:.0f})   "
          f"Pareto size {summary['pareto_size_mean']:.1f}   "
          f"{summary['evaluations_per_seed']} evaluations/seed")
    print(f"  written to {out_dir}/")
    if len(runs) < 3:
        print("  NOTE: fewer than 3 seeds — a gain under ~0.02 NDCG is within noise.")
    print("=" * 88)


def compare_config(args, retriever_factory, out_dir: Path) -> int:
    """Re-evaluate a tuned config against the baseline across seeds."""
    from ragtune.tuning.evaluator import EvalDataset

    cfg = yaml.safe_load(Path(args.compare).read_text())
    tuned = params_from_pipeline_yaml(cfg, args)

    rows = []
    for seed in args.seed_list:
        eval_ds = EvalDataset.from_pyterrier_irds(
            irds_id=DATASETS[args.dataset]["irds_id"], n_queries=args.n_queries, seed=seed
        )
        retriever = retriever_factory()
        for label, params in (("baseline", baseline_params(args)), ("tuned", tuned)):
            t = evaluate_params(params, retriever, eval_ds, args.n_queries)
            rows.append({"seed": seed, "config": label,
                         "ndcg_at_10": round(t.ndcg_at_10, 4),
                         "mean_rerank_docs": round(t.mean_rerank_docs, 2)})
            print(f"  seed {seed}  {label:<9} NDCG@10={t.ndcg_at_10:.4f}  "
                  f"cost={t.mean_rerank_docs:.1f}")

    print("\n" + "=" * 72)
    print(f"  {'config':<10} {'mean NDCG@10':>13} {'stdev':>8} {'seeds':>6}")
    print("-" * 72)
    means = {}
    for label in ("baseline", "tuned"):
        vals = [r["ndcg_at_10"] for r in rows if r["config"] == label]
        means[label] = _mean(vals)
        print(f"  {label:<10} {means[label]:>13.4f} {_stdev(vals):>8.4f} {len(vals):>6}")
    print("-" * 72)
    print(f"  tuned - baseline = {means['tuned'] - means['baseline']:+.4f}")
    if len(args.seed_list) < 3:
        print("  (fewer than 3 seeds — indicative only)")
    print("=" * 72)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2))
    return 0


def params_from_pipeline_yaml(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Flatten a pipeline YAML back into the flat params dict build_controller wants.

    to_pipeline_dict() is lossy in one direction only — the nested form keeps
    everything the builders read, so anything absent falls back to the baseline.
    """
    params = dict(baseline_params(args))
    pipeline = cfg.get("pipeline", cfg)
    comps = pipeline.get("components", {})
    limits = pipeline.get("budget", {}).get("limits", {})

    def comp(name: str) -> Dict[str, Any]:
        return comps.get(name, {}) or {}

    if comp("reranker"):
        params["reranker_type"] = comp("reranker").get("type", params["reranker_type"])
        rp = comp("reranker").get("params", {}) or {}
        if "model_name" in rp:
            key = "ce_model" if params["reranker_type"] == "cross-encoder" else (
                "monot5_model" if params["reranker_type"] == "monot5" else "llm_reranker_model")
            params[key] = rp["model_name"]
        if "batch_size" in rp:
            params["monot5_batch_size"] = rp["batch_size"]
    for slot, key in (("reformulator", "reformulator_type"), ("estimator", "estimator_type"),
                      ("scheduler", "scheduler_type")):
        if comp(slot):
            params[key] = comp(slot).get("type", params[key])
    sp = comp("scheduler").get("params", {}) or {}
    params["scheduler_batch_size"] = sp.get("batch_size", params["scheduler_batch_size"])
    params["gd_llm_limit"] = sp.get("llm_limit", params["gd_llm_limit"])
    params["gd_ce_limit"] = sp.get("cross_encoder_limit", params["gd_ce_limit"])
    ap = comp("assembler").get("params", {}) or {}
    params["assembler_max_docs"] = ap.get("max_docs", params["assembler_max_docs"])
    if pipeline.get("feedback"):
        params["feedback_type"] = pipeline["feedback"].get("type", params["feedback_type"])
    params["budget_rerank_docs"] = limits.get("rerank_docs", params["budget_rerank_docs"])
    params["budget_reformulations"] = limits.get("reformulations", params["budget_reformulations"])
    return params


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def print_plan(args) -> None:
    evals = (args.iterations + 1) * len(args.seed_list)
    print("=" * 88)
    print(f"  RAGtune Bayesian optimization — plan")
    print("=" * 88)
    print(f"  dataset    : {args.dataset}   retriever: {args.retriever}")
    print(f"  sampler    : {args.sampler}   search space: {args.space}")
    print(f"  rerankers  : {', '.join(args.rerankers)}")
    print(f"  budget     : {args.iterations} trials + 1 baseline, x {len(args.seed_list)} seed(s)")
    print(f"  seeds      : {', '.join(map(str, args.seed_list))}")
    print(f"  evaluations: {evals}  ({evals * args.n_queries:,} query evaluations)")
    if "llm" in args.rerankers:
        calls = evals * args.n_queries * args.rerank_depth
        print(f"  LLM reranker selected — up to ~{calls:,} API calls "
              f"({args.rerank_depth} per query)")
    print("=" * 88)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bayesian (Optuna TPE) pipeline optimization on BEIR — the "
                    "configuration-search counterpart to prompt_optimizer.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="nfcorpus",
                   help=f"BEIR dataset; choices: {', '.join(list(DATASETS)[:8])}, …")
    p.add_argument("--retriever", default="bm25",
                   help=f"first-stage retriever (fixed); choices: {', '.join(RETRIEVERS)}")

    e = p.add_argument_group("evaluation")
    e.add_argument("--n-queries", type=int, default=50, help="queries per evaluation")
    e.add_argument("--rerank-depth", type=int, default=50,
                   help="baseline rerank budget (docs/query); for --rerankers llm "
                        "this is also API calls per query")
    e.add_argument("--retrieval-depth", type=int, default=200,
                   help="first-stage depth; also caps the tuner's depth range")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--seeds", default=None,
                   help="comma-separated seeds to repeat the whole run over")

    o = p.add_argument_group("optimization")
    o.add_argument("--iterations", type=int, default=50,
                   help="Optuna trials per seed (matches prompt_optimizer --iterations)")
    o.add_argument("--sampler", default="tpe", choices=["tpe", "random"],
                   help="'random' is the control arm: same space and budget, no surrogate")
    o.add_argument("--rerankers", default="noop,cross-encoder",
                   help="reranker types the tuner may choose from")
    o.add_argument("--space", default="restricted", choices=["restricted", "full"],
                   help="'full' adds llm_rewrite/reformir components — needs an API key")
    o.add_argument("--max-cost", type=float, default=200.0,
                   help="cost pruner threshold and hypervolume reference")
    o.add_argument("--max-trial-seconds", type=float, default=1800.0,
                   help="runtime pruner: abort a trial projected to exceed this")
    o.add_argument("--pareto-pruning", action="store_true",
                   help="also prune trials predicted to be Pareto-dominated")
    o.add_argument("--storage", default=None,
                   help="Optuna storage URL, e.g. sqlite:///study.db (resumable, "
                        "inspectable with optuna-dashboard)")
    o.add_argument("--ce-models", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    o.add_argument("--monot5-models", default="castorini/monot5-base-msmarco")
    o.add_argument("--llm-reranker-model", default="gpt-4o-mini")
    o.add_argument("--search-space-json", default=None,
                   help="JSON file of RAGtuneSearchSpace overrides, merged last")

    i = p.add_argument_group("io")
    i.add_argument("--index-dir", default="./indexes")
    i.add_argument("--out-dir", default="./bayes_opt_results")
    i.add_argument("--compare", default=None,
                   help="evaluate this pipeline YAML against the baseline and exit")
    i.add_argument("--dry-run", action="store_true", help="print the plan and exit")

    args = p.parse_args(argv)

    # Flags the shared helpers expect but this script doesn't expose.
    args.bm25_k1, args.bm25_b = 0.9, 0.4
    args.dense_backend, args.encode_batch_size, args.device = "np", 32, "cpu"
    args.hybrid_sparse, args.hybrid_dense, args.rrf_k = "bm25", "dense-minilm", 60

    if args.dataset not in DATASETS:
        p.error(f"unknown dataset {args.dataset!r}")
    if args.retriever not in RETRIEVERS:
        p.error(f"unknown retriever {args.retriever!r}")
    args.rerankers = [r.strip() for r in args.rerankers.split(",") if r.strip()]
    args.ce_models = [m.strip() for m in args.ce_models.split(",") if m.strip()]
    args.monot5_models = [m.strip() for m in args.monot5_models.split(",") if m.strip()]
    args.seed_list = (
        [int(x) for x in str(args.seeds).split(",") if x.strip()] if args.seeds else [args.seed]
    )
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    print_plan(args)
    if args.dry_run:
        return 0

    pt = init_pyterrier()
    import ragtune.adapters  # noqa: F401 — registry side effects
    import ragtune.components  # noqa: F401
    from ragtune.tuning.evaluator import EvalDataset

    build_sparse_index(pt, args.dataset, args.index_dir)

    def retriever_factory():
        return make_retriever(pt, args.retriever, args.dataset, args.index_dir, args)

    out_dir = Path(args.out_dir)
    if args.compare:
        return compare_config(args, retriever_factory, out_dir)

    runs: List[SeedRun] = []
    for seed in args.seed_list:
        eval_ds = EvalDataset.from_pyterrier_irds(
            irds_id=DATASETS[args.dataset]["irds_id"], n_queries=args.n_queries, seed=seed
        )
        print(f"  [seed {seed}] {len(eval_ds.queries)} topics loaded")
        runs.append(run_seed(seed, retriever_factory(), eval_ds, args, out_dir))

    write_results(runs, args, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
