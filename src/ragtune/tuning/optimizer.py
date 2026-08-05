from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ragtune.tuning.evaluator import EvalDataset, TrialEvaluator
from ragtune.tuning.pruners import CostPruner, ParetoPruner, RuntimePruner
from ragtune.tuning.search_space import RAGtuneSearchSpace
from ragtune.tuning.study_config import TuningStudyConfig


def run_study(
    config: TuningStudyConfig,
    fixed_retriever: Any,
    eval_dataset: EvalDataset,
) -> Any:
    """
    Run a Bayesian multi-objective optimization study.

    Returns the completed optuna.Study.  Pareto-optimal configs can be
    extracted with extract_pareto_configs().

    Parameters
    ----------
    config
        Study configuration (trial counts, pruner thresholds, etc.).
    fixed_retriever
        A pre-built BaseRetriever instance shared across all trials.
        Only the reranking / scheduling / estimation pipeline is tuned.
    eval_dataset
        The queries and qrels used to score each trial.
    """
    import optuna
    from optuna.samplers import TPESampler

    sampler = TPESampler(
        multivariate=True,
        constant_liar=True,
        n_startup_trials=config.n_startup_trials,
        seed=config.seed,
    )

    study = optuna.create_study(
        study_name=config.name,
        directions=["maximize", "minimize"],
        sampler=sampler,
        storage=config.storage_url,
        load_if_exists=True,
    )

    search_space = RAGtuneSearchSpace(**config.search_space_overrides)

    pruners = [
        CostPruner(
            max_mean_rerank_docs=config.max_mean_rerank_docs,
            warmup_steps=3,
        ),
        RuntimePruner(
            max_trial_seconds=config.max_trial_seconds,
            warmup_steps=3,
        ),
        ParetoPruner(
            study=study,
            warmup_trials=config.pareto_warmup_trials,
            zscore=1.645,
        ),
    ]

    evaluator = TrialEvaluator(
        dataset=eval_dataset,
        n_eval_queries=config.n_eval_queries,
        pruners=pruners,
    )

    def objective(trial: Any) -> tuple:
        params = search_space.sample(trial)

        try:
            controller = search_space.build_controller(params, fixed_retriever)
        except Exception as exc:
            # Component construction failed (e.g. model not available).
            # Return worst-case values so the trial is completed but ignored.
            trial.set_user_attr("build_error", str(exc))
            return 0.0, float("inf")

        retrieval_overrides = search_space.to_retrieval_overrides(params)
        objectives = evaluator.evaluate(controller, trial, retrieval_overrides)

        trial.set_user_attr("latency_ms", objectives.latency_ms)
        trial.set_user_attr("queries_evaluated", objectives.queries_evaluated)

        return objectives.ndcg_at_10, objectives.rerank_docs

    study.optimize(
        objective,
        n_trials=config.n_trials,
        n_jobs=config.n_parallel_workers,
        catch=(Exception,),
    )

    return study


def extract_pareto_configs(
    study: Any,
    search_space: RAGtuneSearchSpace,
    output_dir: str,
) -> List[str]:
    """
    Write each Pareto-front trial's pipeline config to output_dir as YAML.

    Files are named: pareto_<trial_number>_ndcg<val>_cost<val>.yaml
    Returns a list of written file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pareto_trials = study.best_trials  # non-dominated trials, built-in for multi-objective
    paths = []

    for trial in pareto_trials:
        if trial.values is None:
            continue
        ndcg_val = trial.values[0]
        cost_val = trial.values[1]

        pipeline_dict = search_space.to_pipeline_dict(trial.params)

        filename = (
            f"pareto_trial_{trial.number}"
            f"_ndcg{ndcg_val:.3f}"
            f"_cost{cost_val:.0f}"
            ".yaml"
        )
        path = os.path.join(output_dir, filename)

        with open(path, "w") as f:
            yaml.dump({"pipeline": pipeline_dict}, f, sort_keys=False)

        paths.append(path)

    return paths
