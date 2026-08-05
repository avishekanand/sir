from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

tune_app = typer.Typer(help="Bayesian multi-objective optimization of the RAGtune pipeline.")
console = Console()


@tune_app.command("run")
def tune_run(
    study_yaml: Path = typer.Argument(..., help="Path to the tuning study YAML config."),
    retriever_index: Optional[str] = typer.Option(
        None,
        "--index",
        "-i",
        help="PyTerrier index path for the fixed retriever (overrides study config).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print search space cardinality and exit without running trials.",
    ),
    n_trials: Optional[int] = typer.Option(
        None,
        "--n-trials",
        help="Override n_trials from study config.",
    ),
    n_queries: Optional[int] = typer.Option(
        None,
        "--n-queries",
        help="Override n_eval_queries from study config.",
    ),
):
    """
    Run a Bayesian multi-objective optimization study over the RAGtune pipeline.

    Example
    -------
    ragtune tune run examples/tune_trec_covid.yaml --index ./index
    """
    if not study_yaml.exists():
        console.print(f"[bold red]Error:[/bold red] Study file not found: {study_yaml}")
        raise typer.Exit(code=1)

    from ragtune.tuning.study_config import TuningStudyConfig
    from ragtune.tuning.search_space import RAGtuneSearchSpace

    try:
        cfg = TuningStudyConfig.from_yaml(str(study_yaml))
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if n_trials is not None:
        cfg.n_trials = n_trials
    if n_queries is not None:
        cfg.n_eval_queries = n_queries

    search_space = RAGtuneSearchSpace(**cfg.search_space_overrides)

    if dry_run:
        console.print(Panel(
            f"Study:        [bold]{cfg.name}[/bold]\n"
            f"Dataset:      {cfg.dataset.name} / {cfg.dataset.split}\n"
            f"Trials:       {cfg.n_trials} ({cfg.n_startup_trials} random startup)\n"
            f"Eval queries: {cfg.n_eval_queries}\n"
            f"Cardinality:  {search_space.get_cardinality():,} discrete combos\n"
            f"Parallelism:  {cfg.n_parallel_workers} worker(s)\n"
            f"Storage:      {cfg.storage_url or 'in-memory'}",
            title="[bold blue]Dry Run — Study Config[/bold blue]",
        ))
        return

    # ── Build fixed retriever ─────────────────────────────────────────────────
    index_path = retriever_index or _infer_index_path(cfg)
    if index_path is None:
        console.print(
            "[bold red]Error:[/bold red] No retriever index provided.\n"
            "Pass --index <path> or set index_path in the study YAML."
        )
        raise typer.Exit(code=1)

    try:
        fixed_retriever = _build_pyterrier_retriever(index_path)
    except Exception as exc:
        console.print(f"[bold red]Retriever error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # ── Load eval dataset ─────────────────────────────────────────────────────
    try:
        eval_dataset = _load_dataset(cfg)
    except Exception as exc:
        console.print(f"[bold red]Dataset error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Starting study:[/bold green] {cfg.name}")
    console.print(f"  Dataset:  {eval_dataset.name} ({len(eval_dataset.queries)} queries loaded)")
    console.print(f"  Trials:   {cfg.n_trials}")
    console.print(f"  Storage:  {cfg.storage_url or 'in-memory'}")

    # ── Run ───────────────────────────────────────────────────────────────────
    from ragtune.tuning.optimizer import run_study, extract_pareto_configs

    try:
        study = run_study(cfg, fixed_retriever, eval_dataset)
    except Exception as exc:
        console.print(f"[bold red]Study error:[/bold red] {exc}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)

    # ── Report results ────────────────────────────────────────────────────────
    paths = extract_pareto_configs(study, search_space, cfg.output_dir)

    table = Table(title="Pareto Front", show_header=True, header_style="bold blue")
    table.add_column("Trial", style="dim")
    table.add_column("NDCG@10", justify="right")
    table.add_column("Mean Rerank Docs", justify="right")
    table.add_column("Config File")

    for path, trial in zip(paths, study.best_trials):
        if trial.values is None:
            continue
        table.add_row(
            str(trial.number),
            f"{trial.values[0]:.4f}",
            f"{trial.values[1]:.1f}",
            Path(path).name,
        )

    console.print(table)
    console.print(f"\n[bold green]Done.[/bold green] Pareto configs written to: {cfg.output_dir}/")


@tune_app.command("llm-run")
def tune_llm_run(
    config_yaml: Path = typer.Argument(..., help="Path to LLM optimizer config YAML."),
    retriever_index: Optional[str] = typer.Option(
        None, "--index", "-i", help="PyTerrier index path for the fixed retriever."
    ),
    n_iterations: Optional[int] = typer.Option(
        None, "--n-iterations", help="Override n_iterations from config."
    ),
    n_queries: Optional[int] = typer.Option(
        None, "--n-queries", help="Override n_eval_queries from config."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override llm_model (any litellm-compatible name)."
    ),
):
    """
    Run LLM-agent-based optimization of the RAGtune pipeline.

    The agent reflects on the evaluation history and Pareto front after each
    iteration and proposes a new configuration to evaluate.

    Example
    -------
    ragtune tune llm-run examples/tune_trec_covid_llm.yaml --index ./index
    """
    if not config_yaml.exists():
        console.print(f"[bold red]Error:[/bold red] Config file not found: {config_yaml}")
        raise typer.Exit(code=1)

    from ragtune.tuning.llm_optimizer import LLMOptimizerConfig, LLMAgentOptimizer, compute_pareto_front

    try:
        cfg = LLMOptimizerConfig.from_yaml(str(config_yaml))
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if n_iterations is not None:
        cfg.n_iterations = n_iterations
    if n_queries is not None:
        cfg.n_eval_queries = n_queries
    if model is not None:
        cfg.llm_model = model

    # ── Build fixed retriever ─────────────────────────────────────────────────
    index_path = retriever_index or cfg.search_space_overrides.get("index_path")
    if index_path is None:
        console.print(
            "[bold red]Error:[/bold red] No retriever index provided.\n"
            "Pass --index <path> or set index_path in search_space_overrides."
        )
        raise typer.Exit(code=1)

    try:
        fixed_retriever = _build_pyterrier_retriever(index_path)
    except Exception as exc:
        console.print(f"[bold red]Retriever error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # ── Load eval dataset ─────────────────────────────────────────────────────
    try:
        eval_dataset = _load_llm_dataset(cfg)
    except Exception as exc:
        console.print(f"[bold red]Dataset error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Starting LLM optimizer:[/bold green] {cfg.name}")
    console.print(f"  Model:      {cfg.llm_model}")
    console.print(f"  Dataset:    {eval_dataset.name} ({len(eval_dataset.queries)} queries)")
    console.print(f"  Iterations: {cfg.n_iterations}")
    console.print(f"  Output:     {cfg.output_dir}/")

    # ── Run ───────────────────────────────────────────────────────────────────
    optimizer = LLMAgentOptimizer(config=cfg)
    try:
        history = optimizer.run(fixed_retriever, eval_dataset)
    except Exception as exc:
        console.print(f"[bold red]Optimizer error:[/bold red] {exc}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)

    # ── Report ────────────────────────────────────────────────────────────────
    pareto = compute_pareto_front(history)
    table = Table(title="LLM Agent Pareto Front", show_header=True, header_style="bold blue")
    table.add_column("Iter", style="dim")
    table.add_column("NDCG@10", justify="right")
    table.add_column("Mean Rerank Docs", justify="right")
    table.add_column("Reranker")
    table.add_column("Scheduler")

    for c in sorted(pareto, key=lambda x: -x.ndcg_at_10):
        table.add_row(
            str(c.iteration),
            f"{c.ndcg_at_10:.4f}",
            f"{c.mean_rerank_docs:.1f}",
            c.params.get("reranker_type", "?"),
            c.params.get("scheduler_type", "?"),
        )

    console.print(table)
    console.print(f"\n[bold green]Done.[/bold green] Pareto configs written to: {cfg.output_dir}/")
    console.print(
        f"History: {len(history)} iterations, "
        f"{sum(1 for c in history if not c.error)} successful, "
        f"{sum(1 for c in history if c.error)} errored."
    )


@tune_app.command("show")
def tune_show(
    storage_url: str = typer.Argument(..., help="Optuna storage URL (e.g. sqlite:///path.db)."),
    study_name: str = typer.Option(..., "--name", "-n", help="Name of the study to show."),
):
    """Show the Pareto front of a completed or in-progress study."""
    try:
        import optuna
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as exc:
        console.print(f"[bold red]Error loading study:[/bold red] {exc}")
        raise typer.Exit(code=1)

    table = Table(title=f"Pareto Front: {study_name}", show_header=True, header_style="bold blue")
    table.add_column("Trial")
    table.add_column("NDCG@10", justify="right")
    table.add_column("Mean Rerank Docs", justify="right")

    for trial in study.best_trials:
        if trial.values is None:
            continue
        table.add_row(
            str(trial.number),
            f"{trial.values[0]:.4f}",
            f"{trial.values[1]:.1f}",
        )
    console.print(table)

    n_complete = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    console.print(f"\nTrials: {n_complete} complete, {n_pruned} pruned, {len(study.trials)} total")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_index_path(cfg: object) -> Optional[str]:
    overrides = getattr(cfg, "search_space_overrides", {})
    return overrides.get("index_path")


def _build_pyterrier_retriever(index_path: str) -> object:
    import ragtune.components  # noqa — populate registry
    import ragtune.adapters    # noqa

    try:
        import pyterrier as pt  # type: ignore
        if not pt.started():
            pt.init()
        from ragtune.adapters.pyterrier import PyTerrierRetriever  # type: ignore
        return PyTerrierRetriever(index_path=index_path)
    except ImportError:
        raise RuntimeError(
            "PyTerrier is required for the fixed retriever. "
            "Install with: pip install python-terrier"
        )


def _load_dataset(cfg: object) -> object:
    from ragtune.tuning.evaluator import EvalDataset
    from ragtune.tuning.study_config import TuningStudyConfig

    assert isinstance(cfg, TuningStudyConfig)
    dataset_cfg = cfg.dataset

    if dataset_cfg.irds_id:
        irds_id = dataset_cfg.irds_id
    else:
        # ir-datasets BEIR entries have no /test sub-path; the test split
        # is the only split and is accessed directly from the top-level ID.
        _IRDS_ALIASES = {
            "trec-covid": "irds:beir/trec-covid",
            "nfcorpus":   "irds:beir/nfcorpus",
            "scifact":    "irds:beir/scifact",
            "fiqa":       "irds:beir/fiqa",
            "arguana":    "irds:beir/arguana",
        }
        irds_id = _IRDS_ALIASES.get(dataset_cfg.name)
        if irds_id is None:
            irds_id = dataset_cfg.name  # pass through as-is

    return EvalDataset.from_pyterrier_irds(
        irds_id=irds_id,
        n_queries=cfg.n_eval_queries,
        seed=cfg.seed,
    )


_IRDS_ALIASES = {
    "trec-covid": "irds:beir/trec-covid",
    "nfcorpus":   "irds:beir/nfcorpus",
    "scifact":    "irds:beir/scifact",
    "fiqa":       "irds:beir/fiqa",
    "arguana":    "irds:beir/arguana",
}


def _load_llm_dataset(cfg: object) -> object:
    from ragtune.tuning.evaluator import EvalDataset
    from ragtune.tuning.llm_optimizer import LLMOptimizerConfig

    assert isinstance(cfg, LLMOptimizerConfig)
    dataset_info = cfg.dataset  # Dict[str, Any]
    irds_id = dataset_info.get("irds_id") or _IRDS_ALIASES.get(dataset_info.get("name", "")) or dataset_info.get("name", "")

    return EvalDataset.from_pyterrier_irds(
        irds_id=irds_id,
        n_queries=cfg.n_eval_queries,
        seed=cfg.seed,
    )
