import typer
import yaml
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from ragtune.registry import registry
from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget

try:
    import ragtune.indexing  # noqa
except ImportError:
    pass

app = typer.Typer(help="RAGtune CLI: Budget-aware RAG middleware.")
console = Console()


@app.command()
def init(
    path: Path = typer.Option(
        Path("ragtune_config.yaml"),
        "--output",
        "-o",
        help="Output path for the config file.",
    ),
    wizard: bool = typer.Option(
        False, "--wizard", "-w", help="Run interactive wizard to construct config."
    ),
):
    """
    Initialize a new RAGtune configuration file.
    """
    if path.exists():
        from rich.prompt import Confirm, Prompt

        console.print(
            f"[bold yellow]Warning:[/bold yellow] File {path} already exists."
        )
        if Confirm.ask(f"Overwrite {path}?", default=False):
            console.print(f"[dim]Overwriting {path}...[/dim]")
        else:
            new_path = Prompt.ask(
                "Enter a new filename", default="ragtune_config_v2.yaml"
            )
            path = Path(new_path)
            if path.exists():
                console.print(
                    f"[bold red]Error:[/bold red] {path} also exists. Aborting."
                )
                raise typer.Exit(code=1)

    if wizard:
        from ragtune.cli.wizard import run_init_wizard

        config_data = run_init_wizard()
    else:
        config_data = {
            "pipeline": {
                "name": "My First RAGtune Pipeline",
                "data": {
                    "collection_path": "./data/bright_sample/corpus.json",
                    "collection_format": "json",
                    "id_field": "doc_id",
                    "text_field": "content",
                    "metadata_fields": ["source"],
                },
                "index": {
                    "framework": "pyterrier",
                    "params": {"index_path": "./index"},
                },
                "components": {
                    "retriever": {
                        "type": "pyterrier",
                        "params": {"index_path": "./index"},
                    },
                    "reranker": {
                        "type": "ollama-listwise",
                        "params": {"model_name": "deepseek-r1:8b"},
                    },
                    "reformulator": {"type": "identity"},
                    "assembler": {"type": "greedy"},
                    "scheduler": {"type": "graceful-degradation"},
                    "estimator": {"type": "baseline"},
                },
                "budget": {
                    "limits": {
                        "tokens": 4000,
                        "rerank_docs": 50,
                        "latency_ms": 2000.0,
                        "retrieval_calls": 5,
                    }
                },
            }
        }

    with open(path, "w") as f:
        yaml.dump(config_data, f, sort_keys=False)

    console.print(f"[bold green]Success![/bold green] Created configuration at {path}")


from ragtune.cli.config_loader import ConfigLoader


@app.command("list")
def list_components():
    """
    List all registered RAGtune components.
    """
    # Force load default components/adapters
    try:
        import ragtune.adapters  # noqa
        import ragtune.components  # noqa
    except ImportError:
        pass

    console.print(Panel("[bold blue]Registered Components[/bold blue]"))

    all_components = registry.list_all()

    for category, components in all_components.items():
        console.print(f"\n[bold]{category.capitalize()}s:[/bold]")
        if not components:
            console.print("  [dim]None[/dim]")
        for name, cls in components.items():
            # Get docstring summary if available
            doc = (
                cls.__doc__.strip().split("\n")[0] if cls.__doc__ else "No description"
            )
            console.print(f"  - [cyan]{name}[/cyan]: {doc}")


@app.command()
def run(
    config_path: Path = typer.Argument(
        ..., help="Path to the pipeline configuration file."
    ),
    query: str = typer.Option(..., "--query", "-q", help="The query to run."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed execution trace."
    ),
    collection_path: Optional[str] = typer.Option(
        None,
        "--collection-path",
        help="Override the data collection path in the config.",
    ),
    limits: Optional[List[str]] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Override budget limits (e.g. --limit tokens=1000). Can be used multiple times.",
    ),
):
    """
    Run a RAGtune pipeline from a configuration file.
    """
    if not config_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Config file {config_path} not found."
        )
        raise typer.Exit(code=1)

    # Parse limits if provided
    budget_overrides = {}
    if limits:
        for limit_str in limits:
            try:
                k, v = limit_str.split("=")
                budget_overrides[k] = float(v)
            except ValueError:
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Invalid limit format: {limit_str}. Expected KEY=VALUE"
                )

    # Force load default components/adapters to ensure registry is populated
    try:
        import ragtune.adapters  # noqa
        import ragtune.components  # noqa
    except ImportError:
        pass

    try:
        console.print(f"[dim]Loading config from {config_path}...[/dim]")
        config_data = ConfigLoader.load_config(config_path)

        if collection_path:
            config_data.setdefault("pipeline", {}).setdefault("data", {})[
                "collection_path"
            ] = collection_path
            console.print(
                f"[dim]Overriding collection_path to: {collection_path}[/dim]"
            )

        pipeline_conf = config_data.get("pipeline", {})
        name = pipeline_conf.get("name", "Unnamed Pipeline")
        console.print(f"[bold green]Running Pipeline:[/bold green] {name}")

        controller = ConfigLoader.create_controller(
            config_data, budget_overrides=budget_overrides
        )

    except Exception as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    console.print(f"[dim]Processing query: {query}[/dim]")

    console.print("[dim]Executing RAGtune pipeline...[/dim]")
    try:
        # Run the controller
        result = controller.run(query)

        # Print generic cost usage
        console.print(
            Panel(
                str(result.final_budget_state), title="[bold]Final Budget State[/bold]"
            )
        )

        # Print Results
        console.print("\n[bold blue]Top Documents:[/bold blue]")
        for i, doc in enumerate(result.documents, 1):
            console.print(f"{i}. [bold]{doc.id}[/bold] (Score: {doc.score:.4f})")
            console.print(f"   {doc.content[:200]}...")
            console.print(f"   [dim]{doc.metadata}[/dim]\n")

        if verbose:
            console.print(
                Panel(str(result.trace), title="[bold]Execution Trace[/bold]")
            )

    except Exception as e:
        console.print(f"[bold red]Runtime Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def validate(
    config_path: Path = typer.Argument(
        ..., help="Path to the configuration file to validate."
    ),
    allow_missing_index: bool = typer.Option(
        False, "--allow-missing-index", help="Don't fail if the index path is missing."
    ),
):
    """
    Validate a RAGtune configuration file against the v0.2 schema.
    """
    if not config_path.exists():
        console.print(f"[bold red]Error:[/bold red] File {config_path} not found.")
        raise typer.Exit(code=1)

    try:
        from ragtune.config.models import RAGtuneConfig

        console.print(f"[dim]Validating schema for {config_path}...[/dim]")
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        # 1. Pydantic Schema Validation
        config_obj = RAGtuneConfig(**data)

        # 2. Registry Check
        # Force load components
        try:
            import ragtune.adapters  # noqa
            import ragtune.components  # noqa
        except ImportError:
            pass

        console.print("[dim]Checking registry for components...[/dim]")
        pipeline = config_obj.pipeline
        components = pipeline.components

        problems = []

        check_list = [
            ("retriever", components.retriever, registry.get_retriever),
            ("reranker", components.reranker, registry.get_reranker),
            ("reformulator", components.reformulator, registry.get_reformulator),
            ("assembler", components.assembler, registry.get_assembler),
            ("scheduler", components.scheduler, registry.get_scheduler),
        ]

        # Handle estimator separately as it can be a list
        estimators = (
            components.estimator
            if isinstance(components.estimator, list)
            else [components.estimator]
        )
        for est in estimators:
            check_list.append(("estimator", est, registry.get_estimator))

        for cat, comp, getter in check_list:
            if not getter(comp.type):
                problems.append(
                    f"Component '{comp.type}' not found in registry for category '{cat}'."
                )

        # 3. Path / Index checks
        if not allow_missing_index and pipeline.index:
            idx_path = pipeline.index.params.get("index_path")
            if idx_path and not Path(idx_path).exists():
                problems.append(
                    f"Index path '{idx_path}' does not exist. Run 'ragtune index' to build it."
                )

        if problems:
            console.print(
                Panel(
                    "\n".join(f"• {p}" for p in problems),
                    title="[bold red]Validation Failed[/bold red]",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

        console.print(
            f"[bold green]Success![/bold green] Configuration {config_path} is valid."
        )

    except Exception as e:
        console.print(f"[bold red]Schema Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def visualize(
    config_path: Path = typer.Argument(
        ..., help="Path to the pipeline configuration file."
    ),
    edit: bool = typer.Option(
        False, "--edit", "-e", help="Enable interactive editing mode."
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for modified config (default: overwrite input).",
    ),
):
    """
    Visualize and optionally edit a RAGtune pipeline configuration.

    Displays an ASCII flow diagram of the pipeline components.
    Use --edit to interactively modify components and see diff before saving.
    """
    if not config_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Config file {config_path} not found."
        )
        raise typer.Exit(code=1)

    from ragtune.cli.visualize import (
        render_pipeline_flow,
        run_interactive_editor,
        show_diff,
        save_config,
    )

    try:
        config_data = ConfigLoader.load_config(config_path)
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        raise typer.Exit(code=1)

    # Display visualization
    render_pipeline_flow(config_data)

    if edit:
        modified = run_interactive_editor(config_path, config_data)
        if modified:
            save_path = output if output else config_path
            save_config(modified, save_path)
    elif output:
        console.print(
            "[yellow]Warning:[/yellow] --output is only used with --edit flag."
        )


@app.command()
def index(
    config_path: Path = typer.Argument(..., help="Path to the configuration file."),
    collection_path: Optional[str] = typer.Option(
        None,
        "--collection-path",
        help="Override the data collection path in the config.",
    ),
):
    """
    Build an index based on the configuration file.
    """
    if not config_path.exists():
        console.print(f"[bold red]Error:[/bold red] File {config_path} not found.")
        raise typer.Exit(code=1)

    try:
        from ragtune.config.models import RAGtuneConfig

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        config_obj = RAGtuneConfig(**data)
        pipeline = config_obj.pipeline

        if collection_path:
            if not pipeline.data:
                from ragtune.config.models import DataConfig

                pipeline.data = DataConfig(
                    collection_path=collection_path, collection_format="json"
                )
            else:
                pipeline.data.collection_path = collection_path
            console.print(
                f"[dim]Overriding collection_path to: {collection_path}[/dim]"
            )

        if not pipeline.data or not pipeline.index:
            console.print(
                "[bold red]Error:[/bold red] Config must have 'pipeline.data' and 'pipeline.index' sections to run indexing."
            )
            raise typer.Exit(code=1)

        # Force load indexers
        try:
            import ragtune.indexing.pyterrier_indexer  # noqa
        except ImportError:
            pass

        framework = pipeline.index.framework
        indexer_cls = registry.get_indexer(framework)

        if not indexer_cls:
            console.print(
                f"[bold red]Error:[/bold red] Indexer framework '{framework}' not found in registry."
            )
            raise typer.Exit(code=1)

        indexer = indexer_cls()

        console.print(f"[bold blue]Building index for {pipeline.name}...[/bold blue]")
        console.print(f" Framework: {framework}")
        console.print(f" Source: {pipeline.data.collection_path}")
        console.print(f" Target: {pipeline.index.params.get('index_path')}")

        with console.status("[bold green]Indexing...[/bold green]"):
            fields = {
                "id_field": pipeline.data.id_field,
                "text_field": pipeline.data.text_field,
                "metadata_fields": {m: m for m in pipeline.data.metadata_fields},
            }
            success = indexer.build(
                collection_path=pipeline.data.collection_path,
                format=pipeline.data.collection_format,
                fields=fields,
                **pipeline.index.params,
            )

        if success:
            console.print("[bold green]Indexing completed successfully![/bold green]")
        else:
            console.print("[bold red]Indexing failed.[/bold red]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[bold red]Indexing Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def budget(
    budget_type: str = typer.Option(
        "vllm",
        "--type",
        "-t",
        help="Budget loader type: vllm, token, gpu_util, carbon, embedding, reranking",
    ),
    prompt_tokens: int = typer.Option(
        512, "--prompt-tokens", help="Number of prompt/input tokens"
    ),
    completion_tokens: int = typer.Option(
        256, "--completion-tokens", help="Number of completion/output tokens"
    ),
    cached_tokens: int = typer.Option(
        0, "--cached-tokens", help="Number of cached input tokens"
    ),
    gpu_type: Optional[str] = typer.Option(
        None, "--gpu", help="GPU type (e.g. H100-NVL-96GB, A100-80GB)"
    ),
    gpu_count: int = typer.Option(1, "--gpu-count", help="Number of GPUs"),
    gpu_hourly_rate: Optional[float] = typer.Option(
        None, "--gpu-rate", help="GPU hourly rate override ($)"
    ),
    model_name: Optional[str] = typer.Option(
        None, "--model", help="Model name (e.g. llama-3.1-8b)"
    ),
    offered_rps: float = typer.Option(
        10.0, "--rps", help="Offered request rate (requests/sec)"
    ),
    latency_slo_ms: int = typer.Option(
        300, "--slo", help="Latency SLO in milliseconds"
    ),
    region: Optional[str] = typer.Option(
        None, "--region", help="Cloud region for carbon intensity"
    ),
    pue: Optional[float] = typer.Option(
        None, "--pue", help="Power Usage Effectiveness (default 1.15)"
    ),
    cache_hit_rate: Optional[float] = typer.Option(
        None, "--cache-hit-rate", help="Cache hit rate (0.0-1.0)"
    ),
    embedding_model: Optional[str] = typer.Option(
        None,
        "--embedding-model",
        help="Embedding model (e.g. openai/text-embedding-3-small)",
    ),
    reranking_model: Optional[str] = typer.Option(
        None, "--reranking-model", help="Reranking model (e.g. cohere/rerank-v4-pro)"
    ),
    queries: int = typer.Option(
        1, "--queries", help="Number of queries (for reranking)"
    ),
    docs_per_query: int = typer.Option(
        10, "--docs", help="Docs per query (for reranking)"
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML budget config"
    ),
    suggest: bool = typer.Option(
        False, "--suggest", "-s", help="Show optimization suggestions"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed breakdown"
    ),
):
    """
    Estimate LLM inference cost, energy, and carbon footprint.

    Examples:
        ragtune budget
        ragtune budget --type vllm --gpu H100-NVL-96GB --model llama-3.1-8b --rps 25
        ragtune budget --type token --prompt-tokens 1024 --completion-tokens 512
        ragtune budget --type carbon --region eu-france --gpu A100-80GB
        ragtune budget --type embedding --embedding-model openai/text-embedding-3-small
        ragtune budget --type reranking --reranking-model cohere/rerank-v4-pro --queries 10
        ragtune budget --suggest  # show optimization suggestions
    """
    from ragtune.budget.main import calculate_budget, format_report
    from ragtune.budget.base import BudgetConfig

    # Build config from CLI options
    config_dict = {}
    if gpu_type:
        config_dict["gpu_type"] = gpu_type
    config_dict["gpu_count"] = gpu_count
    if gpu_hourly_rate is not None:
        config_dict["gpu_hourly_rate"] = gpu_hourly_rate
    if model_name:
        config_dict["model_name"] = model_name
    config_dict["offered_rps"] = offered_rps
    config_dict["latency_slo_ms"] = latency_slo_ms
    if region:
        config_dict["region"] = region
    if pue is not None:
        config_dict["pue"] = pue
    if cache_hit_rate is not None:
        config_dict["cache_hit_rate"] = cache_hit_rate

    # Extra config for embedding/reranking models
    extra = {}
    if embedding_model:
        extra["embedding_model"] = embedding_model
    if reranking_model:
        extra["reranking_model"] = reranking_model
    if extra:
        config_dict["extra"] = extra

    # Load from YAML if provided (CLI options override YAML values)
    if config_path and config_path.exists():
        import yaml as _yaml

        with open(config_path) as f:
            yaml_cfg = _yaml.safe_load(f) or {}
        # YAML values are defaults; CLI options take precedence
        for k, v in yaml_cfg.items():
            if k not in config_dict:
                config_dict[k] = v

    try:
        # Build context for embedding/reranking loaders
        context = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
        }
        if budget_type == "embedding":
            context["tokens"] = prompt_tokens  # embedding processes input tokens
        elif budget_type == "reranking":
            context["queries"] = queries
            context["docs_per_query"] = docs_per_query

        # Compute once, format once (budget_report would recompute)
        result = calculate_budget(
            budget_type=budget_type,
            config=config_dict if config_dict else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
        console.print(format_report(result, budget_type))

        # Suggestions
        if suggest:
            from ragtune.budget.optimizer import (
                suggest_optimizations,
                format_suggestions,
            )

            suggestions = suggest_optimizations(result, config_dict)
            console.print(format_suggestions(suggestions))

        # Verbose: show raw BudgetResult
        if verbose:
            console.print(Panel(str(result), title="[bold]Raw BudgetResult[/bold]"))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
