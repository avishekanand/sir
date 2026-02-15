import typer
import yaml
import os
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from ragtune.registry import registry
from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget

app = typer.Typer(help="RAGtune CLI: Budget-aware RAG middleware.")
console = Console()

@app.command()
def init(
    path: Path = typer.Option(Path("ragtune_config.yaml"), "--output", "-o", help="Output path for the config file.")
):
    """
    Initialize a new RAGtune configuration file.
    """
    default_config = {
        "pipeline": {
            "name": "My First RAGtune Pipeline",
            "components": {
                "retriever": {
                    "type": "bm25",
                    "params": {"index_path": "./index"}
                },
                "reranker": {
                    "type": "cross-encoder",
                    "params": {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
                },
                "reformulator": {"type": "noop"},
                "assembler": {"type": "default"},
                "scheduler": {"type": "default"} # Make sure scheduler is included
            },
            "budget": {
                "tokens": 4000,
                "rerank_docs": 50,
                "latency_ms": 2000.0
            }
        }
    }
    
    if path.exists():
        console.print(f"[bold red]Error:[/bold red] File {path} already exists.")
        raise typer.Exit(code=1)
        
    with open(path, "w") as f:
        yaml.dump(default_config, f, sort_keys=False)
        
    console.print(f"[bold green]Success![/bold green]Created default configuration at {path}")

from ragtune.cli.config_loader import ConfigLoader

@app.command("list")
def list_components():
    """
    List all registered RAGtune components.
    """
    # Force load default components/adapters
    try:
        import ragtune.adapters  # noqa
        import ragtune.components # noqa
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
            doc = cls.__doc__.strip().split('\n')[0] if cls.__doc__ else "No description"
            console.print(f"  - [cyan]{name}[/cyan]: {doc}")

@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to the pipeline configuration file."),
    query: str = typer.Option(..., "--query", "-q", help="The query to run."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed execution trace.")
):
    """
    Run a RAGtune pipeline from a configuration file.
    """
    if not config_path.exists():
        console.print(f"[bold red]Error:[/bold red] Config file {config_path} not found.")
        raise typer.Exit(code=1)
    
    # Force load default components/adapters to ensure registry is populated
    try:
        import ragtune.adapters  # noqa
        import ragtune.components # noqa
    except ImportError:
        pass

    try:
        console.print(f"[dim]Loading config from {config_path}...[/dim]")
        config_data = ConfigLoader.load_config(config_path)
        
        pipeline_conf = config_data.get("pipeline", {})
        name = pipeline_conf.get("name", "Unnamed Pipeline")
        console.print(f"[bold green]Running Pipeline:[/bold green] {name}")
        
        controller = ConfigLoader.create_controller(config_data)
        
    except Exception as e:
         console.print(f"[bold red]Configuration Error:[/bold red] {e}")
         raise typer.Exit(code=1)
         
    console.print(f"[dim]Processing query: {query}[/dim]")
    
    with console.status("[bold green]Executing...[/bold green]"):
        try:
            # Run the controller
            result = controller.run(query)
            
            # Print generic cost usage
            console.print(Panel(str(result.final_budget_state), title="[bold]Final Budget State[/bold]"))
            
            # Print Results
            console.print("\n[bold blue]Top Documents:[/bold blue]")
            for i, doc in enumerate(result.documents, 1):
                console.print(f"{i}. [bold]{doc.id}[/bold] (Score: {doc.score:.4f})")
                console.print(f"   {doc.content[:200]}...")
                console.print(f"   [dim]{doc.metadata}[/dim]\n")
                
            if verbose:
                console.print(Panel(str(result.trace), title="[bold]Execution Trace[/bold]"))

        except Exception as e:
            console.print(f"[bold red]Runtime Error:[/bold red] {e}")
            raise typer.Exit(code=1)

def main():
    app()

if __name__ == "__main__":
    main()
