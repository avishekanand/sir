from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import List, Dict, Any

console = Console()

def print_header(title: str):
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", expand=False))

def print_step(message: str):
    console.print(f"[bold yellow]→[/bold yellow] {message}")

def print_success(message: str):
    console.print(f"[bold green]✔[/bold green] {message}")

def print_error(message: str):
    console.print(f"[bold red]✘[/bold red] {message}")

def print_documents(documents: List[Any], title: str = "Top Documents"):
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", justify="right")
    table.add_column("Content", ratio=1)

    for i, doc in enumerate(documents):
        # Handle both ScoredDocument and LangChain Document if needed
        content = getattr(doc, "content", getattr(doc, "page_content", str(doc)))
        score = getattr(doc, "score", 0.0)
        reranker_score = getattr(doc, "reranker_score", None)

        display_score = f"{score:.2f}"
        if reranker_score is not None:
             display_score += f" ([bold blue]RR: {reranker_score:.2f}[/bold blue])"

        table.add_row(
            str(i+1),
            display_score,
            content[:150] + "..." if len(content) > 150 else content
        )

    console.print(table)

def print_trace(events: List[Any]):
    table = Table(title="Iteration Trace", show_header=True, header_style="bold cyan")
    table.add_column("Round", style="dim")
    table.add_column("Action")
    table.add_column("Details", ratio=1)

    round_idx = 1
    for event in events:
        action = event.action
        details = str(event.details)

        if action == "rerank_batch":
            doc_ids = event.details.get("doc_ids", [])
            utility = event.details.get("utility", 0.0)
            strategy = event.details.get("strategy", "unknown")
            details = f"[bold green]Batch:[/bold green] {doc_ids} | [bold yellow]Utility:[/bold yellow] {utility:.2f} | [dim]{strategy}[/dim]"
            table.add_row(str(round_idx), "[bold green]Rerank[/bold green]", details)
            round_idx += 1
        elif "deny" in action:
            reason = event.details.get("reason", "unknown")
            table.add_row("-", "[bold red]Budget Denied[/bold red]", f"Reason: {reason}")

    console.print(table)

def print_budget(budget_state: Dict[str, Any]):
    table = Table(title="Final Budget State", show_header=True, header_style="bold green")
    for key in budget_state.keys():
        table.add_column(key.capitalize(), justify="center")

    table.add_row(*[str(v) for v in budget_state.values()])
    console.print(table)
