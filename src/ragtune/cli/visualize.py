"""
Pipeline visualization and interactive editing for RAGtune.

Provides ASCII flow diagram rendering and component-level editing
with diff preview before saving changes.
"""

import copy
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from ragtune.registry import registry

console = Console()

# Component display order for the flow diagram
COMPONENT_ORDER = ["retriever", "reformulator", "reranker", "assembler"]
SUPPORT_COMPONENTS = ["scheduler", "estimator"]


class ComponentBox:
    """Renders a single component as an ASCII box."""

    WIDTH = 12

    def __init__(self, name: str, comp_type: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.comp_type = comp_type
        self.params = params or {}

    def render(self) -> List[str]:
        """Return lines for the component box."""
        # Available space for type value: WIDTH - len(" type: ") - len("│") = WIDTH - 7
        max_type_len = self.WIDTH - 7
        type_display = self.comp_type[:max_type_len] if len(self.comp_type) > max_type_len else self.comp_type

        lines = [
            f"┌{'─' * self.WIDTH}┐",
            f"│{self.name.upper():^{self.WIDTH}}│",
            f"├{'─' * self.WIDTH}┤",
            f"│ type: {type_display:<{max_type_len}}│",
            f"└{'─' * self.WIDTH}┘",
        ]
        return lines


class PipelineFlowRenderer:
    """Renders the complete pipeline visualization."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = config.get("pipeline", {})
        self.components = self.pipeline.get("components", {})
        self.budget = self.pipeline.get("budget", {}).get("limits", {})
        self.name = self.pipeline.get("name", "Unnamed Pipeline")

    def _get_component_info(self, name: str) -> Tuple[str, Dict[str, Any]]:
        """Extract type and params from a component config."""
        comp = self.components.get(name, {})
        if isinstance(comp, dict):
            return comp.get("type", "noop"), comp.get("params", {})
        return "noop", {}

    def render(self) -> Panel:
        """Render the complete pipeline flow diagram."""
        lines = []

        # Build main flow components
        main_boxes = []
        for comp_name in COMPONENT_ORDER:
            comp_type, params = self._get_component_info(comp_name)
            box = ComponentBox(comp_name, comp_type, params)
            main_boxes.append(box.render())

        # Build support components
        support_boxes = []
        for comp_name in SUPPORT_COMPONENTS:
            comp_type, params = self._get_component_info(comp_name)
            box = ComponentBox(comp_name, comp_type, params)
            support_boxes.append(box.render())

        # Render main flow (horizontal)
        box_height = 5
        arrow = "───▶"

        # Build each row of the main flow
        for row in range(box_height):
            line_parts = []
            for i, box_lines in enumerate(main_boxes):
                line_parts.append(box_lines[row])
                if i < len(main_boxes) - 1:
                    # Add arrow in the middle row
                    if row == 2:
                        line_parts.append(arrow)
                    else:
                        line_parts.append("    ")
            lines.append("  " + "".join(line_parts))

        # Add connection from reranker to estimator/scheduler
        lines.append("  " + " " * 14 + " " * 18 + "              ▲")

        # Render support components (scheduler and estimator)
        # Position them below, with estimator feeding into reranker
        for row in range(box_height):
            scheduler_line = support_boxes[0][row]
            estimator_line = support_boxes[1][row]

            if row == 2:
                # Arrow from estimator to scheduler
                connector = "◀───"
            else:
                connector = "    "

            # Position: some spacing, then scheduler, then estimator
            line = "  " + " " * 18 + scheduler_line + connector + estimator_line
            lines.append(line)

        # Build budget line
        budget_parts = []
        for key, value in self.budget.items():
            if isinstance(value, float) and value == int(value):
                value = int(value)
            budget_parts.append(f"{key}={value}")
        budget_str = " | ".join(budget_parts)

        # Create the content
        content = "\n".join(lines)
        content += f"\n\n  Budget: {budget_str}"

        return Panel(
            content,
            title=f"[bold blue]RAGtune Pipeline: {self.name}[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )


def render_pipeline_flow(config: Dict[str, Any]) -> None:
    """Display the pipeline visualization."""
    renderer = PipelineFlowRenderer(config)
    panel = renderer.render()
    console.print(panel)


def get_available_types(component_name: str) -> List[str]:
    """Get list of available types for a component from the registry."""
    # Force load components
    try:
        import ragtune.adapters  # noqa
        import ragtune.components  # noqa
    except ImportError:
        pass

    all_components = registry.list_all()
    return list(all_components.get(component_name, {}).keys())


def parse_value(value_str: str) -> Any:
    """Auto-detect and parse value type from string input."""
    value_str = value_str.strip()

    # Boolean
    if value_str.lower() in ("true", "yes"):
        return True
    if value_str.lower() in ("false", "no"):
        return False

    # None
    if value_str.lower() in ("none", "null", ""):
        return None

    # Integer
    try:
        return int(value_str)
    except ValueError:
        pass

    # Float
    try:
        return float(value_str)
    except ValueError:
        pass

    # String (default)
    return value_str


def edit_params(component_config: Dict[str, Any]) -> None:
    """Interactive param editing for a component."""
    params = component_config.setdefault("params", {})

    while True:
        console.print("\n[bold]Current Parameters:[/bold]")
        if not params:
            console.print("  [dim](none)[/dim]")
        else:
            for key, value in params.items():
                console.print(f"  {key}: {value}")

        console.print("\n[dim]a=add, m=modify, r=remove, d=done[/dim]")
        action = Prompt.ask("Action", choices=["a", "m", "r", "d"], default="d")

        if action == "d":
            break
        elif action == "a":
            key = Prompt.ask("Parameter name")
            if key:
                value_str = Prompt.ask(f"Value for '{key}'")
                params[key] = parse_value(value_str)
        elif action == "m":
            if not params:
                console.print("[yellow]No parameters to modify[/yellow]")
                continue
            key = Prompt.ask("Parameter to modify", choices=list(params.keys()))
            current = params[key]
            console.print(f"Current value: {current}")
            value_str = Prompt.ask(f"New value for '{key}'", default=str(current))
            params[key] = parse_value(value_str)
        elif action == "r":
            if not params:
                console.print("[yellow]No parameters to remove[/yellow]")
                continue
            key = Prompt.ask("Parameter to remove", choices=list(params.keys()))
            del params[key]
            console.print(f"[dim]Removed '{key}'[/dim]")


def edit_component(config: Dict[str, Any], component_name: str) -> Dict[str, Any]:
    """Edit a single component configuration."""
    components = config.setdefault("pipeline", {}).setdefault("components", {})
    comp = components.setdefault(component_name, {"type": "noop", "params": {}})

    if not isinstance(comp, dict):
        comp = {"type": str(comp), "params": {}}
        components[component_name] = comp

    current_type = comp.get("type", "noop")
    console.print(f"\n[bold]Editing {component_name.upper()}[/bold]")
    console.print(f"Current type: [cyan]{current_type}[/cyan]")

    # Show available types
    available = get_available_types(component_name)
    if available:
        console.print(f"Available types: {', '.join(available)}")

    # Change type?
    if Confirm.ask("Change type?", default=False):
        if available:
            new_type = Prompt.ask(
                "Select type",
                choices=available + [current_type],
                default=current_type
            )
        else:
            new_type = Prompt.ask("Enter type", default=current_type)

        if new_type != current_type:
            comp["type"] = new_type
            # Clear params when type changes
            if Confirm.ask("Clear existing params?", default=True):
                comp["params"] = {}

    # Edit params?
    if Confirm.ask("Edit parameters?", default=False):
        edit_params(comp)

    return config


def edit_budget(config: Dict[str, Any]) -> Dict[str, Any]:
    """Edit budget limits."""
    budget = config.setdefault("pipeline", {}).setdefault("budget", {}).setdefault("limits", {})

    console.print("\n[bold]Editing Budget Limits[/bold]")

    while True:
        console.print("\n[bold]Current Limits:[/bold]")
        if not budget:
            console.print("  [dim](none)[/dim]")
        else:
            for key, value in budget.items():
                console.print(f"  {key}: {value}")

        console.print("\n[dim]a=add, m=modify, r=remove, d=done[/dim]")
        action = Prompt.ask("Action", choices=["a", "m", "r", "d"], default="d")

        if action == "d":
            break
        elif action == "a":
            key = Prompt.ask("Limit name (e.g., tokens, latency_ms)")
            if key:
                value_str = Prompt.ask(f"Value for '{key}'")
                budget[key] = parse_value(value_str)
        elif action == "m":
            if not budget:
                console.print("[yellow]No limits to modify[/yellow]")
                continue
            key = Prompt.ask("Limit to modify", choices=list(budget.keys()))
            current = budget[key]
            console.print(f"Current value: {current}")
            value_str = Prompt.ask(f"New value for '{key}'", default=str(current))
            budget[key] = parse_value(value_str)
        elif action == "r":
            if not budget:
                console.print("[yellow]No limits to remove[/yellow]")
                continue
            key = Prompt.ask("Limit to remove", choices=list(budget.keys()))
            del budget[key]
            console.print(f"[dim]Removed '{key}'[/dim]")

    return config


def show_diff(original: Dict[str, Any], modified: Dict[str, Any]) -> None:
    """Display a colored unified diff between original and modified configs."""
    original_yaml = yaml.dump(original, sort_keys=False, default_flow_style=False)
    modified_yaml = yaml.dump(modified, sort_keys=False, default_flow_style=False)

    original_lines = original_yaml.splitlines(keepends=True)
    modified_lines = modified_yaml.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile="original",
        tofile="modified",
        lineterm=""
    )

    console.print("\n[bold]Changes Preview:[/bold]")

    has_changes = False
    for line in diff:
        has_changes = True
        line = line.rstrip("\n")
        if line.startswith("+") and not line.startswith("+++"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("-") and not line.startswith("---"):
            console.print(f"[red]{line}[/red]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{line}[/cyan]")
        else:
            console.print(line)

    if not has_changes:
        console.print("[dim]No changes detected.[/dim]")


def run_interactive_editor(config_path: Path, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Main interactive editor loop.

    Returns modified config if saved, None if quit without saving.
    """
    original = copy.deepcopy(config)
    modified = copy.deepcopy(config)

    component_names = COMPONENT_ORDER + SUPPORT_COMPONENTS

    while True:
        console.print("\n[bold]Edit Pipeline[/bold]")
        console.print("─" * 20)

        for i, name in enumerate(component_names, 1):
            comp = modified.get("pipeline", {}).get("components", {}).get(name, {})
            comp_type = comp.get("type", "noop") if isinstance(comp, dict) else str(comp)
            console.print(f"{i}. Edit {name.capitalize()} ({comp_type})")

        console.print(f"{len(component_names) + 1}. Edit Budget")
        console.print("─" * 20)
        console.print("d. Show diff")
        console.print("s. Save and exit")
        console.print("q. Quit without saving")

        choice = Prompt.ask("\nSelect option").strip().lower()

        if choice == "q":
            if modified != original:
                if not Confirm.ask("Discard changes?", default=False):
                    continue
            return None

        elif choice == "s":
            show_diff(original, modified)
            if Confirm.ask("\nSave these changes?", default=True):
                return modified
            continue

        elif choice == "d":
            show_diff(original, modified)
            continue

        elif choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(component_names):
                comp_name = component_names[idx - 1]
                modified = edit_component(modified, comp_name)
                # Re-render the visualization
                render_pipeline_flow(modified)
            elif idx == len(component_names) + 1:
                modified = edit_budget(modified)
            else:
                console.print("[yellow]Invalid option[/yellow]")
        else:
            console.print("[yellow]Invalid option[/yellow]")


def save_config(config: Dict[str, Any], path: Path) -> None:
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    console.print(f"[bold green]Saved configuration to {path}[/bold green]")
