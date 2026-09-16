"""Regression tests: every documented CLI command must be registered.

The `budget` command was originally defined AFTER the
`if __name__ == "__main__": main()` guard in cli/main.py. That made it
invisible to `python -m ragtune.cli.main` (the module executes top-to-bottom,
so main() ran before the decorator registered the command), while the
console-script entry point (`ragtune budget`) still worked because it imports
the module first. These tests pin the invariant: registration must happen at
import time for ALL commands, regardless of invocation style.
"""

import typer.main

from ragtune.cli.main import app

# Every command documented in docs/ (budget.md, cli.md) and README.
DOCUMENTED_COMMANDS = [
    "init",
    "list",
    "run",
    "validate",
    "visualize",
    "index",
    "budget",
]


def _registered_names():
    """Resolve the app into a click command to get FINAL command names.

    typer's registered_commands hold CommandInfo objects whose .name is None
    unless explicitly overridden; the effective names come from the function
    names once converted to click.
    """
    click_cmd = typer.main.get_command(app)
    return set(click_cmd.commands.keys())


def test_budget_command_registered():
    """The budget command must exist on the typer app."""
    assert "budget" in _registered_names()


def test_all_documented_commands_registered():
    """No documented command may silently disappear from the CLI."""
    registered = _registered_names()
    missing = [c for c in DOCUMENTED_COMMANDS if c not in registered]
    assert not missing, f"Documented CLI commands not registered: {missing}"


def test_budget_help_lists_all_loader_types():
    """--type option help must list all six budget loader types."""
    budget_cmd = typer.main.get_command(app).commands["budget"]
    type_param = next(
        p for p in budget_cmd.params if "--type" in getattr(p, "opts", ())
    )
    help_text = type_param.help or ""
    for loader in ("vllm", "token", "gpu_util", "carbon", "embedding", "reranking"):
        assert loader in help_text, f"loader {loader!r} missing from --type help"
