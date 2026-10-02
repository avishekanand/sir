"""Read-only MCP resources: repository docs, specs and built-in configuration files."""

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from ragtune.mcp.state import ServerState
from ragtune.registry import registry

PACKAGE_DIR = Path(__file__).resolve().parent.parent  # src/ragtune


def _markdown(directory: Path, name: str) -> str:
    path = directory / f"{name}.md"
    if not path.is_file():
        available = sorted(p.stem for p in directory.glob("*.md"))
        raise ValueError(f"No document {name!r} in {directory.name}/. Available: {available}")
    return path.read_text()


def register(mcp: MCPServer, state: ServerState) -> None:
    @mcp.resource("ragtune://docs/{name}", mime_type="text/markdown")
    def docs(name: str) -> str:
        """A document from docs/ by file stem (e.g. 'budget', 'cli'); 'README' is the repo README."""
        if name == "README":
            return (state.root / "README.md").read_text()
        return _markdown(state.root / "docs", name)

    @mcp.resource("ragtune://specs/{name}", mime_type="text/markdown")
    def specs(name: str) -> str:
        """A specification from specs/ by file stem (e.g. 'specs-v0-2')."""
        return _markdown(state.root / "specs", name)

    @mcp.resource("ragtune://config/defaults", mime_type="application/yaml")
    def runtime_defaults() -> str:
        """Runtime defaults read by the controller and components (config/defaults.yaml)."""
        return (PACKAGE_DIR / "config" / "defaults.yaml").read_text()

    @mcp.resource("ragtune://config/prompts", mime_type="application/yaml")
    def prompt_templates() -> str:
        """LLM prompt templates used by rerankers and reformulators (config/prompts.yaml)."""
        return (PACKAGE_DIR / "config" / "prompts.yaml").read_text()

    @mcp.resource("ragtune://budget/default-config", mime_type="application/yaml")
    def budget_default_config() -> str:
        """Default BudgetConfig for the cost estimators, with source citations."""
        return (PACKAGE_DIR / "budget" / "configs" / "default.yaml").read_text()

    @mcp.resource("ragtune://registry", mime_type="application/json")
    def registry_listing() -> str:
        """Registered component names per category, as used in pipeline configs."""
        return json.dumps({cat: sorted(items) for cat, items in registry.list_all().items()}, indent=2)
