"""RAGtune MCP server: exposes the repository's capabilities as MCP tools.

Run with ``ragtune-mcp`` or ``python -m ragtune.mcp``. See docs/mcp.md.
"""

import argparse
import importlib
import importlib.metadata
import os
from typing import List, Optional

from mcp.server.mcpserver import MCPServer

from ragtune.mcp import prompts, resources
from ragtune.mcp.state import ServerState
from ragtune.mcp.tools import budget, config, data, discovery, pipeline, runners

INSTRUCTIONS = """\
RAGtune is budget-aware iterative RAG middleware: retrieve, then rerank in
batches while a budget (tokens, rerank docs, latency, ...) remains, then
assemble the final context.

Start with server_info (what can run here) and list_components (valid
'type' strings for pipeline configs). File paths are relative to the
workspace root. Long-running tools accept background=True and return a
job_id; poll job_status for the result. Errors explain how to fix the call.
"""

# Importing these populates the component registry. Each is optional: a
# missing extra (e.g. pyterrier_dr for flex indexing) must not stop the server.
REGISTRY_MODULES = ("ragtune.components", "ragtune.adapters", "ragtune.indexing")

TOOL_MODULES = (discovery, config, data, pipeline, budget, runners)


def _load_registry(state: ServerState) -> None:
    for module in REGISTRY_MODULES:
        try:
            importlib.import_module(module)
        except Exception as e:  # ImportError, or a JVM/native failure inside the import
            state.import_errors[module] = f"{type(e).__name__}: {e}"


def build_server(root: Optional[str] = None) -> MCPServer:
    """Create a server whose file access is confined to ``root``."""
    state = ServerState(root or os.environ.get("RAGTUNE_MCP_ROOT") or os.getcwd())
    _load_registry(state)
    server = MCPServer(
        "ragtune",
        title="RAGtune",
        version=importlib.metadata.version("ragtune"),
        instructions=INSTRUCTIONS,
    )
    for module in TOOL_MODULES:
        module.register(server, state)
    resources.register(server, state)
    prompts.register(server)
    return server


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="RAGtune MCP server")
    parser.add_argument("--root", default=None, help="Workspace root (default: $RAGTUNE_MCP_ROOT or cwd)")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for streamable-http")
    parser.add_argument("--port", type=int, default=8000, help="Port for streamable-http")
    args = parser.parse_args(argv)

    server = build_server(args.root)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
