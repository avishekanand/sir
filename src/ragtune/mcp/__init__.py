"""Model Context Protocol server for RAGtune. Requires the 'mcp' extra: pip install -e ".[mcp]"."""

try:
    from ragtune.mcp.server import build_server, main
except ModuleNotFoundError as e:  # pragma: no cover - only without the extra installed
    if e.name and e.name.split(".")[0] == "mcp":
        raise ModuleNotFoundError(
            "The RAGtune MCP server needs the 'mcp' package: pip install -e \".[mcp]\" (Python >= 3.10)."
        ) from e
    raise

__all__ = ["build_server", "main"]
