"""In-memory MCP harness: drives a real server through the SDK client, no subprocess."""

from typing import Any

import pytest

pytest.importorskip("mcp", reason="install the 'mcp' extra to test ragtune.mcp")

import anyio  # noqa: E402
from mcp import Client  # noqa: E402

from ragtune.mcp import build_server  # noqa: E402


class MCPHarness:
    def __init__(self, root):
        self.root = root
        self.server = build_server(str(root))

    def _call(self, tool: str, args: dict):
        async def go():
            async with Client(self.server) as client:
                return await client.call_tool(tool, args)

        return anyio.run(go)

    def call(self, tool: str, **args: Any) -> Any:
        """Call a tool that must succeed; returns its structured result."""
        result = self._call(tool, args)
        assert not result.is_error, result.content[0].text
        return result.structured_content["result"]

    def error(self, tool: str, **args: Any) -> str:
        """Call a tool that must fail; returns the error text the agent sees."""
        result = self._call(tool, args)
        assert result.is_error, f"{tool} unexpectedly succeeded: {result.structured_content}"
        return result.content[0].text

    def session(self, fn):
        """Run fn(client) inside one client session (for list/read calls)."""
        async def go():
            async with Client(self.server) as client:
                return await fn(client)

        return anyio.run(go)


@pytest.fixture
def mcp_server(tmp_path):
    return MCPHarness(tmp_path)
