"""In-memory MCP harness: drives a real server through the SDK client, no subprocess."""

import json
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


DOCS = {
    "d1": "budget aware reranking with cross encoders",
    "d2": "latency budgets for neural rerankers",
    "d3": "a recipe for tomato soup",
    "d4": "token costs of large language models",
}


@pytest.fixture
def local_files(mcp_server):
    root = mcp_server.root
    (root / "data").mkdir()
    (root / "data/corpus.jsonl").write_text("".join(
        json.dumps({"doc_id": k, "text": v, "title": ""}) + "\n" for k, v in DOCS.items()))
    (root / "data/queries.jsonl").write_text(
        json.dumps({"id": "q1", "text": "reranking budget"}) + "\n" + json.dumps({"id": "q2", "text": "soup"}) + "\n")
    (root / "data/qrels.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t1\nq2\td3\t1\n")
    return {"corpus_path": "data/corpus.jsonl", "queries_path": "data/queries.jsonl", "qrels_path": "data/qrels.tsv"}


@pytest.fixture
def local_dataset(mcp_server, local_files):
    return mcp_server.call("load_dataset", benchmark="local", options=local_files)["dataset_id"]
