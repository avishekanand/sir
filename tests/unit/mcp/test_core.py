"""ragtune.mcp scaffold: resources, error translation, JSON conversion, path confinement."""

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from ragtune.core.types import ScoredDocument
from ragtune.mcp._common import merge_limits, to_jsonable, tool_errors
from ragtune.mcp.state import ServerState


def test_resources_serve_docs_and_registry(mcp_server):
    (mcp_server.root / "docs").mkdir()
    (mcp_server.root / "docs" / "guide.md").write_text("# Guide")

    def read(uri):
        return mcp_server.session(lambda c: c.read_resource(uri)).contents[0].text

    assert read("ragtune://docs/guide") == "# Guide"
    assert '"reranker"' in read("ragtune://registry")
    assert "max_pool_size" in read("ragtune://config/defaults")


def test_paths_are_confined_to_the_workspace_root(tmp_path):
    state = ServerState(str(tmp_path))
    assert state.resolve("configs/a.yaml") == tmp_path.resolve() / "configs" / "a.yaml"
    for outside in ("../escape.yaml", "/etc/passwd"):
        with pytest.raises(PermissionError, match="outside the workspace root"):
            state.resolve(outside)
    with pytest.raises(FileNotFoundError):
        state.resolve("missing.yaml", must_exist=True)


def test_tool_errors_keep_the_message():
    @tool_errors
    def broken():
        raise ValueError("bad type 'x'; valid: ['y']")

    with pytest.raises(ToolError, match=r"ValueError: bad type 'x'; valid: \['y'\]"):
        broken()


def test_to_jsonable_and_merge_limits():
    value = to_jsonable({"a": np.float32(1.5), "b": np.arange(2), "c": float("inf"),
                         "d": ScoredDocument(id="x", content="y")})
    assert value["a"] == 1.5 and value["b"] == [0, 1] and value["c"] == "inf"
    assert value["d"]["id"] == "x"
    assert merge_limits({"tokens": 10, "rerank_docs": 5}, {"tokens": None, "latency_ms": 9}) == {
        "rerank_docs": 5, "latency_ms": 9.0}
