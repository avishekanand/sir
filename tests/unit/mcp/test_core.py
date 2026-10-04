"""ragtune.mcp core: discovery tools, resources, error translation, paths, jobs."""

import sys
import time

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from ragtune.core.types import ScoredDocument
from ragtune.mcp._common import merge_limits, to_jsonable, tool_errors
from ragtune.mcp.state import JobManager, ServerState

CORE_TOOLS = {
    "server_info", "list_components", "describe_component", "get_settings", "set_setting",
    "list_default_scenarios", "job_status", "list_jobs", "cancel_job",
}


def wait_for(job, timeout=10.0):
    deadline = time.time() + timeout
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.02)
    return job


def test_core_tools_are_listed_with_schemas(mcp_server):
    tools = mcp_server.session(lambda c: c.list_tools())
    by_name = {t.name: t for t in tools.tools}
    assert CORE_TOOLS <= set(by_name)
    assert by_name["server_info"].annotations.read_only_hint is True
    assert by_name["set_setting"].annotations.read_only_hint is False
    assert by_name["describe_component"].input_schema["required"] == ["category", "name"]


def test_server_info_reports_root_and_versions(mcp_server):
    info = mcp_server.call("server_info")
    assert info["workspace_root"] == str(mcp_server.root.resolve())
    assert info["ragtune_version"]
    assert "pyterrier" in info["optional_packages"]


def test_list_components_accepts_plural_and_shows_parameters(mcp_server):
    schedulers = mcp_server.call("list_components", category="schedulers")["scheduler"]
    active = next(s for s in schedulers if s["name"] == "active-learning")
    assert {"name": "batch_size", "required": False, "type": "int", "default": 5} in active["parameters"]
    assert active["summary"] == ""  # no own docstring: must not inherit ABC's


def test_describe_component_errors_name_the_alternatives(mcp_server):
    detail = mcp_server.call("describe_component", category="reranker", name="cross-encoder")
    assert detail["class"] == "ragtune.components.rerankers.CrossEncoderReranker"
    message = mcp_server.error("describe_component", category="reranker", name="nope")
    assert "No reranker named 'nope'" in message and "cross-encoder" in message
    assert "Unknown category" in mcp_server.error("list_components", category="widgets")


def test_get_and_set_settings(mcp_server):
    from ragtune.utils.config import config

    original = config.get("retrieval.max_pool_size")
    try:
        assert mcp_server.call("get_settings", key="retrieval.max_pool_size")["value"] == original
        changed = mcp_server.call("set_setting", key="retrieval.max_pool_size", value=7)
        assert changed == {"key": "retrieval.max_pool_size", "previous": original, "value": 7}
        assert config.get("retrieval.max_pool_size") == 7
    finally:
        config.set("retrieval.max_pool_size", original)
    assert "No setting" in mcp_server.error("get_settings", key="retrieval.nope")
    assert "llm_rewrite" in mcp_server.call("get_settings", key="reformulation", prompts=True)["value"]


def test_default_scenarios(mcp_server):
    scenarios = mcp_server.call("list_default_scenarios")["scenarios"]
    assert len(scenarios) == 7 and scenarios[0]["name"] == "bm25_only"


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


def test_task_jobs_report_results_and_errors(tmp_path):
    jobs = JobManager(tmp_path)
    ok = wait_for(jobs.start_task("adds", lambda: {"sum": 3}))
    assert ok.describe()["result"] == {"sum": 3}
    failed = wait_for(jobs.start_task("fails", lambda: 1 / 0))
    assert failed.status == "failed" and "ZeroDivisionError" in failed.error
    with pytest.raises(ValueError, match="cannot be interrupted"):
        jobs.cancel(jobs.start_task("sleeps", lambda: time.sleep(0.2)).id)


def test_process_jobs_log_timeout_and_cancel(tmp_path):
    jobs = JobManager(tmp_path / "logs")
    done = wait_for(jobs.start_process("echo", [sys.executable, "-c", "print('hello')"], cwd=tmp_path))
    info = done.describe(tail_lines=5)
    assert info["status"] == "succeeded" and info["returncode"] == 0 and "hello" in info["log_tail"]

    slow = [sys.executable, "-c", "import time; time.sleep(30)"]
    assert wait_for(jobs.start_process("slow", slow, cwd=tmp_path, timeout_s=0.3)).status == "timed_out"
    running = jobs.start_process("slow", slow, cwd=tmp_path)
    assert jobs.cancel(running.id).status == "cancelled"
    with pytest.raises(KeyError, match="Unknown job_id"):
        jobs.get("job-999")
