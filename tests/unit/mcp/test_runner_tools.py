"""ragtune.mcp runners (scripts, tests, CLI), prompts, the full tool inventory, and a stdio smoke test."""

import sys
import time
from pathlib import Path

import anyio
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

import ragtune

# Section 4 of specs/mcp-server.md
SPEC_TOOLS = {
    "server_info", "list_components", "describe_component", "list_benchmarks", "get_settings", "set_setting",
    "list_default_scenarios", "config_template", "read_config", "write_config", "update_config",
    "validate_config", "visualize_config", "create_pipeline", "run_pipeline", "list_pipelines", "close_pipeline",
    "evaluate_run", "evaluate_pipeline", "evaluate_scenarios", "load_dataset", "list_datasets", "get_queries",
    "get_documents", "get_qrels", "export_corpus", "drop_dataset", "build_index", "build_index_from_config",
    "index_status", "search_index", "estimate_cost", "compare_costs", "validate_budget_config",
    "budget_reference", "estimate_hardware", "estimate_throughput", "cost_history", "clear_cost_history",
    "list_scripts", "run_script", "run_tests", "run_cli", "job_status", "list_jobs", "cancel_job",
}


def finish(mcp_server, started, timeout=120, tail_lines=20):
    deadline = time.time() + timeout
    while (status := mcp_server.call("job_status", job_id=started["job_id"], tail_lines=tail_lines))["status"] == "running":
        assert time.time() < deadline, status
        time.sleep(0.1)
    return status


@pytest.fixture
def workspace(mcp_server):
    root = mcp_server.root
    (root / "scripts").mkdir()
    (root / "scripts/demo.py").write_text(
        '"""Demo benchmark script."""\nimport argparse, os, sys\n'
        'p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=1)\n'
        'print("n =", p.parse_args().n, "mode =", os.environ.get("DEMO_MODE", "x")); sys.exit(0)\n')
    (root / "elsewhere.py").write_text("print('should not run')\n")
    (root / "tests").mkdir()
    (root / "tests/test_tiny.py").write_text("def test_ok():\n    assert True\n")
    return mcp_server


def test_every_spec_tool_is_registered(mcp_server):
    tools = {t.name: t for t in mcp_server.session(lambda c: c.list_tools()).tools}
    assert SPEC_TOOLS == set(tools), f"missing: {SPEC_TOOLS - set(tools)}, extra: {set(tools) - SPEC_TOOLS}"
    assert all(t.description for t in tools.values())


def test_list_and_run_scripts(workspace):
    scripts = workspace.call("list_scripts")["scripts"]
    assert scripts == [{"path": "scripts/demo.py", "summary": "Demo benchmark script.", "flags": ["--n"],
                        "env_vars": ["DEMO_MODE"]}]
    status = finish(workspace, workspace.call("run_script", script="scripts/demo.py", args=["--n", "3"],
                                              env={"DEMO_MODE": "fast"}))
    assert status["status"] == "succeeded" and "n = 3 mode = fast" in status["log_tail"]
    assert "Only .py files under" in workspace.error("run_script", script="elsewhere.py")
    assert "outside the workspace root" in workspace.error("run_script", script="../x.py")


def test_run_tests_is_confined_to_tests_dir(workspace):
    status = finish(workspace, workspace.call("run_tests", paths=["tests/test_tiny.py"]))
    assert status["status"] == "succeeded" and "1 passed" in status["log_tail"]
    assert "must be under tests/" in workspace.error("run_tests", paths=["scripts"])


def test_run_cli_executes_ragtune_commands(mcp_server):
    status = finish(mcp_server, mcp_server.call("run_cli", args=["list"]), tail_lines=200)
    assert status["status"] == "succeeded" and "cross-encoder" in status["log_tail"]


def test_prompts_are_listed_and_render(mcp_server):
    names = {p.name for p in mcp_server.session(lambda c: c.list_prompts()).prompts}
    assert names == {"build_pipeline", "benchmark_pipeline", "estimate_deployment_cost"}
    prompt = mcp_server.session(lambda c: c.get_prompt("benchmark_pipeline", {"benchmark": "sra_bench", "dataset": "toolqa"}))
    assert "sra_bench/toolqa" in prompt.messages[0].content.text


def test_stdio_server_starts_and_lists_tools(tmp_path):
    # The SDK starts the server with a minimal environment; pin it to the ragtune under test.
    src = str(Path(ragtune.__file__).resolve().parents[1])
    params = StdioServerParameters(command=sys.executable, args=["-m", "ragtune.mcp", "--root", str(tmp_path)],
                                   env={"PYTHONPATH": src})

    async def go():
        async with Client(params, read_timeout_seconds=120) as client:
            tools = (await client.list_tools()).tools
            info = await client.call_tool("server_info", {})
            return {t.name for t in tools}, info.structured_content["result"]

    names, info = anyio.run(go)
    assert SPEC_TOOLS == names
    assert info["workspace_root"] == str(tmp_path.resolve())
