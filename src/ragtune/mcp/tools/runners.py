"""Runner tools: repo scripts, the test suite and the CLI as background subprocess jobs."""

import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

from ragtune.mcp._common import READ_ONLY, WRITES, add_tools
from ragtune.mcp.state import ServerState

SCRIPT_DIRS = ("scripts", "examples")
_FLAG = re.compile(r"add_argument\(\s*[\"'](--?[\w-]+)")
_ENV = re.compile(r"os\.environ\.get\(\s*[\"'](\w+)")
# Unbuffered output keeps job logs live; a fixed width keeps rich tables readable in logs.
_JOB_ENV = {"PYTHONUNBUFFERED": "1", "COLUMNS": "120"}


def _describe_script(path: Path, root: Path) -> Dict[str, Any]:
    source = path.read_text(errors="replace")
    try:
        doc = ast.get_docstring(ast.parse(source)) or ""
    except SyntaxError:
        doc = ""
    return {"path": str(path.relative_to(root)), "summary": doc.strip().split("\n")[0] if doc else "",
            "flags": sorted(set(_FLAG.findall(source))), "env_vars": sorted(set(_ENV.findall(source)))}


def register(mcp: MCPServer, state: ServerState) -> None:
    def within(path: Path, *dirs: str) -> bool:
        return any(path == state.root / d or (state.root / d) in path.parents for d in dirs)

    def start(description: str, argv: List[str], env: Optional[Dict[str, str]], timeout_s: float) -> Dict[str, Any]:
        job = state.jobs.start_process(description, argv, cwd=state.root,
                                       env={**_JOB_ENV, **(env or {})}, timeout_s=timeout_s)
        return {**job.describe(), "next": "poll job_status(job_id, tail_lines=...)"}

    def list_scripts() -> Dict[str, Any]:
        """Runnable files under scripts/ and examples/: benchmarks, experiment grid, indexing,
        result summaries, demos. Shows each script's summary, CLI flags and env vars."""
        found = [p for d in SCRIPT_DIRS for p in sorted((state.root / d).rglob("*.py"))
                 if "__pycache__" not in p.parts]
        return {"scripts": [_describe_script(p, state.root) for p in found]}

    def run_script(
        script: str, args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None, timeout_s: float = 3600
    ) -> Dict[str, Any]:
        """Run a script from scripts/ or examples/ with this server's Python, as a background job.

        Example: run_script("scripts/run_tool_retrieval.py", ["--benchmark", "sra", "--queries", "20"]).
        Many benchmark scripts are configured through env vars (see list_scripts).
        """
        path = state.resolve(script, must_exist=True)
        if path.suffix != ".py" or not within(path, *SCRIPT_DIRS):
            raise PermissionError(f"Only .py files under {list(SCRIPT_DIRS)} can be run (got {script!r})")
        return start(f"script {state.relative(path)}", [sys.executable, str(path), *(args or [])], env, timeout_s)

    def run_tests(
        paths: Optional[List[str]] = None,
        keyword: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        timeout_s: float = 1800,
    ) -> Dict[str, Any]:
        """Run pytest on paths under tests/ (default: tests/unit) as a background job.

        keyword maps to pytest -k. The log tail in job_status holds the summary line.
        """
        resolved = [state.resolve(p, must_exist=True) for p in (paths or ["tests/unit"])]
        outside = [str(p) for p in resolved if not within(p, "tests")]
        if outside:
            raise PermissionError(f"Test paths must be under tests/ (got {outside})")
        argv = [sys.executable, "-m", "pytest", *map(str, resolved), "-q"]
        if keyword:
            argv += ["-k", keyword]
        return start(f"pytest {' '.join(paths or ['tests/unit'])}", argv + list(extra_args or []), None, timeout_s)

    def run_cli(args: List[str], timeout_s: float = 600) -> Dict[str, Any]:
        """Run any `ragtune` CLI command verbatim as a background job, e.g. ["list"] or
        ["run", "cfg.yaml", "-q", "query", "--verbose"]. stdin is closed, so interactive
        commands (init --wizard, visualize --edit) fail fast; use the config tools instead."""
        return start(f"ragtune {' '.join(args)}", [sys.executable, "-m", "ragtune.cli.main", *args], None, timeout_s)

    add_tools(mcp, READ_ONLY, list_scripts)
    add_tools(mcp, WRITES, run_script, run_tests, run_cli)
