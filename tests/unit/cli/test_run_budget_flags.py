"""`ragtune run` budget flags: --limit with scoped keys, --only-limits, --breakdown."""

import yaml
import pytest
from typer.testing import CliRunner

from ragtune.cli.main import app

runner = CliRunner()
WIDE = {"COLUMNS": "300"}  # keep rich from wrapping the panels asserted on below


@pytest.fixture
def config_path(tmp_path):
    docs = [{"id": f"d{i}", "content": f"fox doc {i}", "score": 1.0 - i * 0.05} for i in range(6)]
    config = {"pipeline": {
        "name": "budget-flags",
        "components": {
            "retriever": {"type": "in-memory", "params": {"documents": docs}},
            "reranker": {"type": "simulated"},
            "scheduler": {"type": "active-learning", "params": {"batch_size": 2}},
            "estimator": {"type": "baseline"},
            "assembler": {"type": "greedy", "params": {"max_docs": 3}},
        },
        # rerank_docs: 0 would block all reranking unless --only-limits drops it.
        "budget": {"limits": {"rerank_docs": 0, "latency_ms": 60000}},
    }}
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_only_limits_with_scoped_key_and_breakdown(config_path):
    result = runner.invoke(
        app, ["run", str(config_path), "-q", "fox", "--only-limits", "-l", "rerank.docs=3", "--breakdown"], env=WIDE
    )
    assert result.exit_code == 0, result.output
    assert "Per-Component Usage" in result.output
    assert "'rerank.docs': 3.0" in result.output  # final budget state panel
    assert "3 / 3" in result.output  # breakdown shows used / limit


def test_limit_none_removes_config_limit(config_path):
    result = runner.invoke(app, ["run", str(config_path), "-q", "fox", "-l", "rerank_docs=none"], env=WIDE)
    assert result.exit_code == 0, result.output
    assert "'rerank_docs': 6.0" in result.output


@pytest.mark.parametrize("bad, message", [
    ("tokens=abc", "Invalid limit"),
    ("rerenk.docs=1", "Unknown budget component"),
])
def test_bad_limits_fail_loudly(config_path, bad, message):
    result = runner.invoke(app, ["run", str(config_path), "-q", "fox", "-l", bad], env=WIDE)
    assert result.exit_code == 1
    assert message in result.output
