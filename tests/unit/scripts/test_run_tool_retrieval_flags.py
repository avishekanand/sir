"""Benchmark runner budget flags: --limit, --only-limits, --report-keys."""

import argparse
import importlib.util
from pathlib import Path

import pytest
import yaml

from ragtune.core.types import ControllerOutput, ControllerTrace, ScoredDocument

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_tool_retrieval.py"


@pytest.fixture(scope="module")
def runner_module():
    pytest.importorskip("pyterrier")
    spec = importlib.util.spec_from_file_location("run_tool_retrieval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as e:  # optional indexing deps (e.g. pyterrier_dr) missing
        pytest.skip(f"runner dependencies unavailable: {e}")
    return module


def make_args(**overrides):
    defaults = dict(
        config=None, benchmark=None, subset=None, queries=None, top_k=None, eval_ks=None,
        index_type=None, index_dir=None, force_reindex=False,
        limit=None, only_limits=False, report_keys=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_cli_limits_merge_over_config_file(runner_module, tmp_path, monkeypatch):
    for env in ("BENCHMARK", "SCENARIOS"):
        monkeypatch.delenv(env, raising=False)
    config = tmp_path / "bench.yaml"
    config.write_text(yaml.safe_dump({"limit_overrides": {"tokens": 100, "rerank_docs": 5}}))
    cfg = runner_module._load_config(make_args(
        config=str(config),
        limit=["rerank.latency_ms=500", "tokens=none"],
        only_limits=True,
        report_keys="rerank.latency_ms, rerank.docs",
    ))
    assert cfg["limit_overrides"] == {"tokens": None, "rerank_docs": 5, "rerank.latency_ms": 500.0}
    assert cfg["only_limits"] is True
    assert cfg["report_keys"] == ["rerank.latency_ms", "rerank.docs"]


def test_invalid_keys_are_rejected_before_any_data_loads(runner_module, tmp_path):
    with pytest.raises(ValueError, match="Unknown budget component"):
        runner_module._load_config(make_args(report_keys="rerenk.latency_ms"))
    config = tmp_path / "bench.yaml"
    config.write_text(yaml.safe_dump({"limit_overrides": {"rerank.lat": 1}}))
    with pytest.raises(ValueError, match="Unknown budget dimension"):
        runner_module._load_config(make_args(config=str(config)))


class FixedController:
    def __init__(self, states):
        self.states = iter(states)

    def run(self, query):
        return ControllerOutput(
            query=query,
            documents=[ScoredDocument(id="d1", content="")],
            trace=ControllerTrace(),
            final_budget_state=next(self.states),
        )


def test_report_keys_become_per_query_average_columns(runner_module):
    pytest.importorskip("pytrec_eval")
    controller = FixedController([{"rerank.latency_ms": 10.0}, {"rerank.latency_ms": 30.0}])
    row = runner_module.run_scenario(
        "s", controller, {"q1": "a", "q2": "b"}, {"q1": {"d1": 1}, "q2": {"d1": 1}},
        eval_ks=[10], report_rerank=False, report_keys=["rerank.latency_ms", "embedding.latency_ms"],
    )
    assert row["avg_rerank.latency_ms"] == 20.0
    assert row["avg_embedding.latency_ms"] == 0.0  # never recorded counts as zero
