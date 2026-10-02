"""--eval-ks must become a list of ints (it used to stay the string "10,50")."""

import argparse
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_tool_retrieval.py"


@pytest.fixture(scope="module")
def runner_module():
    pytest.importorskip("pyterrier")
    spec = importlib.util.spec_from_file_location("run_tool_retrieval_eval_ks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as e:  # optional indexing deps (e.g. pyterrier_dr) missing
        pytest.skip(f"runner dependencies unavailable: {e}")
    return module


def test_parse_ks(runner_module):
    assert runner_module._parse_ks("10,50") == [10, 50]
    assert runner_module._parse_ks("10") == [10]
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated integers"):
        runner_module._parse_ks("10,fifty")


class UnsetArgs(argparse.Namespace):
    """Parsed args where every flag was left out (missing attributes read as None)."""

    def __getattr__(self, name):
        return None


def test_env_var_uses_the_same_parser(runner_module, monkeypatch):
    monkeypatch.setenv("EVAL_KS", "5,20")
    assert runner_module._load_config(UnsetArgs())["eval_ks"] == [5, 20]
