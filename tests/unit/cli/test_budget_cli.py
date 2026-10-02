"""`ragtune budget` flags must reach the loader, and CLI defaults must not override --config."""

import re

import pytest
import yaml
from typer.testing import CliRunner

from ragtune.budget import BudgetResult, calculate_budget
from ragtune.cli.main import BUDGET_CLI_DEFAULTS, app

runner = CliRunner()
WIDE = {"COLUMNS": "1000"}  # keep the --verbose panel on one line


def cli_result(*flags) -> BudgetResult:
    result = runner.invoke(app, ["budget", "--verbose", *flags], env=WIDE)
    assert result.exit_code == 0, result.output
    raw = re.search(r"(BudgetResult\(.*?\))\s*│", result.output, re.S).group(1)
    return eval(" ".join(raw.split()), {"BudgetResult": BudgetResult})


@pytest.fixture
def h100_yaml(tmp_path):
    path = tmp_path / "h100.yaml"
    path.write_text(yaml.safe_dump({"gpu_type": "H100-NVL-96GB", "gpu_count": 2, "offered_rps": 25.0,
                                    "latency_slo_ms": 300, "extra": {"reranking_model": "voyage/rerank-2.5"}}))
    return path


def test_reranking_queries_and_docs_reach_the_loader():
    got = cli_result("--type", "reranking", "--reranking-model", "voyage/rerank-2.5", "--queries", "10", "--docs", "50")
    want = calculate_budget("reranking", config={**BUDGET_CLI_DEFAULTS, "extra": {"reranking_model": "voyage/rerank-2.5"}},
                            queries=10, docs_per_query=50)
    assert got.cost_usd == want.cost_usd and got.total_tokens == 10 * (20 + 50 * 200)


def test_yaml_values_are_not_overwritten_by_cli_defaults(h100_yaml):
    got = cli_result("--config", str(h100_yaml))
    assert got.breakdown["gpu_hourly_rate"] == 2 * 4.50  # both GPUs from the YAML
    assert got.cost_usd == calculate_budget("vllm", config_path=str(h100_yaml)).cost_usd


def test_explicit_flags_still_override_yaml(h100_yaml):
    got = cli_result("--config", str(h100_yaml), "--gpu-count", "4", "--rps", "5")
    want = calculate_budget("vllm", config_path=str(h100_yaml), config={"gpu_count": 4, "offered_rps": 5.0})
    assert got.cost_usd == want.cost_usd and got.breakdown["gpu_hourly_rate"] == 4 * 4.50


def test_cli_extra_merges_with_yaml_extra(h100_yaml):
    got = cli_result("--config", str(h100_yaml), "--type", "reranking", "--embedding-model", "cohere/embed-v4")
    assert got.cost_usd == calculate_budget("reranking", config_path=str(h100_yaml)).cost_usd  # YAML reranker kept


def test_defaults_without_config_are_unchanged():
    assert cli_result().cost_usd == calculate_budget("vllm", config=BUDGET_CLI_DEFAULTS).cost_usd


def test_missing_config_file_is_an_error(tmp_path):
    result = runner.invoke(app, ["budget", "--config", str(tmp_path / "nope.yaml")], env=WIDE)
    assert result.exit_code == 1 and "not found" in result.output
