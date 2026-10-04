"""ragtune.mcp budget tools: parity with ragtune.budget, sweeps, reference data, history."""

import shutil
from pathlib import Path

import pytest

from ragtune.budget import calculate_budget, hardware, throughput

H100_YAML = Path(__file__).resolve().parents[3] / "src/ragtune/budget/configs/h100_us_east.yaml"


def test_estimate_cost_matches_calculate_budget(mcp_server):
    shutil.copy(H100_YAML, mcp_server.root / "h100.yaml")
    out = mcp_server.call("estimate_cost", config_path="h100.yaml", context={"prompt_tokens": 512})
    expected = calculate_budget("vllm", config_path=str(H100_YAML), prompt_tokens=512)
    assert out["result"]["cost_usd"] == expected.cost_usd
    assert out["result"]["breakdown"]["gpu_hourly_rate"] == 13.96  # both GPUs from the YAML
    assert "Budget Report (vllm)" in out["report"]


def test_reranking_context_is_honored(mcp_server):
    out = mcp_server.call("estimate_cost", budget_type="reranking",
                          config={"extra": {"reranking_model": "cohere/rerank-v4-pro"}},
                          context={"queries": 10, "docs_per_query": 50})
    assert out["result"]["reranking_cost_usd"] == pytest.approx(0.025)  # 10 queries x $0.0025


def test_suggestions_alerts_and_history(mcp_server):
    out = mcp_server.call("estimate_cost", suggest=True, thresholds={"max_cost_usd": 0.0}, log_to="logs/costs.jsonl")
    assert any(s["category"] == "quantization" for s in out["suggestions"])
    assert out["alerts"][0]["severity"] == "critical" and out["logged_to"] == "logs/costs.jsonl"
    mcp_server.call("estimate_cost", budget_type="token", log_to="logs/costs.jsonl")

    assert len(mcp_server.call("cost_history", path="logs/costs.jsonl")["entries"]) == 2
    assert mcp_server.call("cost_history", path="logs/costs.jsonl", budget_type="token", summary=True)["summary"]["count"] == 1
    mcp_server.call("clear_cost_history", path="logs/costs.jsonl")
    assert not (mcp_server.root / "logs/costs.jsonl").exists()


def test_compare_costs_sorts_variants(mcp_server):
    rows = mcp_server.call("compare_costs", context={"runtime_s": 3600}, budget_type="gpu_util", variants=[
        {"label": "h100", "config": {"gpu_type": "H100-NVL-96GB"}},
        {"label": "t4", "config": {"gpu_type": "T4-16GB"}},
    ])["rows"]
    assert [r["label"] for r in rows] == ["t4", "h100"] and rows[0]["cost_usd"] == pytest.approx(0.80)
    assert "unknown keys" in mcp_server.error("compare_costs", variants=[{"gpu": "x"}])


def test_validate_budget_config_flags_ignored_keys(mcp_server):
    report = mcp_server.call("validate_budget_config", config={"pue": 0.5, "embedding_model": "x"})
    assert report["valid"] is False and "pue must be >= 1.0" in report["errors"][0]
    assert report["ignored_keys"] == ["embedding_model"]  # belongs under "extra"


def test_reference_tables(mcp_server):
    loaders = mcp_server.call("budget_reference", table="loaders")["loaders"]
    assert set(loaders) == {"vllm", "token", "gpu_util", "carbon", "embedding", "reranking"}
    assert "docs_per_query" in loaders["reranking"]["context"]
    everything = mcp_server.call("budget_reference")
    assert everything["calibrated_theta_max"]["H100-NVL-96GB|llama-3.1-8b|fp16"] == 6238
    assert "Unknown table" in mcp_server.error("budget_reference", table="nope")


def test_hardware_and_throughput_match_the_library(mcp_server):
    hw = mcp_server.call("estimate_hardware", gpu_type="A100-80GB", utilization=1.0, runtime_s=3600, region="eu-france")
    assert hw["gpu_power_w"] == hardware.estimate_gpu_power("A100-80GB", 1, 1.0)
    assert hw["carbon_intensity_g_per_kwh"] == 45
    assert "Unknown region" in mcp_server.error("estimate_hardware", region="mars")

    tp = mcp_server.call("estimate_throughput", gpu_type="H100-NVL-96GB", model_name="llama-3.1-8b", offered_rps=25)
    assert tp["peak_source"] == "empirical" and tp["peak_tps"] == 6238
    expected, _ = throughput.estimate_actual_throughput("H100-NVL-96GB", "llama-3.1-8b", "fp16", 25, 500, 256,
                                                       8.0, 8.0, "dense")
    assert tp["achieved_tps"] == expected and tp["fits_in_vram"] is True
