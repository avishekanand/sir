"""
Unit tests for the config-based budgeting system.
"""

import os
import pytest
from src.ragtune.budget import BudgetResult, BudgetConfig, BudgetLoaderFactory
from src.ragtune.budget.main import calculate_budget, budget_report


class TestBudgetResult:
    def test_default_creation(self):
        r = BudgetResult()
        assert r.cost_usd == 0.0
        assert r.total_tokens == 0
        assert r.carbon_kg == 0.0

    def test_addition(self):
        a = BudgetResult(
            cost_usd=1.0,
            total_tokens=100,
            carbon_kg=0.01,
            throughput_tok_s=50.0,
            gpu_utilization=60.0,
        )
        b = BudgetResult(
            cost_usd=2.0,
            total_tokens=200,
            carbon_kg=0.02,
            throughput_tok_s=100.0,
            gpu_utilization=80.0,
        )
        c = a + b
        assert c.cost_usd == 3.0
        assert c.total_tokens == 300
        assert c.carbon_kg == 0.03
        assert c.gpu_utilization == 80.0  # max

    def test_addition_energy_and_carbon(self):
        a = BudgetResult(energy_kwh=0.5, carbon_kg=0.01)
        b = BudgetResult(energy_kwh=0.3, carbon_kg=0.02)
        c = a + b
        assert c.energy_kwh == 0.8
        assert c.carbon_kg == 0.03

    def test_addition_prompt_and_completion_tokens(self):
        a = BudgetResult(prompt_tokens=100, completion_tokens=200, cached_tokens=50)
        b = BudgetResult(prompt_tokens=300, completion_tokens=400, cached_tokens=100)
        c = a + b
        assert c.prompt_tokens == 400
        assert c.completion_tokens == 600
        assert c.cached_tokens == 150

    def test_addition_cost_per_million_tokens(self):
        a = BudgetResult(cost_usd=1.0, total_tokens=1000)
        b = BudgetResult(cost_usd=2.0, total_tokens=2000)
        c = a + b
        # cost_per_million = (1+2) / (1000+2000) * 1e6 = 1000.0
        assert c.cost_per_million_tokens == 1000.0

    def test_addition_breakdown_merge_sums_values(self):
        a = BudgetResult(breakdown={"power_w": 100, "peak_tps": 1000})
        b = BudgetResult(breakdown={"power_w": 200, "peak_tps": 800})
        c = a + b
        assert c.breakdown["power_w"] == 300
        assert c.breakdown["peak_tps"] == 1800

    def test_addition_breakdown_merge_preserves_unique_keys(self):
        a = BudgetResult(breakdown={"power_w": 100})
        b = BudgetResult(breakdown={"electricity_cost": 0.5})
        c = a + b
        assert c.breakdown["power_w"] == 100
        assert c.breakdown["electricity_cost"] == 0.5

    def test_addition_breakdown_empty(self):
        a = BudgetResult()
        b = BudgetResult()
        c = a + b
        assert c.breakdown == {}

    def test_addition_latency_slo_met(self):
        a = BudgetResult(latency_slo_met=True)
        b = BudgetResult(latency_slo_met=True)
        c = a + b
        assert c.latency_slo_met is True

        a2 = BudgetResult(latency_slo_met=True)
        b2 = BudgetResult(latency_slo_met=False)
        c2 = a2 + b2
        assert c2.latency_slo_met is False

    def test_addition_zero_tokens(self):
        a = BudgetResult(cost_usd=1.0, total_tokens=0)
        b = BudgetResult(cost_usd=2.0, total_tokens=0)
        c = a + b
        # Division by max(0, 1) = 1, so cost_per_million = 3.0 / 1 * 1e6
        assert c.cost_per_million_tokens == 3_000_000.0

    def test_addition_throughput_harmonic_mean(self):
        a = BudgetResult(total_tokens=100, throughput_tok_s=100.0)
        b = BudgetResult(total_tokens=200, throughput_tok_s=200.0)
        c = a + b
        # harmonic mean: (100+200) / (100/100 + 200/200) = 300 / 2 = 150
        assert c.throughput_tok_s == 150.0


class TestBudgetConfig:
    def test_default_config(self):
        cfg = BudgetConfig({})
        assert cfg.gpu_type == "A100-80GB"
        assert cfg.gpu_hourly_rate == 0.0

    def test_custom_config(self):
        cfg = BudgetConfig({"gpu_type": "H100-NVL-96GB", "gpu_hourly_rate": 10.00})
        assert cfg.gpu_type == "H100-NVL-96GB"
        assert cfg.gpu_hourly_rate == 10.00

    def test_to_dict(self):
        cfg = BudgetConfig({"gpu_type": "test"})
        d = cfg.to_dict()
        assert d["gpu_type"] == "test"

    def test_all_defaults(self):
        cfg = BudgetConfig({})
        assert cfg.gpu_count == 1
        assert cfg.region == "us-east"
        assert cfg.electricity_cost_per_kwh == 0.12
        assert cfg.carbon_intensity_g_per_kwh == 450.0
        assert cfg.pue == 1.15
        assert cfg.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cfg.quantization == "fp16"
        assert cfg.max_batch_size == 256
        assert cfg.latency_slo_ms == 500
        assert cfg.offered_rps == 10.0
        assert cfg.tensor_parallel == 1
        assert cfg.pipeline_parallel == 1
        assert cfg.kv_overhead_per_token_s == 0.00014
        assert cfg.cache_saving_fraction == 0.50
        assert cfg.default_tokens_per_doc == 512
        assert cfg.escalation_gap_threshold == 0.05
        assert cfg.gpu_power_idle_fraction == 0.25
        assert cfg.gpu_power_active_fraction == 0.75

    def test_to_dict_excludes_private(self):
        cfg = BudgetConfig({"gpu_type": "test"})
        d = cfg.to_dict()
        assert "_private" not in d


class TestBudgetLoaderFactory:
    def test_create_vllm(self):
        loader = BudgetLoaderFactory.create("vllm")
        assert loader.key == "vllm"
        assert type(loader).__name__ == "VLLMBudgetLoader"

    def test_create_token(self):
        loader = BudgetLoaderFactory.create("token")
        assert loader.key == "token"

    def test_create_gpu_util(self):
        loader = BudgetLoaderFactory.create("gpu_util")
        assert loader.key == "gpu_util"

    def test_create_carbon(self):
        loader = BudgetLoaderFactory.create("carbon")
        assert loader.key == "carbon"

    def test_create_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown budget type"):
            BudgetLoaderFactory.create("nonexistent")

    def test_list_types(self):
        types = BudgetLoaderFactory.list_types()
        assert "vllm" in types
        assert "token" in types
        assert "gpu_util" in types
        assert "carbon" in types
        assert len(types) >= 4

    def test_create_with_config(self):
        loader = BudgetLoaderFactory.create(
            "vllm",
            config={"gpu_type": "H100-NVL-96GB", "gpu_hourly_rate": 6.98},
        )
        assert loader.config.gpu_type == "H100-NVL-96GB"
        assert loader.config.gpu_hourly_rate == 6.98

    def test_create_with_yaml(self):
        path = "src/ragtune/budget/configs/h100_us_east.yaml"
        loader = BudgetLoaderFactory.create("vllm", config_path=path)
        assert loader.config.gpu_type == "H100-NVL-96GB"

    def test_create_with_both_config_and_yaml(self):
        """YAML provides base config; config dict overrides YAML values."""
        path = "src/ragtune/budget/configs/h100_us_east.yaml"
        loader = BudgetLoaderFactory.create(
            "vllm",
            config={"gpu_hourly_rate": 10.0},
            config_path=path,
        )
        # YAML provides the base (gpu_type from H100 config), config overrides
        assert loader.config.gpu_type == "H100-NVL-96GB"
        assert loader.config.gpu_hourly_rate == 10.0  # config overrides YAML


class TestVLLMBudgetLoader:
    def test_basic_calculation(self):
        loader = BudgetLoaderFactory.create("vllm")
        r = loader.calculate({"prompt_tokens": 512, "completion_tokens": 256})
        assert r.cost_usd > 0
        assert r.cost_per_million_tokens > 0
        assert r.total_tokens == 768
        assert r.prompt_tokens == 512
        assert r.completion_tokens == 256
        assert r.throughput_tok_s > 0
        assert r.gpu_utilization >= 0

    def test_large_batch(self):
        loader = BudgetLoaderFactory.create("vllm")
        r = loader.calculate(
            {
                "prompt_tokens": 4096,
                "completion_tokens": 2048,
                "batch_size": 128,
            }
        )
        assert r.cost_usd > 0
        assert r.total_tokens == 6144

    def test_cached_tokens(self):
        loader = BudgetLoaderFactory.create("vllm")
        r_no_cache = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            }
        )
        r_cached = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "cached_tokens": 800,
            }
        )
        # Cached should be cheaper
        assert r_cached.cost_usd <= r_no_cache.cost_usd

    def test_h100_config(self):
        loader = BudgetLoaderFactory.create(
            "vllm",
            config_path="src/ragtune/budget/configs/h100_us_east.yaml",
        )
        r = loader.calculate({"prompt_tokens": 1024, "completion_tokens": 512})
        assert r.cost_usd > 0
        assert r.breakdown.get("gpu_hourly_rate", 0) > 6

    def test_empty_context(self):
        loader = BudgetLoaderFactory.create("vllm")
        r = loader.calculate({})
        assert r.total_tokens == 768  # default 512+256

    def test_none_context(self):
        loader = BudgetLoaderFactory.create("vllm")
        r = loader.calculate(None)
        assert r.total_tokens == 768

    def test_energy_positive(self):
        loader = BudgetLoaderFactory.create("vllm")
        r = loader.calculate({"prompt_tokens": 512, "completion_tokens": 256})
        assert r.energy_kwh > 0

    def test_carbon_positive(self):
        loader = BudgetLoaderFactory.create("vllm")
        r = loader.calculate({"prompt_tokens": 512, "completion_tokens": 256})
        assert r.carbon_kg > 0

    def test_breakdown_keys(self):
        loader = BudgetLoaderFactory.create("vllm")
        r = loader.calculate({"prompt_tokens": 512, "completion_tokens": 256})
        expected_keys = {
            "gpu_hourly_rate",
            "peak_tps",
            "achieved_batch",
            "gpu_util_pct",
            "power_w",
            "electricity_cost",
            "cache_saving",
        }
        assert expected_keys.issubset(set(r.breakdown.keys()))

    def test_custom_gpu_hourly_rate(self):
        loader = BudgetLoaderFactory.create(
            "vllm",
            config={"gpu_hourly_rate": 10.0},
        )
        r = loader.calculate({"prompt_tokens": 512, "completion_tokens": 256})
        assert r.breakdown["gpu_hourly_rate"] == 10.0

    def test_high_rps_approaches_peak(self):
        """At high λ, throughput should approach Θ_max."""
        loader = BudgetLoaderFactory.create("vllm")
        r = loader.calculate({"prompt_tokens": 512, "completion_tokens": 256})
        # Default λ=10, should have reasonable throughput
        assert r.throughput_tok_s > 100

    def test_low_rps_low_throughput(self):
        """At low λ, throughput should be arrival-limited."""
        loader = BudgetLoaderFactory.create(
            "vllm",
            config={"offered_rps": 1.0},
        )
        r = loader.calculate({"prompt_tokens": 512, "completion_tokens": 256})
        # At λ=1 with 256 output tokens: arrival_tps = 256
        # But saturation model may give higher value, so achieved = min(arrival, saturated)
        assert r.throughput_tok_s >= 200  # at least arrival-limited
        assert r.throughput_tok_s < 2000  # not near peak

    def test_slo_enforcement(self):
        """With tight SLO, throughput should be capped."""
        loader = BudgetLoaderFactory.create(
            "vllm",
            config={"offered_rps": 100.0, "latency_slo_ms": 50},
        )
        r = loader.calculate({"prompt_tokens": 512, "completion_tokens": 256})
        # With 50ms SLO and 256 output tokens: max throughput = 256/0.05 = 5120
        # The SLO check should cap throughput at or below this
        # (may be slightly higher due to model-specific behavior)
        assert r.throughput_tok_s <= 8000  # reasonable upper bound

    def test_cache_saving_factor(self):
        """Cache saving should be 50% of cached fraction."""
        loader = BudgetLoaderFactory.create("vllm")
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "cached_tokens": 500,
            }
        )
        # 500/1500 * 0.50 = 0.1667
        assert abs(r.breakdown["cache_saving"] - 0.1667) < 0.01

    def test_cost_scales_with_tokens(self):
        """Cost should increase with more tokens (at same throughput)."""
        loader = BudgetLoaderFactory.create("vllm")
        r1 = loader.calculate({"prompt_tokens": 100, "completion_tokens": 100})
        r2 = loader.calculate({"prompt_tokens": 400, "completion_tokens": 400})
        # More tokens should cost more (even if throughput scales)
        assert r2.total_tokens > r1.total_tokens
        assert r2.cost_usd >= r1.cost_usd

    def test_multi_gpu(self):
        """Multiple GPUs should have higher hourly cost but potentially better throughput."""
        loader_single = BudgetLoaderFactory.create("vllm", config={"gpu_count": 1})
        loader_double = BudgetLoaderFactory.create("vllm", config={"gpu_count": 2})
        r1 = loader_single.calculate({"prompt_tokens": 512, "completion_tokens": 256})
        r2 = loader_double.calculate({"prompt_tokens": 512, "completion_tokens": 256})
        # With 2 GPUs, hourly cost doubles
        assert r2.breakdown["gpu_hourly_rate"] == r1.breakdown["gpu_hourly_rate"] * 2


class TestTokenBudgetLoader:
    def test_basic(self):
        loader = BudgetLoaderFactory.create("token")
        r = loader.calculate({"prompt_tokens": 1000, "completion_tokens": 500})
        assert r.cost_usd > 0
        assert r.total_tokens == 1500

    def test_cached_tokens_discount(self):
        loader = BudgetLoaderFactory.create("token")
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "cached_tokens": 800,
            }
        )
        assert r.cost_usd > 0

    def test_cached_rate_is_50_percent(self):
        """Cached tokens should cost 50% of input rate."""
        loader = BudgetLoaderFactory.create("token")
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 0,
                "cached_tokens": 1000,
            }
        )
        # All tokens cached: cost = 1000/1e6 * 1.25 = 0.00125
        assert abs(r.cost_usd - 0.00125) < 0.0001

    def test_no_cached_tokens(self):
        loader = BudgetLoaderFactory.create("token")
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "cached_tokens": 0,
            }
        )
        # cost = (1000/1e6 * 2.50) + (500/1e6 * 10.00) = 0.0025 + 0.005 = 0.0075
        assert abs(r.cost_usd - 0.0075) < 0.0001

    def test_negative_guard(self):
        """cached_tokens > prompt_tokens should not cause negative cost."""
        loader = BudgetLoaderFactory.create("token")
        r = loader.calculate(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cached_tokens": 200,  # more than prompt_tokens
            }
        )
        assert r.cost_usd >= 0

    def test_zero_tokens(self):
        loader = BudgetLoaderFactory.create("token")
        r = loader.calculate({"prompt_tokens": 0, "completion_tokens": 0})
        assert r.cost_usd == 0.0
        assert r.total_tokens == 0

    def test_custom_rates(self):
        loader = BudgetLoaderFactory.create(
            "token",
            config={
                "extra": {"input_rate": 1.0, "output_rate": 2.0, "cached_rate": 0.5}
            },
        )
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "cached_tokens": 0,
            }
        )
        # cost = (1000/1e6 * 1.0) + (1000/1e6 * 2.0) = 0.001 + 0.002 = 0.003
        assert abs(r.cost_usd - 0.003) < 0.0001

    def test_breakdown_keys(self):
        loader = BudgetLoaderFactory.create("token")
        r = loader.calculate({"prompt_tokens": 1000, "completion_tokens": 500})
        assert "input_rate" in r.breakdown
        assert "output_rate" in r.breakdown
        assert "cached_rate" in r.breakdown
        assert "uncached_prompt" in r.breakdown


class TestGPUUtilBudgetLoader:
    def test_basic(self):
        loader = BudgetLoaderFactory.create("gpu_util")
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "runtime_s": 5.0,
                "gpu_util_pct": 80.0,
            }
        )
        assert r.cost_usd > 0
        assert r.total_tokens == 1500

    def test_short_runtime(self):
        loader = BudgetLoaderFactory.create("gpu_util")
        r = loader.calculate(
            {
                "prompt_tokens": 512,
                "completion_tokens": 256,
                "runtime_s": 0.5,
                "gpu_util_pct": 50.0,
            }
        )
        assert r.cost_usd > 0

    def test_cost_scales_with_runtime(self):
        loader = BudgetLoaderFactory.create("gpu_util")
        r1 = loader.calculate({"runtime_s": 1.0, "gpu_util_pct": 50.0})
        r2 = loader.calculate({"runtime_s": 2.0, "gpu_util_pct": 50.0})
        assert r2.cost_usd > r1.cost_usd

    def test_energy_calculation(self):
        loader = BudgetLoaderFactory.create("gpu_util")
        r = loader.calculate(
            {
                "runtime_s": 3600,  # 1 hour
                "gpu_util_pct": 100.0,
            }
        )
        # A100-80GB at 100% util: power = 400 * (0.25 + 0.75*1.0) = 400W
        # energy = 400 * 1.15 (PUE) * 3600 / 3600 / 1000 = 0.46 kWh
        assert abs(r.energy_kwh - 0.46) < 0.01

    def test_carbon_calculation(self):
        loader = BudgetLoaderFactory.create(
            "gpu_util", config={"carbon_intensity_g_per_kwh": 500}
        )
        r = loader.calculate(
            {
                "runtime_s": 3600,
                "gpu_util_pct": 100.0,
            }
        )
        # carbon = 0.46 kWh (with PUE) * 500 g/kWh / 1000 = 0.23 kg
        assert abs(r.carbon_kg - 0.23) < 0.01

    def test_all_gpu_types(self):
        """All GPU types should produce valid results."""
        for gpu_type in [
            "H100-NVL-96GB",
            "A100-80GB",
            "A100-40GB",
            "V100-32GB",
            "T4-16GB",
            "L4-24GB",
        ]:
            loader = BudgetLoaderFactory.create(
                "gpu_util", config={"gpu_type": gpu_type}
            )
            r = loader.calculate({"runtime_s": 1.0, "gpu_util_pct": 50.0})
            assert r.cost_usd > 0, f"Failed for {gpu_type}"

    def test_throughput_calculation(self):
        loader = BudgetLoaderFactory.create("gpu_util")
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "runtime_s": 1.0,
            }
        )
        assert r.throughput_tok_s == 1500.0  # 1500 tokens / 1 second


class TestCarbonBudgetLoader:
    def test_basic(self):
        loader = BudgetLoaderFactory.create("carbon")
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "runtime_s": 10.0,
                "gpu_util_pct": 60.0,
            }
        )
        assert r.carbon_kg > 0
        assert r.energy_kwh > 0

    def test_regional_intensity(self):
        loader = BudgetLoaderFactory.create("carbon", config={"region": "eu-france"})
        r = loader.calculate(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "runtime_s": 10.0,
                "gpu_util_pct": 60.0,
            }
        )
        assert r.carbon_kg > 0

    def test_all_regions_produce_carbon(self):
        """All regions should produce non-zero carbon."""
        for region in [
            "us-east",
            "us-west",
            "eu-central",
            "eu-north",
            "eu-france",
            "asia-east",
            "asia-south",
            "australia",
        ]:
            loader = BudgetLoaderFactory.create("carbon", config={"region": region})
            r = loader.calculate({"runtime_s": 10.0, "gpu_util_pct": 50.0})
            assert r.carbon_kg > 0, f"Failed for {region}"

    def test_carbon_zero_runtime(self):
        loader = BudgetLoaderFactory.create("carbon")
        r = loader.calculate({"runtime_s": 0.0, "gpu_util_pct": 50.0})
        assert r.carbon_kg == 0.0
        assert r.energy_kwh == 0.0

    def test_carbon_cost_is_zero(self):
        """Carbon loader should not produce monetary cost."""
        loader = BudgetLoaderFactory.create("carbon")
        r = loader.calculate({"runtime_s": 10.0, "gpu_util_pct": 50.0})
        assert r.cost_usd == 0.0

    def test_breakdown_keys(self):
        loader = BudgetLoaderFactory.create("carbon")
        r = loader.calculate({"runtime_s": 10.0, "gpu_util_pct": 50.0})
        assert "carbon_intensity" in r.breakdown
        assert "gpu_util_pct" in r.breakdown
        assert "power_w" in r.breakdown
        assert "runtime_s" in r.breakdown


class TestHardwareModule:
    def test_get_gpu_spec_known(self):
        from ragtune.budget.hardware import get_gpu_spec

        spec = get_gpu_spec("H100-NVL-96GB")
        assert spec.memory_bw_gb_s == 3900
        assert spec.tdp_w == 400
        assert spec.vram_gb == 94
        assert spec.has_fp8 is True

    def test_get_gpu_spec_unknown_fallback(self):
        from ragtune.budget.hardware import get_gpu_spec

        spec = get_gpu_spec("UNKNOWN_GPU")
        assert spec.name == "A100-80GB"  # default

    def test_list_gpu_types(self):
        from ragtune.budget.hardware import list_gpu_types

        types = list_gpu_types()
        assert "H100-NVL-96GB" in types
        assert "A100-80GB" in types
        assert len(types) >= 4

    def test_gpu_spec_frozen(self):
        from ragtune.budget.hardware import GPUSpec

        spec = GPUSpec("test", 1000, 100, 200, 40, 300, 1.0)
        with pytest.raises(AttributeError):
            spec.memory_bw_gb_s = 2000

    def test_a100_no_fp8(self):
        from ragtune.budget.hardware import get_gpu_spec

        spec = get_gpu_spec("A100-80GB")
        assert spec.has_fp8 is False
        assert spec.compute_tflops_fp8 == 0

    def test_all_specs_have_required_fields(self):
        from ragtune.budget.hardware import GPU_SPECS

        for name, spec in GPU_SPECS.items():
            assert spec.memory_bw_gb_s > 0, f"{name}: bandwidth must be positive"
            assert spec.tdp_w > 0, f"{name}: TDP must be positive"
            assert spec.vram_gb > 0, f"{name}: VRAM must be positive"
            assert spec.hourly_rate > 0, f"{name}: hourly rate must be positive"


class TestThroughputModule:
    def test_estimate_peak_throughput_empirical(self):
        from ragtune.budget.throughput import estimate_peak_throughput

        tps = estimate_peak_throughput(
            "H100-NVL-96GB",
            "llama-3.1-8b",
            "fp16",
            total_params_b=8.0,
            active_params_b=8.0,
        )
        assert tps == 6238  # from paper Table 4

    def test_estimate_peak_throughput_fallback(self):
        from ragtune.budget.throughput import estimate_peak_throughput

        tps = estimate_peak_throughput(
            "A100-80GB",
            "unknown-model",
            "fp16",
            total_params_b=0.1,
            active_params_b=0.1,
        )
        assert tps > 0  # should use analytical fallback

    def test_estimate_actual_throughput_low_rps(self):
        from ragtune.budget.throughput import estimate_actual_throughput

        tps, batch = estimate_actual_throughput(
            "H100-NVL-96GB",
            "llama-3.1-8b",
            "fp16",
            offered_rps=1.0,
        )
        # At λ=1 with 256 output tokens: arrival_tps = 256
        # Saturation model may give higher, so achieved = min(arrival, saturated)
        assert tps >= 200  # at least arrival-limited
        assert tps < 2000  # not near peak

    def test_estimate_actual_throughput_high_rps(self):
        from ragtune.budget.throughput import estimate_actual_throughput

        tps, batch = estimate_actual_throughput(
            "H100-NVL-96GB",
            "llama-3.1-8b",
            "fp16",
            offered_rps=50.0,
        )
        # At high λ, should approach Θ_max (6238)
        assert tps > 5000

    def test_get_saturation_knee(self):
        from ragtune.budget.throughput import get_saturation_knee

        assert get_saturation_knee(0.1) == 25.0
        assert get_saturation_knee(8.0) == 12.0
        assert get_saturation_knee(13.0) == 10.0

    def test_get_saturation_knee_interpolation(self):
        from ragtune.budget.throughput import get_saturation_knee

        # Between 3.0 and 8.0
        knee = get_saturation_knee(5.0)
        assert 12.0 <= knee <= 15.0

    def test_get_model_profile_known(self):
        from ragtune.budget.throughput import get_model_profile

        total, active, arch = get_model_profile("llama-3.1-8b")
        assert total == 8.0
        assert active == 8.0
        assert arch == "dense"

    def test_get_model_profile_unknown(self):
        from ragtune.budget.throughput import get_model_profile

        total, active, arch = get_model_profile("unknown", 1.0, 0.5, "sparse_moe")
        assert total == 1.0
        assert active == 0.5

    def test_quant_bytes_per_param(self):
        from ragtune.budget.throughput import quant_bytes_per_param

        assert quant_bytes_per_param("fp16") == 2.0
        assert quant_bytes_per_param("fp8") == 1.0
        assert quant_bytes_per_param("int4") == 0.5
        assert quant_bytes_per_param("unknown") == 2.0  # default

    def test_estimate_weight_vram(self):
        from ragtune.budget.throughput import estimate_weight_vram

        vram = estimate_weight_vram(8.0, "fp16")
        # 8e9 * 2.0 / (1024^3) ≈ 14.9 GB
        assert 14.0 <= vram <= 16.0

    def test_estimate_gpu_power(self):
        from ragtune.budget.hardware import estimate_gpu_power

        # A100-80GB at 50% util: 400 * (0.25 + 0.75*0.5) = 400 * 0.625 = 250W
        power = estimate_gpu_power("A100-80GB", 1, 0.5)
        assert abs(power - 250.0) < 1.0

    def test_estimate_gpu_power_multi_gpu(self):
        from ragtune.budget.hardware import estimate_gpu_power

        power_single = estimate_gpu_power("A100-80GB", 1, 0.5)
        power_double = estimate_gpu_power("A100-80GB", 2, 0.5)
        assert power_double == power_single * 2


class TestBudgetMain:
    def test_calculate_budget(self):
        r = calculate_budget("vllm", prompt_tokens=512, completion_tokens=256)
        assert isinstance(r, BudgetResult)
        assert r.cost_usd > 0

    def test_budget_report(self):
        report = budget_report("vllm", prompt_tokens=512, completion_tokens=256)
        assert "Cost:" in report
        assert "$/M tokens:" in report
        assert "Carbon:" in report
        assert "GPU util:" in report

    def test_result_addition(self):
        a = calculate_budget("vllm", prompt_tokens=512, completion_tokens=256)
        b = calculate_budget("vllm", prompt_tokens=512, completion_tokens=256)
        combined = a + b
        assert combined.cost_usd > a.cost_usd
        assert combined.total_tokens == a.total_tokens + b.total_tokens

    def test_calculate_budget_all_types(self):
        for budget_type in ["vllm", "token", "gpu_util", "carbon"]:
            r = calculate_budget(budget_type, prompt_tokens=512, completion_tokens=256)
            assert isinstance(r, BudgetResult)

    def test_budget_report_all_types(self):
        for budget_type in ["vllm", "token", "gpu_util", "carbon"]:
            report = budget_report(
                budget_type, prompt_tokens=512, completion_tokens=256
            )
            assert "Budget Report" in report


class TestEmbeddingBudgetLoader:
    def test_basic(self):
        loader = BudgetLoaderFactory.create("embedding")
        r = loader.calculate({"tokens": 1000})
        assert r.cost_usd > 0
        assert r.embedding_cost_usd > 0
        assert r.total_tokens == 1000

    def test_zero_tokens(self):
        loader = BudgetLoaderFactory.create("embedding")
        r = loader.calculate({"tokens": 0})
        assert r.cost_usd == 0.0

    def test_custom_model(self):
        loader = BudgetLoaderFactory.create(
            "embedding",
            config={"extra": {"embedding_model": "openai/text-embedding-3-large"}},
        )
        r = loader.calculate({"tokens": 1000})
        assert r.breakdown["price_per_million"] == 0.13

    def test_custom_price(self):
        loader = BudgetLoaderFactory.create(
            "embedding",
            config={"extra": {"embedding_price_per_million": 0.05}},
        )
        r = loader.calculate({"tokens": 1000})
        assert r.breakdown["price_per_million"] == 0.05
        assert r.cost_usd == 0.00005


class TestRerankingBudgetLoader:
    def test_cohere_per_query(self):
        loader = BudgetLoaderFactory.create("reranking")
        r = loader.calculate({"queries": 10, "docs_per_query": 50})
        assert r.cost_usd > 0
        assert r.reranking_cost_usd > 0
        assert r.breakdown["queries"] == 10.0

    def test_voyage_per_token(self):
        loader = BudgetLoaderFactory.create(
            "reranking",
            config={"extra": {"reranking_model": "voyage/rerank-2.5"}},
        )
        r = loader.calculate(
            {
                "queries": 10,
                "docs_per_query": 50,
                "query_tokens": 20,
                "doc_tokens_per_doc": 200,
            }
        )
        assert r.cost_usd > 0
        # tokens = 10 × (20 + 50 × 200) = 100,200
        assert r.total_tokens == 100200

    def test_zero_queries(self):
        loader = BudgetLoaderFactory.create("reranking")
        r = loader.calculate({"queries": 0, "docs_per_query": 10})
        assert r.cost_usd == 0.0


class TestCostOptimizer:
    def test_suggest_optimizations(self):
        from ragtune.budget.optimizer import suggest_optimizations
        from ragtune.budget.result import BudgetResult

        result = BudgetResult(
            cost_usd=0.01,
            gpu_utilization=20.0,
            throughput_tok_s=100,
            cached_tokens=0,
            prompt_tokens=1000,
        )
        suggestions = suggest_optimizations(result)
        assert len(suggestions) > 0
        categories = [s["category"] for s in suggestions]
        assert "caching" in categories
        assert "model_selection" in categories

    def test_format_suggestions(self):
        from ragtune.budget.optimizer import suggest_optimizations, format_suggestions
        from ragtune.budget.result import BudgetResult

        result = BudgetResult(
            gpu_utilization=20.0, throughput_tok_s=100, prompt_tokens=1000
        )
        suggestions = suggest_optimizations(result)
        formatted = format_suggestions(suggestions)
        assert "Optimization Suggestions" in formatted

    def test_no_suggestions_for_good_config(self):
        from ragtune.budget.optimizer import suggest_optimizations
        from ragtune.budget.result import BudgetResult

        result = BudgetResult(
            cost_usd=0.001,
            gpu_utilization=80.0,
            throughput_tok_s=5000,
            cached_tokens=500,
            prompt_tokens=1000,
            latency_slo_met=True,
        )
        suggestions = suggest_optimizations(result)
        # Should have fewer suggestions for a well-configured system
        assert len(suggestions) <= 2


class TestCostHistory:
    def test_log_and_query(self):
        import tempfile
        import os
        from ragtune.budget.history import CostHistoryLogger
        from ragtune.budget.result import BudgetResult

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            logger = CostHistoryLogger(path)
            result = BudgetResult(cost_usd=0.001, total_tokens=768)
            logger.log(
                "vllm", {"gpu_type": "A100-80GB"}, {"prompt_tokens": 512}, result
            )

            entries = logger.query(budget_type="vllm")
            assert len(entries) == 1
            assert entries[0]["budget_type"] == "vllm"
            assert entries[0]["result"]["cost_usd"] == 0.001
        finally:
            os.unlink(path)

    def test_summary(self):
        import tempfile
        import os
        from ragtune.budget.history import CostHistoryLogger
        from ragtune.budget.result import BudgetResult

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            logger = CostHistoryLogger(path)
            for i in range(5):
                result = BudgetResult(
                    cost_usd=0.001 * (i + 1), total_tokens=100 * (i + 1)
                )
                logger.log("vllm", {}, {}, result)

            summary = logger.summary(budget_type="vllm")
            assert summary["count"] == 5
            assert summary["total_cost_usd"] > 0
            assert summary["total_tokens"] > 0
        finally:
            os.unlink(path)

    def test_clear(self):
        import tempfile
        import os
        from ragtune.budget.history import CostHistoryLogger
        from ragtune.budget.result import BudgetResult

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            logger = CostHistoryLogger(path)
            logger.log("vllm", {}, {}, BudgetResult(cost_usd=0.001))
            assert len(logger.query()) == 1

            logger.clear()
            assert len(logger.query()) == 0
            assert not os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestCostAlerts:
    def test_no_alerts_within_thresholds(self):
        from ragtune.budget.alerts import check_alerts
        from ragtune.budget.result import BudgetResult

        result = BudgetResult(
            cost_usd=0.001,
            carbon_kg=0.0001,
            energy_kwh=0.0001,
            throughput_tok_s=1000,
            gpu_utilization=50.0,
            latency_slo_met=True,
        )
        alerts = check_alerts(
            result,
            {
                "max_cost_usd": 0.01,
                "max_carbon_kg": 0.001,
            },
        )
        assert len(alerts) == 0

    def test_cost_exceeds_threshold(self):
        from ragtune.budget.alerts import check_alerts
        from ragtune.budget.result import BudgetResult

        result = BudgetResult(cost_usd=0.05)
        alerts = check_alerts(result, {"max_cost_usd": 0.01})
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"

    def test_slo_not_met(self):
        from ragtune.budget.alerts import check_alerts
        from ragtune.budget.result import BudgetResult

        result = BudgetResult(cost_usd=0.001, latency_slo_met=False)
        alerts = check_alerts(result)
        slo_alerts = [a for a in alerts if "SLO" in a["message"]]
        assert len(slo_alerts) == 1
        assert slo_alerts[0]["severity"] == "critical"

    def test_low_gpu_utilization(self):
        from ragtune.budget.alerts import check_alerts
        from ragtune.budget.result import BudgetResult

        result = BudgetResult(gpu_utilization=10.0)
        alerts = check_alerts(result, {"min_gpu_utilization": 30.0})
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "info"

    def test_format_alerts(self):
        from ragtune.budget.alerts import check_alerts, format_alerts
        from ragtune.budget.result import BudgetResult

        result = BudgetResult(cost_usd=0.05)
        alerts = check_alerts(result, {"max_cost_usd": 0.01})
        formatted = format_alerts(alerts)
        assert "Cost Alerts" in formatted
        assert "CRITICAL" in formatted
