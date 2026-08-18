"""
vLLM Budget Loader
====================
Concurrency-aware cost estimation based on the formula from
arXiv 2606.11690 (Patil, June 2026):

    C_eff = (P_GPU × 1e6) / (3600 × Θ_achieved(λ, L))

Delegates throughput estimation to ragtune.budget.throughput and
hardware specs to ragtune.budget.hardware.
"""

from typing import Any, Dict, Optional

from ragtune.budget.base import BaseBudgetLoader, BudgetConfig
from ragtune.budget.factory import BudgetLoaderFactory
from ragtune.budget.hardware import (
    get_gpu_spec,
    estimate_gpu_power,
    estimate_energy_kwh,
    estimate_carbon_kg,
)
from ragtune.budget.result import BudgetResult
from ragtune.budget.throughput import (
    estimate_actual_throughput,
    estimate_peak_throughput,
    get_model_profile,
)


@BudgetLoaderFactory.register("vllm")
class VLLMBudgetLoader(BaseBudgetLoader):
    """Concurrency-aware budget loader based on arXiv 2606.11690.

    Calculates cost per million tokens under offered load, accounting
    for GPU utilization, model architecture, quantization, and latency SLO.

    All parameters flow from BudgetConfig — no hardcoded values.
    """

    def calculate(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetResult:
        ctx = context or {}

        prompt_tokens = ctx.get("prompt_tokens", 512)
        completion_tokens = ctx.get("completion_tokens", 256)
        cached_tokens = ctx.get("cached_tokens", 0)

        cfg = self.config
        hw = get_gpu_spec(cfg.gpu_type)

        # ── Throughput estimation ──
        total_b, active_b, arch = get_model_profile(
            cfg.model_name,
            cfg.total_params_b,
            cfg.active_params_b,
            cfg.model_architecture,
        )
        actual_tps, achieved_batch = estimate_actual_throughput(
            gpu_type=cfg.gpu_type,
            model_name=cfg.model_name,
            quantization=cfg.quantization,
            offered_rps=cfg.offered_rps,
            latency_slo_ms=cfg.latency_slo_ms,
            output_tokens=completion_tokens,
            total_params_b=total_b,
            active_params_b=active_b,
            architecture=arch,
            tensor_parallel=cfg.tensor_parallel,
            max_batch_size=cfg.max_batch_size,
            kv_overhead_per_token_s=cfg.kv_overhead_per_token_s,
            lam_sat_fallback=cfg.lam_sat_fallback,
            peak_utilization_threshold=cfg.peak_utilization_threshold,
        )
        peak_tps = estimate_peak_throughput(
            cfg.gpu_type,
            cfg.model_name,
            cfg.quantization,
            total_b,
            active_b,
            arch,
            cfg.tensor_parallel,
            cfg.max_batch_size,
            cfg.kv_overhead_per_token_s,
        )
        gpu_util = actual_tps / peak_tps if peak_tps > 0 else 0.0

        # ── Cost ──
        gpu_hourly = cfg.gpu_hourly_rate if cfg.gpu_hourly_rate > 0 else hw.hourly_rate
        total_gpu_hourly = gpu_hourly * cfg.gpu_count
        cost_per_million = (
            (total_gpu_hourly * 1_000_000 / (3600 * actual_tps))
            if actual_tps > 0
            else 0.0
        )
        total_tokens = prompt_tokens + completion_tokens
        request_cost = cost_per_million * total_tokens / 1_000_000

        # ── Energy (with PUE) ──
        power_w = estimate_gpu_power(
            cfg.gpu_type,
            cfg.gpu_count,
            gpu_util,
            cfg.gpu_power_idle_fraction,
            cfg.gpu_power_active_fraction,
        )
        request_time_s = total_tokens / max(actual_tps, 1)
        energy_kwh = estimate_energy_kwh(power_w, request_time_s, cfg.pue)

        # ── Carbon ──
        carbon_kg = estimate_carbon_kg(energy_kwh, cfg.carbon_intensity_g_per_kwh)

        # ── Electricity cost ──
        electricity_cost = energy_kwh * cfg.electricity_cost_per_kwh

        # ── Caching savings ──
        # Two models:
        # 1. Explicit cached_tokens (per-request): from context
        # 2. Cache hit rate (statistical): from config, applied to prompt tokens
        effective_cached = cached_tokens
        if cfg.cache_hit_rate > 0 and effective_cached == 0:
            # Apply cache hit rate to prompt tokens (statistical model)
            effective_cached = int(prompt_tokens * cfg.cache_hit_rate)
        cache_saving = (
            effective_cached / max(total_tokens, 1)
        ) * cfg.cache_saving_fraction

        # ── SLO compliance ──
        slo_met = True
        if cfg.latency_slo_ms and cfg.latency_slo_ms > 0 and actual_tps > 0:
            est_latency_ms = (completion_tokens / actual_tps) * 1000
            slo_met = est_latency_ms <= cfg.latency_slo_ms

        # ── Per-component cost breakdown ──
        # In vLLM, the dominant cost is GPU inference (generation).
        # Embedding cost is negligible for self-hosted models.
        # Reranking cost is separate (use reranking loader for API-based).
        generation_cost = request_cost  # GPU inference IS the generation cost

        return BudgetResult(
            cost_usd=round(request_cost * (1 - cache_saving), 6),
            cost_per_million_tokens=round(cost_per_million * (1 - cache_saving), 4),
            generation_cost_usd=round(generation_cost, 6),
            energy_kwh=round(energy_kwh, 8),
            carbon_kg=round(carbon_kg, 8),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            throughput_tok_s=round(actual_tps, 1),
            gpu_utilization=round(gpu_util * 100, 1),
            latency_slo_met=slo_met,
            breakdown={
                "gpu_hourly_rate": total_gpu_hourly,
                "peak_tps": round(peak_tps, 1),
                "achieved_batch": round(achieved_batch, 1),
                "gpu_util_pct": round(gpu_util * 100, 1),
                "power_w": round(power_w, 1),
                "electricity_cost": round(electricity_cost, 8),
                "cache_saving": round(cache_saving, 4),
                "pue": cfg.pue,
            },
        )
