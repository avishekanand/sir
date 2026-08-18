"""
GPU Utilization Budget Loader
==============================
Cost estimation based purely on GPU utilization and runtime.

Formula:
    cost = GPU_hourly_rate × (runtime_seconds / 3600)

Useful for self-hosted deployments where you know GPU runtime
but don't need the full vLLM concurrency model.
"""

from typing import Dict, Any, Optional

from ragtune.budget.base import BaseBudgetLoader, BudgetConfig
from ragtune.budget.factory import BudgetLoaderFactory
from ragtune.budget.hardware import (
    get_gpu_spec,
    estimate_gpu_power,
    estimate_energy_kwh,
    estimate_carbon_kg,
)
from ragtune.budget.result import BudgetResult


@BudgetLoaderFactory.register("gpu_util")
class GPUUtilBudgetLoader(BaseBudgetLoader):
    """Budget based on GPU runtime and utilization.

    Simple: cost = GPU_time × hourly_rate
    All parameters flow from BudgetConfig.
    """

    def calculate(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetResult:
        ctx = context or {}
        runtime_s = ctx.get("runtime_s", 1.0)
        prompt_tokens = ctx.get("prompt_tokens", 512)
        completion_tokens = ctx.get("completion_tokens", 256)
        gpu_util_pct = ctx.get("gpu_util_pct", 50.0)

        cfg = self.config
        hw = get_gpu_spec(cfg.gpu_type)
        total_hourly = hw.hourly_rate * cfg.gpu_count

        # Cost: just GPU time × rate
        cost = total_hourly * (runtime_s / 3600)

        # Energy (with PUE)
        power_w = estimate_gpu_power(
            cfg.gpu_type,
            cfg.gpu_count,
            gpu_util_pct / 100,
            cfg.gpu_power_idle_fraction,
            cfg.gpu_power_active_fraction,
        )
        energy_kwh = estimate_energy_kwh(power_w, runtime_s, cfg.pue)

        # Carbon
        carbon_kg = estimate_carbon_kg(energy_kwh, cfg.carbon_intensity_g_per_kwh)

        total_tokens = prompt_tokens + completion_tokens

        return BudgetResult(
            cost_usd=round(cost, 6),
            cost_per_million_tokens=round(cost / max(total_tokens, 1) * 1_000_000, 4),
            energy_kwh=round(energy_kwh, 8),
            carbon_kg=round(carbon_kg, 8),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            throughput_tok_s=round(total_tokens / runtime_s, 1) if runtime_s > 0 else 0,
            gpu_utilization=gpu_util_pct,
            breakdown={
                "hourly_rate": total_hourly,
                "runtime_s": runtime_s,
                "gpu_util_pct": gpu_util_pct,
                "power_w": round(power_w, 1),
                "pue": cfg.pue,
            },
        )
