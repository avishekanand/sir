"""
Carbon Budget Loader
=====================
Estimates carbon footprint of LLM inference based on:
- Energy consumption (kWh) with PUE multiplier
- Carbon intensity of the regional grid (g CO2/kWh)

Uses data from IPCC Tier 1 methodology, Ember 2025, EPA eGRID.

Formula (IPCC Tier 1):
    energy_kwh = power_w × time_s × PUE / 3600 / 1000
    carbon_kg = energy_kwh × carbon_intensity / 1000

Where:
    power_w = GPU power draw (watts)
    PUE = Power Usage Effectiveness (default 1.15), applied inside
          estimate_energy_kwh()
    carbon_intensity = g CO2e per kWh (grid-specific)
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

# Regional carbon intensity (g CO2e/kWh) — 2024 data
# Sources: Ember Global Electricity Review 2025, Our World in Data,
#          EPA eGRID 2023, IEA Electricity 2025, Google Cloud Sustainability
REGIONAL_INTENSITY = {
    "us-east": 350,  # EPA eGRID 2023: US national avg ≈ 350
    "us-west": 200,  # Oregon=79, California=195, weighted ≈ 200
    "eu-central": 280,  # Germany=330, Netherlands=251, weighted ≈ 280
    "eu-north": 50,  # Nordic weighted avg (Norway=28, Sweden=35, Finland=57)
    "eu-france": 45,  # France grid: 41-52 g CO2/kWh (nuclear-heavy)
    "asia-east": 500,  # China=526, Japan=477, Korea=417, weighted ≈ 500
    "asia-south": 700,  # India dominates: 670-705
    "australia": 525,  # Australia: 525
    "global-average": 450,  # Ember 2025: ~450 gCO2e/kWh
}


@BudgetLoaderFactory.register("carbon")
class CarbonBudgetLoader(BaseBudgetLoader):
    """Carbon footprint estimation for LLM inference.

    Formula (IPCC Tier 1):
        carbon_kg = energy_kwh × PUE × carbon_intensity / 1000

    All parameters flow from BudgetConfig — no hardcoded values.
    Source: IPCC GHG Protocol, Ember 2025, Google Cloud Carbon Footprint.
    """

    def calculate(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetResult:
        ctx = context or {}
        prompt_tokens = ctx.get("prompt_tokens", 512)
        completion_tokens = ctx.get("completion_tokens", 256)
        runtime_s = ctx.get("runtime_s", 1.0)
        gpu_util_pct = ctx.get("gpu_util_pct", 50.0)

        cfg = self.config

        # Carbon intensity from config or region lookup
        intensity = cfg.carbon_intensity_g_per_kwh
        if not cfg._carbon_intensity_set and cfg.region:
            intensity = REGIONAL_INTENSITY.get(
                cfg.region, REGIONAL_INTENSITY["global-average"]
            )

        # Energy with PUE (IPCC Tier 1 methodology)
        power_w = estimate_gpu_power(
            cfg.gpu_type,
            cfg.gpu_count,
            gpu_util_pct / 100,
            cfg.gpu_power_idle_fraction,
            cfg.gpu_power_active_fraction,
        )
        energy_kwh = estimate_energy_kwh(power_w, runtime_s, cfg.pue)

        # Carbon: energy_kwh (already includes PUE) × intensity / 1000
        carbon_kg = estimate_carbon_kg(energy_kwh, intensity)

        total_tokens = prompt_tokens + completion_tokens
        cost_usd = 0.0  # Carbon-only, no monetary cost

        return BudgetResult(
            cost_usd=cost_usd,
            energy_kwh=round(energy_kwh, 8),
            carbon_kg=round(carbon_kg, 8),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            breakdown={
                "carbon_intensity": intensity,
                "gpu_util_pct": gpu_util_pct,
                "power_w": round(power_w, 1),
                "runtime_s": runtime_s,
                "pue": cfg.pue,
            },
        )
