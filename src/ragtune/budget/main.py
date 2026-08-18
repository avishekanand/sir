"""
Budget Main
=============
Unified budget entry point. Calculates cost, carbon, and energy
for any operation using the selected budget loader.

Usage:
    from ragtune.budget import calculate_budget

    result = calculate_budget(
        budget_type="vllm",
        config_path="configs/h100_us_east.yaml",
        prompt_tokens=512,
        completion_tokens=256,
        batch_size=32,
    )
    print(f"Cost: ${result.cost_usd}")
    print(f"Carbon: {result.carbon_kg} kg CO2")
    print(f"Energy: {result.energy_kwh} kWh")
"""

from typing import Dict, Any, Optional

from ragtune.budget.factory import BudgetLoaderFactory
from ragtune.budget.result import BudgetResult


def calculate_budget(
    budget_type: str = "vllm",
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **context,
) -> BudgetResult:
    """Unified budget calculation.

    Args:
        budget_type: "vllm", "token", "gpu_util", "carbon"
        config_path: Path to YAML config file
        config: Inline config dict (overrides YAML)
        **context: Per-request context (token counts, batch size, etc.)

    Returns:
        BudgetResult with cost, carbon, energy, tokens
    """
    loader = BudgetLoaderFactory.create(
        budget_type=budget_type,
        config=config,
        config_path=config_path,
    )
    return loader.calculate(context=context)


def budget_report(
    budget_type: str = "vllm",
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **context,
) -> str:
    """Format a human-readable budget report."""
    r = calculate_budget(budget_type, config_path, config, **context)
    lines = [
        "=" * 55,
        f"  Budget Report ({budget_type})",
        "=" * 55,
        f"  Cost:            ${r.cost_usd:.6f}",
        f"  $/M tokens:      ${r.cost_per_million_tokens:.4f}",
        f"  Energy:          {r.energy_kwh:.8f} kWh",
        f"  Carbon:          {r.carbon_kg:.8f} kg CO2",
        f"  Tokens:          {r.total_tokens} total ({r.prompt_tokens}p + {r.completion_tokens}c)",
        f"  Throughput:      {r.throughput_tok_s:.1f} tok/s",
        f"  GPU util:        {r.gpu_utilization:.1f}%",
        "-" * 55,
    ]
    if r.breakdown:
        lines.append("  Breakdown:")
        for k, v in r.breakdown.items():
            lines.append(f"    {k}: {v}")
    lines.append("=" * 55)
    return "\n".join(lines)
