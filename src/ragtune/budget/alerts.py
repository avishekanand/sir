"""
Cost Alerts
===========
Threshold-based cost monitoring and alerting.

Checks BudgetResult against configurable thresholds and returns alerts.

Usage:
    from ragtune.budget.alerts import check_alerts

    alerts = check_alerts(result, thresholds={
        "max_cost_usd": 0.01,
        "max_carbon_kg": 0.001,
        "max_energy_kwh": 0.001,
        "min_throughput_tok_s": 100,
    })
"""

from typing import Dict, List, Any, Optional

from ragtune.budget.result import BudgetResult


def check_alerts(
    result: BudgetResult,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict[str, str]]:
    """Check BudgetResult against thresholds and return alerts.

    Args:
        result: The BudgetResult to check
        thresholds: Dict of threshold checks. Keys:
            - max_cost_usd: Maximum allowed cost
            - max_carbon_kg: Maximum allowed carbon
            - max_energy_kwh: Maximum allowed energy
            - min_throughput_tok_s: Minimum required throughput
            - max_gpu_utilization: Maximum GPU utilization (for overheating)
            - min_gpu_utilization: Minimum GPU utilization (for underutilization)

    Returns:
        List of alerts, each with "severity" (critical/warning/info) and "message"
    """
    if not thresholds:
        thresholds = {}

    alerts = []

    # Cost thresholds
    max_cost = thresholds.get("max_cost_usd")
    if max_cost is not None and result.cost_usd > max_cost:
        alerts.append(
            {
                "severity": "critical",
                "message": f"Cost ${result.cost_usd:.6f} exceeds threshold ${max_cost:.6f}",
            }
        )

    # Carbon thresholds
    max_carbon = thresholds.get("max_carbon_kg")
    if max_carbon is not None and result.carbon_kg > max_carbon:
        alerts.append(
            {
                "severity": "warning",
                "message": f"Carbon {result.carbon_kg:.8f} kg exceeds threshold {max_carbon:.8f} kg",
            }
        )

    # Energy thresholds
    max_energy = thresholds.get("max_energy_kwh")
    if max_energy is not None and result.energy_kwh > max_energy:
        alerts.append(
            {
                "severity": "warning",
                "message": f"Energy {result.energy_kwh:.8f} kWh exceeds threshold {max_energy:.8f} kWh",
            }
        )

    # Throughput thresholds
    min_throughput = thresholds.get("min_throughput_tok_s")
    if min_throughput is not None and result.throughput_tok_s < min_throughput:
        alerts.append(
            {
                "severity": "warning",
                "message": f"Throughput {result.throughput_tok_s:.1f} tok/s below minimum {min_throughput:.1f} tok/s",
            }
        )

    # GPU utilization thresholds
    max_gpu = thresholds.get("max_gpu_utilization")
    if max_gpu is not None and result.gpu_utilization > max_gpu:
        alerts.append(
            {
                "severity": "critical",
                "message": f"GPU utilization {result.gpu_utilization:.1f}% exceeds maximum {max_gpu:.1f}% (overheating risk)",
            }
        )

    min_gpu = thresholds.get("min_gpu_utilization")
    if min_gpu is not None and result.gpu_utilization < min_gpu:
        alerts.append(
            {
                "severity": "info",
                "message": f"GPU utilization {result.gpu_utilization:.1f}% below minimum {min_gpu:.1f}% (underutilized)",
            }
        )

    # SLO compliance
    if not result.latency_slo_met:
        alerts.append(
            {
                "severity": "critical",
                "message": "Latency SLO not met",
            }
        )

    return alerts


def format_alerts(alerts: List[Dict[str, str]]) -> str:
    """Format alerts for display."""
    if not alerts:
        return "No alerts — all thresholds within bounds."

    severity_colors = {"critical": "red", "warning": "yellow", "info": "dim"}
    lines = ["=" * 55, "  Cost Alerts", "=" * 55]

    for alert in alerts:
        color = severity_colors.get(alert["severity"], "dim")
        lines.append(
            f"  [{color}]{alert['severity'].upper()}[/{color}]: {alert['message']}"
        )

    lines.append("=" * 55)
    return "\n".join(lines)
