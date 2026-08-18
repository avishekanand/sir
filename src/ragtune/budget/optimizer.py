"""
Cost Optimization Suggestions
==============================
Analyzes BudgetResult and provides actionable optimization recommendations.

Based on industry research:
- Generation dominates cost (60-80%) — optimize model selection first
- Reranking is cheap but valuable — $0.002/query for significant gains
- Cache hits reduce costs by 35% (industry average)
- Not all queries need retrieval — Flare-Aug shows 30-50% savings

Source: aicostcheck.com, agentcalc.com, Flare-Aug (Su et al., 2025)
"""

from typing import Dict, List, Any, Optional

from ragtune.budget.result import BudgetResult


def suggest_optimizations(
    result: BudgetResult,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Analyze a BudgetResult and suggest cost optimizations.

    Args:
        result: The BudgetResult to analyze
        config: Optional config dict for context

    Returns:
        List of optimization suggestions, each with:
        - "category": Area of optimization
        - "suggestion": What to do
        - "estimated_savings": Estimated cost reduction (if quantifiable)
        - "priority": high/medium/low
    """
    suggestions = []
    cfg = config or {}

    # 1. Cache hit rate optimization
    if result.cached_tokens == 0 and result.prompt_tokens > 100:
        suggestions.append(
            {
                "category": "caching",
                "suggestion": "Enable semantic caching to reduce repeated embedding/generation costs. Industry average: 35% cache hit rate reduces costs by ~35%.",
                "estimated_savings": f"${result.cost_usd * 0.35:.6f}",
                "priority": "high",
            }
        )

    # 2. Model selection optimization
    if result.gpu_utilization < 30:
        suggestions.append(
            {
                "category": "model_selection",
                "suggestion": f"GPU utilization is low ({result.gpu_utilization:.0f}%). Consider using a smaller model or batching more requests.",
                "estimated_savings": "20-50% cost reduction by right-sizing model",
                "priority": "high",
            }
        )

    # 3. Batch size optimization
    if result.throughput_tok_s < 1000:
        suggestions.append(
            {
                "category": "batching",
                "suggestion": "Throughput is low. Increase batch size or use continuous batching to improve GPU utilization.",
                "estimated_savings": "2-5x throughput improvement",
                "priority": "medium",
            }
        )

    # 4. Quantization optimization
    if cfg.get("quantization") == "fp16":
        suggestions.append(
            {
                "category": "quantization",
                "suggestion": "Consider FP8 quantization for ~2x throughput improvement with minimal quality loss.",
                "estimated_savings": "30-50% cost reduction",
                "priority": "medium",
            }
        )

    # 5. Tensor parallelism
    if cfg.get("tensor_parallel", 1) == 1 and cfg.get("gpu_count", 1) == 1:
        suggestions.append(
            {
                "category": "parallelism",
                "suggestion": "For models >8B parameters, consider tensor parallelism across multiple GPUs.",
                "estimated_savings": "Latency reduction for large models",
                "priority": "low",
            }
        )

    # 6. Latency SLO optimization
    if not result.latency_slo_met:
        suggestions.append(
            {
                "category": "latency",
                "suggestion": "Latency SLO not met. Consider relaxing the SLO or using a faster model.",
                "estimated_savings": "SLO compliance enables higher throughput",
                "priority": "high",
            }
        )

    # 7. Carbon optimization
    if result.carbon_kg > 0.001:
        suggestions.append(
            {
                "category": "carbon",
                "suggestion": "Consider running in a region with lower carbon intensity (e.g., eu-north, eu-france).",
                "estimated_savings": f"Up to {result.carbon_kg * 0.8:.6f} kg CO2 reduction",
                "priority": "low",
            }
        )

    return suggestions


def format_suggestions(suggestions: List[Dict[str, str]]) -> str:
    """Format optimization suggestions for display."""
    if not suggestions:
        return "No optimization suggestions — configuration looks good!"

    lines = ["=" * 55, "  Cost Optimization Suggestions", "=" * 55]

    for i, s in enumerate(suggestions, 1):
        priority_color = {"high": "red", "medium": "yellow", "low": "dim"}.get(
            s["priority"], "dim"
        )
        lines.append(
            f"\n  {i}. [{priority_color}]{s['priority'].upper()}[/{priority_color}] {s['category']}"
        )
        lines.append(f"     {s['suggestion']}")
        if s.get("estimated_savings"):
            lines.append(f"     Estimated savings: {s['estimated_savings']}")

    lines.append("\n" + "=" * 55)
    return "\n".join(lines)
