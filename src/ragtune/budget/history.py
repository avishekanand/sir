"""
Cost History Logger
====================
Logs budget calculations to a JSONL file for historical analysis.

Each entry contains:
- timestamp
- budget_type
- config snapshot
- context (input parameters)
- result (BudgetResult as dict)

Usage:
    from ragtune.budget.history import CostHistoryLogger

    logger = CostHistoryLogger("cost_history.jsonl")
    logger.log("vllm", config_dict, context_dict, result)
    entries = logger.query(budget_type="vllm", since="2026-07-01")
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from ragtune.budget.result import BudgetResult


class CostHistoryLogger:
    """Logs budget calculations for historical analysis.

    Writes JSONL (one JSON object per line) for efficient append-only logging.
    """

    def __init__(self, path: str = "cost_history.jsonl"):
        self.path = path

    def log(
        self,
        budget_type: str,
        config: Dict[str, Any],
        context: Dict[str, Any],
        result: BudgetResult,
    ) -> None:
        """Log a budget calculation entry.

        Args:
            budget_type: Loader type (vllm, token, gpu_util, carbon, etc.)
            config: BudgetConfig as dict
            context: Input context (tokens, rps, etc.)
            result: BudgetResult output
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "budget_type": budget_type,
            "config": config,
            "context": context,
            "result": {
                "cost_usd": result.cost_usd,
                "cost_per_million_tokens": result.cost_per_million_tokens,
                "energy_kwh": result.energy_kwh,
                "carbon_kg": result.carbon_kg,
                "total_tokens": result.total_tokens,
                "throughput_tok_s": result.throughput_tok_s,
                "gpu_utilization": result.gpu_utilization,
                "latency_slo_met": result.latency_slo_met,
            },
        }

        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def query(
        self,
        budget_type: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query logged entries.

        Args:
            budget_type: Filter by loader type
            since: ISO timestamp to filter from
            limit: Max entries to return

        Returns:
            List of matching entries
        """
        if not os.path.exists(self.path):
            return []

        entries = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if budget_type and entry.get("budget_type") != budget_type:
                    continue
                if since and entry.get("timestamp", "") < since:
                    continue

                entries.append(entry)
                if len(entries) >= limit:
                    break

        return entries

    def summary(self, budget_type: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics from logged entries.

        Returns:
            Dict with count, total_cost, avg_cost, total_tokens, etc.
        """
        entries = self.query(budget_type=budget_type, limit=10000)

        if not entries:
            return {"count": 0}

        total_cost = sum(e.get("result", {}).get("cost_usd", 0) for e in entries)
        total_tokens = sum(e.get("result", {}).get("total_tokens", 0) for e in entries)
        total_energy = sum(e.get("result", {}).get("energy_kwh", 0) for e in entries)
        total_carbon = sum(e.get("result", {}).get("carbon_kg", 0) for e in entries)

        return {
            "count": len(entries),
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_usd": round(total_cost / len(entries), 8) if entries else 0,
            "total_tokens": total_tokens,
            "total_energy_kwh": round(total_energy, 8),
            "total_carbon_kg": round(total_carbon, 8),
            "first_timestamp": entries[0].get("timestamp"),
            "last_timestamp": entries[-1].get("timestamp"),
        }

    def clear(self) -> None:
        """Clear all logged entries."""
        if os.path.exists(self.path):
            os.remove(self.path)
