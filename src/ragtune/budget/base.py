"""
Base Budget Loader
=================
Abstract base class for all budget calculation backends.

Every loader produces a BudgetResult with cost in USD, carbon, kWh, and
tokens. The specific formula depends on the loader — vLLM concurrency-aware,
simple token counting, GPU utilization, etc.

Usage:
    class MyBudgetLoader(BaseBudgetLoader):
        def calculate(self, context) -> BudgetResult:
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ragtune.budget.result import BudgetResult


class BudgetConfig:
    """Configuration for a budget calculation.

    All parameters that affect cost/energy/carbon estimation are configurable.
    No hardcoded values in loader code — everything flows from this config.

    Source citations for defaults:
    - gpu_type: "A100-80GB" — most common cloud GPU (AWS/Azure/GCP)
    - pue: 1.15 — hyperscale average (Uptime Institute 2024)
    - carbon_intensity: 450 gCO2/kWh — Ember Global Electricity Review 2025
    - electricity_cost: $0.12/kWh — US commercial average (EIA 2024)
    - kv_overhead_per_token_s: 0.00014 — calibrated from arXiv 2606.11690
    - cache_saving_fraction: 0.50 — vLLM APC documentation (prefill only)
    - default_tokens_per_doc: 512 — conservative estimate for reranking
    - escalation_gap_threshold: 0.05 — strategy escalation trigger
    - gpu_power_idle_fraction: 0.25 — NVIDIA GPU idle power (25% of TDP)
    - gpu_power_active_fraction: 0.75 — NVIDIA GPU active power swing
    """

    def __init__(self, config: Dict[str, Any]):
        # ── GPU Configuration ──
        self.gpu_type: str = config.get("gpu_type", "A100-80GB")
        self.gpu_count: int = config.get("gpu_count", 1)
        self.gpu_hourly_rate: float = config.get("gpu_hourly_rate", 0.0)

        # ── Cloud/Region Configuration ──
        self.region: str = config.get("region", "us-east")
        self.electricity_cost_per_kwh: float = config.get(
            "electricity_cost_per_kwh", 0.12
        )
        self.carbon_intensity_g_per_kwh: float = config.get(
            "carbon_intensity_g_per_kwh", 450.0
        )
        self._carbon_intensity_set: bool = "carbon_intensity_g_per_kwh" in config
        self.pue: float = config.get("pue", 1.15)  # Power Usage Effectiveness

        # ── Model Configuration ──
        self.model_name: str = config.get(
            "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.model_architecture: str = config.get("model_architecture", "dense")
        self.active_params_b: float = config.get("active_params_b", 0.1)
        self.total_params_b: float = config.get("total_params_b", 0.1)
        self.quantization: str = config.get("quantization", "fp16")

        # ── Throughput Configuration ──
        self.max_batch_size: int = config.get("max_batch_size", 256)
        self.latency_slo_ms: int = config.get("latency_slo_ms", 500)
        self.offered_rps: float = config.get("offered_rps", 10.0)
        self.tensor_parallel: int = config.get("tensor_parallel", 1)
        self.pipeline_parallel: int = config.get("pipeline_parallel", 1)
        self.kv_overhead_per_token_s: float = config.get(
            "kv_overhead_per_token_s", 0.00014
        )
        self.lam_sat_fallback: float = config.get("lam_sat_fallback", 15.0)
        self.peak_utilization_threshold: float = config.get(
            "peak_utilization_threshold", 0.9
        )

        # ── Cache Configuration ──
        self.cache_saving_fraction: float = config.get("cache_saving_fraction", 0.50)
        self.cache_hit_rate: float = config.get("cache_hit_rate", 0.0)

        # ── Token Estimation ──
        self.default_tokens_per_doc: int = config.get("default_tokens_per_doc", 512)

        # ── Scheduler Configuration ──
        self.escalation_gap_threshold: float = config.get(
            "escalation_gap_threshold", 0.05
        )

        # ── GPU Power Model ──
        self.gpu_power_idle_fraction: float = config.get(
            "gpu_power_idle_fraction", 0.25
        )
        self.gpu_power_active_fraction: float = config.get(
            "gpu_power_active_fraction", 0.75
        )

        # ── CPU Configuration ──
        self.cpu_type: str = config.get(
            "cpu_type", ""
        )  # e.g., "Intel Xeon Platinum 8375C"
        self.cpu_cores: int = config.get(
            "cpu_cores", 0
        )  # Physical cores (0 = not configured)
        self.cpu_threads: int = config.get(
            "cpu_threads", 0
        )  # Logical threads (0 = not configured)
        self.cpu_hourly_rate: float = config.get("cpu_hourly_rate", 0.0)  # $/hr per CPU
        self.cpu_tdp_w: int = config.get(
            "cpu_tdp_w", 0
        )  # CPU TDP in watts (0 = not configured)
        self.num_processes: int = config.get(
            "num_processes", 1
        )  # Number of parallel processes
        self.num_threads_per_process: int = config.get(
            "num_threads_per_process", 1
        )  # Threads per process

        # ── Extra (loader-specific) ──
        self.extra: Dict[str, Any] = config.get("extra", {})

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def validate(self) -> List[str]:
        """Validate configuration values. Returns list of error messages."""
        errors = []
        if self.gpu_count < 1:
            errors.append(f"gpu_count must be >= 1, got {self.gpu_count}")
        if self.latency_slo_ms < 0:
            errors.append(f"latency_slo_ms must be >= 0, got {self.latency_slo_ms}")
        if self.offered_rps < 0:
            errors.append(f"offered_rps must be >= 0, got {self.offered_rps}")
        if self.max_batch_size < 1:
            errors.append(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        if self.tensor_parallel < 1:
            errors.append(f"tensor_parallel must be >= 1, got {self.tensor_parallel}")
        if self.electricity_cost_per_kwh < 0:
            errors.append(
                f"electricity_cost_per_kwh must be >= 0, got {self.electricity_cost_per_kwh}"
            )
        if self.carbon_intensity_g_per_kwh < 0:
            errors.append(
                f"carbon_intensity_g_per_kwh must be >= 0, got {self.carbon_intensity_g_per_kwh}"
            )
        if self.pue < 1.0:
            errors.append(f"pue must be >= 1.0, got {self.pue}")
        if self.kv_overhead_per_token_s < 0:
            errors.append(
                f"kv_overhead_per_token_s must be >= 0, got {self.kv_overhead_per_token_s}"
            )
        if not (0.0 <= self.cache_saving_fraction <= 1.0):
            errors.append(
                f"cache_saving_fraction must be in [0, 1], got {self.cache_saving_fraction}"
            )
        if not (0.0 <= self.gpu_power_idle_fraction <= 1.0):
            errors.append(
                f"gpu_power_idle_fraction must be in [0, 1], got {self.gpu_power_idle_fraction}"
            )
        if not (0.0 <= self.gpu_power_active_fraction <= 1.0):
            errors.append(
                f"gpu_power_active_fraction must be in [0, 1], got {self.gpu_power_active_fraction}"
            )
        if self.cpu_cores < 0:
            errors.append(f"cpu_cores must be >= 0, got {self.cpu_cores}")
        if self.cpu_threads < 0:
            errors.append(f"cpu_threads must be >= 0, got {self.cpu_threads}")
        if self.num_processes < 1:
            errors.append(f"num_processes must be >= 1, got {self.num_processes}")
        if self.num_threads_per_process < 1:
            errors.append(
                f"num_threads_per_process must be >= 1, got {self.num_threads_per_process}"
            )
        return errors


class BaseBudgetLoader(ABC):
    """Abstract base class for all budget calculation backends.

    Subclasses implement `calculate()` which takes a BudgetConfig and
    optional per-request context, and returns a BudgetResult.

    The class-level `key` attribute identifies this loader in the
    registry (e.g., "vllm", "token", "gpu_util").
    """

    key: str = ""

    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig({})

    @abstractmethod
    def calculate(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetResult:
        """Calculate cost for the given operation context.

        Args:
            context: Optional per-request/per-batch context containing
                things like token counts, batch size, latency, etc.

        Returns:
            BudgetResult with cost, carbon, energy, tokens.
        """
        ...
