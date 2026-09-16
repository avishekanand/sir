"""
Throughput Model
================
Empirical and analytical throughput estimation for LLM inference.

Based on arXiv 2606.11690 (Patil, June 2026):
- Empirical Θ_max lookup table from paper Table 4
- Smooth saturation model for Θ_achieved(λ)
- Model-size-dependent saturation knee (λ_sat)

The paper explicitly states: "Θ_achieved is empirically measured, not
derived from an analytical formula." Our lookup table approach follows
this methodology.
"""

import math
from typing import Dict, Optional, Tuple

from ragtune.budget.hardware import get_gpu_spec, GPUSpec


# ── Empirical Θ_max lookup table ──────────────────────────────────────────
# Source: arXiv 2606.11690, Table 4 (Patil, June 2026)
# Measured on H100 NVL GPUs with vLLM defaults (continuous batching,
# PagedAttention). I/O shape: 512 input, 256 output tokens.
#
# Key: (gpu_type, model_name, quantization) → Θ_max (tok/s)

CALIBRATED_THETA_MAX: Dict[Tuple[str, str, str], float] = {
    ("H100-NVL-96GB", "llama-3.1-8b", "fp16"): 6238,
    ("H100-NVL-96GB", "llama-3.1-8b", "fp8"): 8155,
    ("H100-NVL-96GB", "qwen3-30b-a3b", "fp16"): 5319,
    ("H100-NVL-96GB", "qwen3-30b-a3b", "fp8"): 9271,
    ("H100-NVL-96GB", "mixtral-8x7b", "fp16"): 4454,
    ("H100-NVL-96GB", "mixtral-8x7b", "fp8"): 7524,
}

# ── Model profiles ────────────────────────────────────────────────────────
# Source: HuggingFace model cards, sentence-transformers documentation
# (total_params_b, active_params_b, architecture)

MODEL_PROFILES = {
    # Cross-encoders (reranking)
    "cross-encoder/ms-marco-MiniLM-L-6-v2": (0.023, 0.023, "dense"),  # 22.7M params
    "cross-encoder/ms-marco-MiniLM-L-12-v2": (0.034, 0.034, "dense"),  # 33.5M params
    "BAAI/bge-reranker-v2-m3": (0.568, 0.568, "dense"),  # 568M params
    "BAAI/bge-reranker-v2-gemma": (2.6, 2.6, "dense"),  # ~2.6B params
    "castorini/monot5-base-msmarco": (0.22, 0.22, "dense"),  # 220M params
    # LLMs for generation
    "llama-3.1-8b": (8.0, 8.0, "dense"),
    "mixtral-8x7b": (46.7, 12.9, "sparse_moe"),
    "qwen3-30b-a3b": (30.0, 3.0, "ultra_sparse_moe"),
}

# ── Quantization memory factors ──────────────────────────────────────────

QUANT_FACTORS = {
    "fp16": 2.0,  # bytes per param
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}

# ── Model-size-dependent saturation knee ──────────────────────────────────
# λ_sat controls how quickly throughput approaches Θ_max.
# Larger models saturate at lower λ because they have more memory contention.
# Calibrated from paper Table 3 data (H100 NVL).

LAM_SAT_TABLE = {
    0.1: 25.0,  # Small models (cross-encoders): saturate slowly
    1.0: 20.0,  # Medium models
    3.0: 15.0,  # Qwen3-30B-A3B (3B active): paper-calibrated
    8.0: 12.0,  # Llama 3.1 8B: paper-calibrated
    13.0: 10.0,  # Mixtral 8x7B (12.9B active): saturates faster
}


def get_saturation_knee(active_params_b: float, fallback: float = 15.0) -> float:
    """Get λ_sat for a given model size via logarithmic interpolation.

    Args:
        active_params_b: Active parameters in billions.
        fallback: Value returned when no table entry or interpolation applies
            (e.g. empty table). Wired from BudgetConfig.lam_sat_fallback.
    """
    if active_params_b in LAM_SAT_TABLE:
        return LAM_SAT_TABLE[active_params_b]

    sorted_sizes = sorted(LAM_SAT_TABLE.keys())
    if active_params_b <= sorted_sizes[0]:
        return LAM_SAT_TABLE[sorted_sizes[0]]
    if active_params_b >= sorted_sizes[-1]:
        return LAM_SAT_TABLE[sorted_sizes[-1]]

    for i in range(len(sorted_sizes) - 1):
        lo, hi = sorted_sizes[i], sorted_sizes[i + 1]
        if lo <= active_params_b <= hi:
            t = (math.log(active_params_b) - math.log(lo)) / (
                math.log(hi) - math.log(lo)
            )
            return LAM_SAT_TABLE[lo] + t * (LAM_SAT_TABLE[hi] - LAM_SAT_TABLE[lo])

    return fallback


def get_model_profile(
    model_name: str,
    total_params_b: float = 0.1,
    active_params_b: float = 0.1,
    architecture: str = "dense",
) -> tuple:
    """Return (total_params_b, active_params_b, architecture) for a model."""
    if model_name in MODEL_PROFILES:
        return MODEL_PROFILES[model_name]
    return (total_params_b, active_params_b, architecture)


def quant_bytes_per_param(quantization: str) -> float:
    """Get bytes per parameter for a quantization format."""
    return QUANT_FACTORS.get(quantization, 2.0)


def estimate_peak_throughput(
    gpu_type: str,
    model_name: str,
    quantization: str,
    total_params_b: float = 0.1,
    active_params_b: float = 0.1,
    architecture: str = "dense",
    tensor_parallel: int = 1,
    max_batch_size: int = 256,
    kv_overhead_per_token_s: float = 0.00014,
) -> float:
    """Θ_max: peak output-token throughput at saturation.

    Strategy:
    1. Check empirical lookup table (paper Table 4)
    2. Fall back to calibrated analytical model
    """
    # Strategy 1: Empirical lookup (paper Table 4)
    lookup_key = (gpu_type, model_name, quantization)
    if lookup_key in CALIBRATED_THETA_MAX:
        return CALIBRATED_THETA_MAX[lookup_key] * tensor_parallel

    # Strategy 2: Calibrated analytical fallback
    return _estimate_peak_throughput_fallback(
        gpu_type,
        total_params_b,
        active_params_b,
        quantization,
        tensor_parallel,
        max_batch_size,
        kv_overhead_per_token_s,
    )


def _estimate_peak_throughput_fallback(
    gpu_type: str,
    total_params_b: float,
    active_params_b: float,
    quantization: str,
    tensor_parallel: int,
    max_batch_size: int,
    kv_overhead_per_token_s: float = 0.00014,
) -> float:
    """Calibrated analytical throughput for configs not in the empirical table.

    Uses configurable kv_overhead_per_token_s (default 0.00014s, calibrated
    from arXiv 2606.11690 paper data: at batch=256 on H100 NVL with
    Llama 3.1 8B, total step time ≈ 40ms vs weight_read_time ≈ 4ms,
    giving per-token overhead ≈ (40-4)/256 ≈ 0.14ms).
    """
    hw = get_gpu_spec(gpu_type)
    bpb = quant_bytes_per_param(quantization)
    mem_bw = hw.memory_bw_gb_s * tensor_parallel

    params_to_read = active_params_b * 1e9 * bpb
    weight_read_time_s = params_to_read / (mem_bw * (1024**3))

    max_batch = min(max_batch_size, 256)
    step_time_s = weight_read_time_s + max_batch * kv_overhead_per_token_s
    return max_batch / step_time_s


def estimate_actual_throughput(
    gpu_type: str,
    model_name: str,
    quantization: str,
    offered_rps: float,
    latency_slo_ms: int = 300,
    output_tokens: int = 256,
    total_params_b: float = 0.1,
    active_params_b: float = 0.1,
    architecture: str = "dense",
    tensor_parallel: int = 1,
    max_batch_size: int = 256,
    kv_overhead_per_token_s: float = 0.00014,
    lam_sat_fallback: float = 15.0,
    peak_utilization_threshold: float = 0.9,
) -> Tuple[float, float]:
    """Θ_achieved(λ): throughput under offered load.

    Returns (achieved_tps, achieved_batch).
    """
    # Auto-resolve model profile if model_name is in MODEL_PROFILES
    profile = get_model_profile(
        model_name, total_params_b, active_params_b, architecture
    )
    total_b, active_b, arch = profile

    peak = estimate_peak_throughput(
        gpu_type,
        model_name,
        quantization,
        total_b,
        active_b,
        arch,
        tensor_parallel,
        max_batch_size,
        kv_overhead_per_token_s,
    )
    lam = max(offered_rps, 1.0)

    # Arrival-limited throughput
    arrival_tps = lam * output_tokens

    # Model-size-dependent saturation knee (use resolved active_b)
    lam_sat = get_saturation_knee(active_b, fallback=lam_sat_fallback)

    # Smooth saturation
    saturation_factor = 1.0 - math.exp(-lam / lam_sat)
    saturated_tps = peak * saturation_factor

    achieved_tps = min(arrival_tps, saturated_tps)

    # SLO enforcement — only DECREASE throughput, never increase
    if latency_slo_ms and latency_slo_ms > 0 and achieved_tps > 0:
        est_latency_ms = (output_tokens / achieved_tps) * 1000
        if est_latency_ms > latency_slo_ms:
            max_tps_for_slo = output_tokens / (latency_slo_ms / 1000.0)
            achieved_tps = min(achieved_tps, max_tps_for_slo)

    # Achieved batch size
    if achieved_tps >= peak * peak_utilization_threshold:
        achieved_batch = max_batch_size
    elif output_tokens > 0:
        achieved_batch = achieved_tps / output_tokens
    else:
        achieved_batch = 0.0

    return max(achieved_tps, 1.0), achieved_batch


def estimate_weight_vram(total_params_b: float, quantization: str) -> float:
    """VRAM for model weights in GB."""
    bpb = quant_bytes_per_param(quantization)
    return total_params_b * 1e9 * bpb / (1024**3)
