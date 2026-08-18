"""
GPU Hardware Specifications
============================
Single source of truth for GPU specs used by all budget loaders.

Source: NVIDIA official datasheets (nvidia.com/en-us/data-center/h100/,
        nvidia.com/en-us/data-center/a100/)
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class GPUSpec:
    """Immutable GPU hardware specification."""

    name: str
    memory_bw_gb_s: int  # Memory bandwidth (GB/s)
    compute_tflops_fp16: int  # FP16 Tensor Core TFLOPS (dense)
    compute_tflops_fp8: int  # FP8 Tensor Core TFLOPS (dense, 0 if unsupported)
    vram_gb: int  # VRAM (GB)
    tdp_w: int  # TDP (Watts)
    hourly_rate: float  # Cloud on-demand hourly rate ($)

    @property
    def has_fp8(self) -> bool:
        return self.compute_tflops_fp8 > 0


# ── Hardware database ──────────────────────────────────────────────────────
# H100 NVL specs differ from H100 SXM — key variant distinction.

GPU_SPECS: Dict[str, GPUSpec] = {
    "H100-NVL-96GB": GPUSpec(
        name="H100-NVL-96GB",
        memory_bw_gb_s=3900,  # H100 NVL: 3.9 TB/s
        compute_tflops_fp16=835,  # H100 NVL dense FP16
        compute_tflops_fp8=1670,  # H100 NVL dense FP8
        vram_gb=94,  # H100 NVL: 94 GB HBM3
        tdp_w=400,  # H100 NVL: 350-400W
        hourly_rate=4.50,  # Azure on-demand estimate
    ),
    "A100-80GB": GPUSpec(
        name="A100-80GB",
        memory_bw_gb_s=2039,  # A100 SXM: 2.039 TB/s
        compute_tflops_fp16=312,  # A100 dense FP16
        compute_tflops_fp8=0,  # No native FP8 on A100 (Ampere)
        vram_gb=80,  # A100: 80 GB HBM2e
        tdp_w=400,  # A100 SXM: 400W
        hourly_rate=3.50,  # Azure on-demand estimate
    ),
    "A100-40GB": GPUSpec(
        name="A100-40GB",
        memory_bw_gb_s=1555,  # A100-40GB: 1.555 TB/s
        compute_tflops_fp16=312,  # Same compute as 80GB variant
        compute_tflops_fp8=0,  # No native FP8
        vram_gb=40,  # A100-40GB: 40 GB HBM2e
        tdp_w=400,  # A100-40GB SXM: 400W
        hourly_rate=2.90,  # Azure on-demand estimate
    ),
    "V100-32GB": GPUSpec(
        name="V100-32GB",
        memory_bw_gb_s=900,  # V100 SXM2: 900 GB/s
        compute_tflops_fp16=125,  # V100 dense FP16
        compute_tflops_fp8=0,  # No FP8 on V100 (Volta)
        vram_gb=32,  # V100: 32 GB HBM2
        tdp_w=300,  # V100 SXM2: 300W
        hourly_rate=2.00,  # Cloud estimate
    ),
    "T4-16GB": GPUSpec(
        name="T4-16GB",
        memory_bw_gb_s=320,  # T4: 320 GB/s
        compute_tflops_fp16=65,  # T4 dense FP16
        compute_tflops_fp8=0,  # No FP8 on T4 (Turing)
        vram_gb=16,  # T4: 16 GB GDDR6
        tdp_w=70,  # T4: 70W
        hourly_rate=0.80,  # Cloud estimate
    ),
    "L4-24GB": GPUSpec(
        name="L4-24GB",
        memory_bw_gb_s=300,  # L4: 300 GB/s
        compute_tflops_fp16=242,  # L4 dense FP16
        compute_tflops_fp8=485,  # L4 dense FP8
        vram_gb=24,  # L4: 24 GB GDDR6
        tdp_w=72,  # L4: 72W
        hourly_rate=1.00,  # Cloud estimate
    ),
}

# Default GPU when type not found
DEFAULT_GPU = "A100-80GB"


def get_gpu_spec(gpu_type: str) -> GPUSpec:
    """Get GPU spec by type, falling back to default."""
    return GPU_SPECS.get(gpu_type, GPU_SPECS[DEFAULT_GPU])


def list_gpu_types():
    """List all available GPU types."""
    return list(GPU_SPECS.keys())


def estimate_gpu_power(
    gpu_type: str,
    gpu_count: int,
    utilization: float,
    idle_fraction: float = 0.25,
    active_fraction: float = 0.75,
) -> float:
    """GPU power draw in watts, adjusted for utilization.

    Linear model: power = TDP × (idle_fraction + active_fraction × util) × gpu_count

    Source: NVIDIA GPU power management documentation.
    Idle power ≈ 25% of TDP (memory controllers, NVLink, PCIe).
    Peak power = TDP at 100% utilization.

    Args:
        gpu_type: GPU type (e.g., "A100-80GB", "H100-NVL-96GB")
        gpu_count: Number of GPUs
        utilization: GPU utilization (0.0 to 1.0)
        idle_fraction: Fraction of TDP at idle (default 0.25)
        active_fraction: Fraction of TDP swing from idle to peak (default 0.75)
    """
    spec = get_gpu_spec(gpu_type)
    return spec.tdp_w * (idle_fraction + active_fraction * utilization) * gpu_count


def estimate_energy_kwh(
    power_w: float,
    time_s: float,
    pue: float = 1.15,
) -> float:
    """Estimate energy consumption in kWh.

    Includes PUE (Power Usage Effectiveness) to account for cooling,
    power distribution, and other facility overhead.

    Source: IPCC Tier 1 methodology, Google Cloud Carbon Footprint.
    PUE default: 1.15 (hyperscale average, Uptime Institute 2024).

    Args:
        power_w: IT equipment power in watts
        time_s: Duration in seconds
        pue: Power Usage Effectiveness (>= 1.0)
    """
    return power_w * pue * time_s / 3600 / 1000


def estimate_carbon_kg(
    energy_kwh: float,
    carbon_intensity_g_per_kwh: float,
) -> float:
    """Estimate carbon emissions in kg CO2e.

    Formula: carbon_kg = energy_kwh × intensity_g_per_kwh / 1000

    Source: IPCC Tier 1 methodology, GHG Protocol Scope 2.
    """
    return energy_kwh * carbon_intensity_g_per_kwh / 1000


# ── CPU Power Model ──────────────────────────────────────────────────────
# CPU power is typically 10-50% of TDP depending on workload.
# For LLM inference (memory-bound), CPU utilization is low.

CPU_IDLE_FRACTION = 0.15  # CPU idle power ~15% of TDP
CPU_ACTIVE_FRACTION = 0.85  # CPU active power swing


def estimate_cpu_power(
    cpu_tdp_w: int,
    num_cores: int = 1,
    num_threads: int = 1,
    utilization: float = 0.5,
    total_cores: int = 0,
) -> float:
    """Estimate CPU power draw in watts.

    Args:
        cpu_tdp_w: CPU TDP in watts
        num_cores: Number of physical cores actively used, OR a fraction
            (0.0-1.0) of the CPU being used. If total_cores is provided and
            num_cores is a count, power scales by num_cores/total_cores.
        num_threads: Number of logical threads used (informational; cores drive
            the scaling since SMT threads share the same physical core).
        utilization: CPU utilization (0.0 to 1.0).
        total_cores: Total physical cores on the CPU. If given and > 0,
            num_cores is treated as a count and scaled to a fraction.
    """
    if cpu_tdp_w <= 0:
        return 0.0
    # Resolve the active-core fraction:
    #   - fraction form (0.0-1.0): use directly
    #   - count form (>=1) with total_cores: num_cores / total_cores
    #   - count form without total_cores: treat as full CPU (no scaling)
    if 0.0 < num_cores <= 1.0:
        core_fraction = num_cores
    elif total_cores and total_cores > 0:
        core_fraction = min(num_cores / total_cores, 1.0)
    else:
        core_fraction = 1.0
    effective_tdp = cpu_tdp_w * core_fraction
    return effective_tdp * (CPU_IDLE_FRACTION + CPU_ACTIVE_FRACTION * utilization)


def estimate_total_system_power(
    gpu_power_w: float = 0.0,
    cpu_power_w: float = 0.0,
    memory_power_w: float = 0.0,
    network_power_w: float = 0.0,
) -> float:
    """Estimate total system power in watts.

    Args:
        gpu_power_w: GPU power draw
        cpu_power_w: CPU power draw
        memory_power_w: Memory power draw
        network_power_w: Network power draw
    """
    return gpu_power_w + cpu_power_w + memory_power_w + network_power_w
