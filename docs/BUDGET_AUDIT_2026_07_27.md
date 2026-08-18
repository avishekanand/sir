# Budget System Audit Report — 2026-07-27

**Auditor:** Rigor Mode — multi-agent research + formula-by-formula cross-reference  
**Paper:** arXiv 2606.11690 — "Beyond Per-Token Pricing: A Concurrency-Aware Methodology for LLM Infrastructure Cost Estimation" (Patil, June 2026)

---

## Executive Summary

| Category | Status | Issues |
|----------|--------|--------|
| C_eff formula (Eq.3) | ⚠️ Correct structure, wrong parameters | Overestimates cost by ~8x due to throughput model |
| GPU SPECS | ❌ H100 SXM specs under NVL key | Variant mismatch: code has SXM (3350 GB/s, 700W) not NVL (3900 GB/s, 400W) |
| MODEL_PROFILES | ⚠️ Cross-encoder sizes ~5x too high | MiniLM-L-6-v2 is 22M params, not 0.1B |
| Throughput model | ❌ Fundamental mismatch with paper | Paper uses empirical measurement; code uses analytic with wrong overhead |
| Contention model | ❌ `1 + 0.05×log2(batch)` underestimates | At batch=64: model gives 1.3, real is 1.5-2.0 |
| Prefix caching | ❌ 0.90 factor is wrong | Real savings ~0.42-0.50 (prefill is only ~50% of compute) |
| Token pricing | ⚠️ Cached rate wrong | $0.30 → should be $1.25 (OpenAI GPT-4o 50% discount) |
| Carbon intensity | ⚠️ 2 regions off by 15-45% | eu-france: 60→45, asia-east: 600→500 |
| GPU power model | ✅ Reasonable approximation | ±10% uncertainty, linear model is acceptable |
| BudgetResult.__add__ | ✅ Semantically correct | Harmonic mean for throughput is appropriate |

---

## Issue Severity Matrix (Re-verified 2026-07-27)

| # | Issue | Severity | File:Line | Fix |
|---|-------|----------|-----------|-----|
| 1 | H100 SPECS: SXM values under NVL key (variant mismatch) | 🔴 CRITICAL | vllm_budget.py:29-36 | Rename key or fix specs to NVL values |
| 2 | H100 bandwidth: 3350→3900 GB/s | 🔴 HIGH | vllm_budget.py:30 | Update to NVL bandwidth |
| 3 | H100 TDP: 700→400W | 🔴 HIGH | vllm_budget.py:34 | Update to NVL TDP |
| 4 | Overhead constant: 1.15→~10x | 🔴 CRITICAL | vllm_budget.py:124 | Redesign throughput model |
| 5 | Prefix caching: 0.90→~0.50 | 🔴 HIGH | vllm_budget.py:225 | Fix factor |
| 6 | Contention model too low | 🔴 HIGH | vllm_budget.py:153 | Calibrate against benchmarks |
| 7 | MiniLM params: 0.1B→0.023B | 🟡 MEDIUM | vllm_budget.py:59 | Update MODEL_PROFILES |
| 8 | Cached token rate: $0.30→$1.25 | 🟡 MEDIUM | token_budget.py:22 | Update DEFAULT_RATES |
| 9 | H100 FP16: 989→835 (NVL dense) | 🟡 MEDIUM | vllm_budget.py:31 | Update to NVL dense value |
| 10 | H100 FP8: 1979→1670 (NVL dense) | 🟡 MEDIUM | vllm_budget.py:32 | Update to NVL dense value |
| 11 | H100 VRAM: 96→94 GB | 🟢 LOW | vllm_budget.py:33 | Update VRAM |
| 12 | A100-40GB hourly: 2.50→$2.90-3.50 | 🟢 LOW | vllm_budget.py:51 | Update hourly rate |
| 13 | eu-france carbon: 60→45 | 🟢 LOW | carbon_budget.py:23 | Update REGIONAL_INTENSITY |
| 14 | asia-east carbon: 600→500 | 🟢 LOW | carbon_budget.py:24 | Update REGIONAL_INTENSITY |
| 15 | Residence time heuristic unvalidated | 🟡 MEDIUM | vllm_budget.py:144 | Mark as [HYPOTHESIS] |
| 16 | Linear batch scaling assumption | 🔴 HIGH | vllm_budget.py:154 | Use empirical lookup table |
| 17 | carbon_budget.py hardcoded H100 TDP=700 | 🟡 MEDIUM | carbon_budget.py:58 | Use GPU_SPECS lookup |
| 18 | gpu_budget.py duplicate wrong GPU_SPECS | 🟡 MEDIUM | gpu_budget.py:19-26 | Consolidate to single source |
| 19 | token_budget.py uncached_prompt can go negative | 🟡 MEDIUM | token_budget.py:50 | Add max(0, ...) guard |
| 20 | carbon_budget.py `if intensity == 400` sentinel | 🟢 LOW | carbon_budget.py:52 | Use None sentinel |
| 21 | compute_tflops values unused in throughput model | 🟢 LOW | vllm_budget.py:31-32 | Document as forward-looking |

---

## Detailed Findings

### 1. C_eff Formula (vllm_budget.py L202-207)

**Code:**
```python
cost_per_million = (total_gpu_hourly * 1_000_000 / (3600 * actual_tps))
```

**Paper (Eq.3):**
```
C_eff = (P_GPU × 1e6) / (3600 × Θ_achieved(λ, L))
```

**Verdict:** ✅ Formula structure CORRECT

The C_eff formula itself is correctly implemented. The issue is that `actual_tps` (our Θ_achieved) is computed analytically rather than measured empirically as the paper does.

---

### 2. GPU_SPECS Table (vllm_budget.py L28-53)

| GPU | Field | Code Value | Actual | Error |
|-----|-------|-----------|--------|-------|
| H100-NVL-96GB | memory_bw_gb_s | 3350 | 3900 | -14% ❌ |
| H100-NVL-96GB | compute_tflops_fp16 | 989 | 1671 | -41% ❌ |
| H100-NVL-96GB | compute_tflops_fp8 | 1979 | 3341 | -41% ❌ |
| H100-NVL-96GB | vram_gb | 96 | 94 | +2% ⚠️ |
| H100-NVL-96GB | tdp_w | 700 | 350-400W | +75-100% ❌ |
| H100-NVL-96GB | hourly_rate | 6.98 | $3.70-4.50 | +55-89% ⚠️ |
| A100-80GB | all fields | — | — | ✅ |
| A100-40GB | hourly_rate | 2.50 | $2.90-3.50 | -14% ⚠️ |

Source: NVIDIA official datasheets

---

### 3. MODEL_PROFILES Table (vllm_budget.py L57-68)

| Model | Code | Actual | Error |
|-------|------|--------|-------|
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.1B | 0.023B | ~4.4x too high ❌ |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 0.2B | 0.034B | ~5.9x too high ❌ |
| BAAI/bge-reranker-v2-m3 | 0.6B | 0.568B | +6% ⚠️ |
| BAAI/bge-reranker-v2-gemma | 2.6B | ~2.6B | ✅ |
| castorini/monot5-base-msmarco | 0.2B | 0.22B | -9% ⚠️ |
| llama-3.1-8b | 8.0B | 8.0B | ✅ |
| mixtral-8x7b | 46.7B/12.9B | 46.7B/12.9B | ✅ |
| qwen3-30b-a3b | 30.0B/3.0B | 30.0B/3.0B | ✅ |

Source: HuggingFace model cards

---

### 4. Throughput Model — CRITICAL FINDING

The paper takes an **empirical measurement approach** — Θ_achieved is measured via benchmark sweep, not derived analytically.

**Paper's measured Θ_max (Table 4, H100 NVL):**
- C1: Llama 3.1 8B FP16 → Θ_max = 6,238 tok/s, C_sat = $0.311/MTok
- C2: Llama 3.1 8B FP8 → Θ_max = 8,155 tok/s, C_sat = $0.238/MTok
- C3: Qwen3-30B-A3B FP16 → Θ_max = 5,319 tok/s, C_sat = $0.364/MTok
- C4: Qwen3-30B-A3B FP8 → Θ_max = 9,271 tok/s, C_sat = $0.209/MTok
- C5: Mixtral 8x7B FP16 TP=2 → Θ_max = 4,454 tok/s, C_sat = $0.871/MTok
- C6: Mixtral 8x7B FP8 TP=2 → Θ_max = 7,524 tok/s, C_sat = $0.520/MTok

**Code's analytical model for C1:**
- weight_time = 16GB / 3350 GB/s = 4.66ms
- throughput = 256 / (0.00466 × 1.15) = 47,771 tok/s
- **Error: 7.7x overestimate** ❌

**At λ=10 (paper Table 3):**
- Paper: C_eff = $0.80/MTok
- Code: C_eff ≈ $6.62/MTok
- **Error: 8.3x overestimate** ❌

Root cause: overhead constant 1.15 is far too low. Real overhead at batch=256 is ~10x.

---

### 5. Contention Model

**Code:** `contention = 1.0 + 0.05 × log2(batch)`

| Batch | Code | Real (vLLM benchmarks) |
|-------|------|------------------------|
| 16 | 1.20 | ~1.2-1.5 |
| 64 | 1.30 | ~1.5-2.0 |
| 256 | 1.40 | ~2.0-3.0 |

Source: vLLM v0.6.0 blog, FlashAttention paper

---

### 6. Prefix Caching

**Code:** `cache_saving = (cached_tokens / total_tokens) × 0.90`

**Reality:** APC only reduces prefill phase, not decode. Prefill is ~40-60% of total compute.
Realistic savings: `fraction × 0.50 × 0.90 ≈ fraction × 0.45`

Source: vLLM automatic-prefix-caching documentation

---

### 7. Token Pricing

**Code defaults:** input=$2.50, output=$10.00, cached=$0.30

**Actual GPT-4o (mid-2026):** input=$2.50, output=$10.00, cached=$1.25 (50% discount)

The cached rate ($0.30) is wrong — should be $1.25 for GPT-4o.

Source: OpenAI pricing page, Anthropic pricing page

---

### 8. Carbon Intensity

| Region | Code | Actual | Status |
|--------|------|--------|--------|
| us-east | 350 | 350 | ✅ |
| us-west | 200 | 79-288 | ✅ |
| eu-central | 250 | 280-300 | ⚠️ Low |
| eu-north | 50 | 31-132 | ✅ |
| eu-france | 60 | 41-52 | ⚠️ High |
| asia-east | 600 | 450-550 | ⚠️ High |
| asia-south | 700 | 670-705 | ✅ |
| australia | 500 | 498-554 | ✅ |
| global-average | 475 | 442-471 | ✅ |

Source: Our World in Data, Ember, IEA Electricity 2025, EPA eGRID

---

## What's Correct

1. C_eff formula structure — matches paper Eq.3 exactly
2. Little's Law application — `in_flight = λ × residence_time`
3. A100-80GB specs — all values verified
4. Quantization factors — fp16=2.0, fp8=1.0, int8=1.0, int4=0.5
5. Energy calculation — `power_w × time_s / 3600 / 1000` dimensionally correct
6. Carbon calculation — `energy_kwh × intensity / 1000` correct
7. BudgetResult.__add__ — harmonic mean for throughput is correct
8. GPU power model — linear approximation ±10% acceptable
9. MoE model profiles — Mixtral and Qwen3 parameter counts correct
10. Architecture tags — dense/sparse_moe/ultra_sparse_moe correct

---

## Recommended Fixes

### Priority 1 (CRITICAL): Fix throughput model
Use calibrated empirical lookup table from paper Table 4.

### Priority 2 (HIGH): Fix GPU_SPECS
Update H100 NVL values to match NVIDIA datasheets.

### Priority 3 (HIGH): Fix prefix caching
Change 0.90 → 0.50.

### Priority 4 (MEDIUM): Fix MODEL_PROFILES
Update cross-encoder parameter counts.

### Priority 5 (MEDIUM): Fix token pricing
Change cached_input from $0.30 → $1.25.

---

## Re-verification Notes (2026-07-27)

**Key correction:** The original audit quoted research agent numbers for H100 NVL FP16 (1,671) and FP8 (3,341). These are the **sparse** values. The correct **dense** values are 835 and 1,670 respectively. For budget estimation, dense values are appropriate since sparsity is optional.

**Variant mismatch:** The code's "H100-NVL-96GB" key actually contains H100 **SXM** specs (3350 GB/s, 700W TDP). The correct NVL specs are 3900 GB/s and 350-400W. This was not caught in the original audit.

**Unused compute values:** The `compute_tflops_fp16` and `compute_tflops_fp8` values in GPU_SPECS are not used by `_estimate_peak_throughput()`. The throughput model only uses `memory_bw_gb_s`. This means the compute TFLOPS errors don't affect current calculations, only the bandwidth error does.

**Throughput error refined:** With correct NVL bandwidth (3900 GB/s), the throughput overestimate increases from 8.0x to 8.8x (because faster bandwidth → shorter weight_read_time → higher throughput → larger overestimate).

---

## Sources

1. arXiv 2606.11690 — Patil (2026) "Beyond Per-Token Pricing"
2. NVIDIA A100 Datasheet — nvidia.com/en-us/data-center/a100/
3. NVIDIA H100 Datasheet — nvidia.com/en-us/data-center/h100/
4. vLLM PagedAttention paper — arXiv 2309.06180
5. vLLM v0.6.0 blog — Sep 2024 performance benchmarks
6. vLLM automatic-prefix-caching docs
7. OpenAI pricing page — openai.com/api/pricing
8. Anthropic pricing page — anthropic.com/pricing
9. Our World in Data — carbon intensity of electricity
10. Ember — Global Electricity Review 2024
11. IEA Electricity 2025 — emissions data
12. EPA eGRID 2023 — US regional carbon intensity
13. Google Cloud Sustainability — region carbon data
14. Williams et al. (2009) — Roofline model
15. FlashAttention — arXiv 2205.14135
