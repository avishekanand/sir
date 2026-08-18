# Budget System

## Overview

RAGtune's budget system estimates LLM inference cost, energy consumption, and carbon footprint. It provides 6 budget loaders covering different cost models, from GPU-based inference to API pricing.

**Key principle:** All parameters flow through `BudgetConfig` — no hardcoded values in Python code. Every value is configurable via YAML, environment variables, or CLI options.

---

## Architecture

```
budget/
├── base.py              # BudgetConfig (34 fields), BaseBudgetLoader ABC
├── factory.py           # BudgetLoaderFactory registry
├── hardware.py          # GPUSpec dataclass, power/energy/carbon functions
├── throughput.py        # Θ_max lookup, saturation model, power estimation
├── main.py              # calculate_budget(), budget_report()
├── result.py            # BudgetResult (17 fields, per-component costs)
├── optimizer.py         # Cost optimization suggestions
├── history.py           # JSONL cost logging
├── alerts.py            # Threshold-based alerts
├── configs/             # YAML configs (default.yaml, h100_us_east.yaml)
└── loaders/
    ├── vllm_budget.py   # Concurrency-aware (arXiv 2606.11690)
    ├── token_budget.py  # API per-token pricing
    ├── gpu_budget.py    # GPU runtime × hourly rate
    ├── carbon_budget.py # IPCC Tier 1 + PUE
    ├── embedding_budget.py  # OpenAI/Cohere/Voyage embedding pricing
    └── reranking_budget.py  # Cohere/Voyage per-query pricing
```

---

## Loaders

### 1. VLLM Budget Loader

**Source:** arXiv 2606.11690 (Patil, June 2026)

**Formula:**
```
C_eff = (P_GPU × 1e6) / (3600 × Θ_achieved(λ, L))
```

**How it works:**
1. Look up empirical Θ_max from paper Table 4 (if available)
2. Estimate Θ_achieved using saturation model: `min(λ × output_tokens, Θ_max × (1 - e^(-λ/λ_sat)))`
3. Apply SLO enforcement if latency would breach
4. Compute cost, energy, carbon from throughput and GPU specs

**Config keys:**
- `gpu_type`: GPU hardware (e.g., "H100-NVL-96GB")
- `model_name`: Model (e.g., "llama-3.1-8b")
- `offered_rps`: Request rate (requests/sec)
- `latency_slo_ms`: Latency SLO (milliseconds)
- `cache_hit_rate`: Statistical cache hit rate (0.0-1.0)
- `pue`: Power Usage Effectiveness (default 1.15)

**Usage:**
```bash
ragtune budget --gpu H100-NVL-96GB --model llama-3.1-8b --rps 25
```

---

### 2. Token Budget Loader

**Source:** OpenAI/Cohere API pricing

**Formula:**
```
cost = (uncached_prompt × input_rate) + (cached_tokens × cached_rate) + (completion × output_rate)
```

**Config keys (via extra):**
- `input_rate`: Input token price ($/1M tokens)
- `output_rate`: Output token price ($/1M tokens)
- `cached_rate`: Cached token price ($/1M tokens)

**Usage:**
```bash
ragtune budget --type token --prompt-tokens 1024 --completion-tokens 512
ragtune budget --type token --prompt-tokens 1024 --completion-tokens 512 --cached-tokens 800
```

---

### 3. GPU Utilization Budget Loader

**Formula:**
```
cost = GPU_hourly_rate × (runtime_seconds / 3600)
```

**Config keys:**
- `gpu_type`: GPU hardware
- `runtime_s`: Runtime in seconds (via context)
- `gpu_util_pct`: GPU utilization percentage (via context)

**Usage:**
```bash
ragtune budget --type gpu_util --gpu A100-80GB
```

---

### 4. Carbon Budget Loader

**Source:** IPCC Tier 1 methodology, Ember 2025

**Formula:**
```
carbon_kg = energy_kwh × PUE × carbon_intensity / 1000
```

Where:
- `energy_kwh = power_w × time_s / 3600 / 1000`
- `PUE` = Power Usage Effectiveness (default 1.15)
- `carbon_intensity` = g CO2e per kWh (grid-specific)

**Config keys:**
- `region`: Cloud region for carbon intensity lookup
- `pue`: Power Usage Effectiveness
- `carbon_intensity_g_per_kwh`: Override regional intensity

**Regional intensities (g CO2e/kWh):**
| Region | Value | Source |
|--------|-------|--------|
| us-east | 350 | EPA eGRID 2023 |
| us-west | 200 | Oregon/California mix |
| eu-central | 280 | Germany/Netherlands mix |
| eu-north | 50 | Nordic weighted avg |
| eu-france | 45 | Nuclear-heavy grid |
| asia-east | 500 | China/Japan/Korea mix |
| asia-south | 700 | India dominates |
| australia | 525 | Australia grid |
| global-average | 450 | Ember 2025 |

**Usage:**
```bash
ragtune budget --type carbon --region eu-france
ragtune budget --type carbon --region asia-south --gpu H100-NVL-96GB
```

---

### 5. Embedding Budget Loader

**Source:** OpenAI, Cohere, Voyage AI pricing pages

**Formula:**
```
cost = tokens × price_per_token
```

**Pre-configured models:**
| Model | Price/1M tokens | Dimensions |
|-------|-----------------|------------|
| openai/text-embedding-3-small | $0.02 | 1536 |
| openai/text-embedding-3-large | $0.13 | 3072 |
| cohere/embed-v4 | $0.12 | 1024 |
| voyage/voyage-4 | $0.06 | 1024 |
| voyage/voyage-4-lite | $0.02 | 512 |

**Usage:**
```bash
ragtune budget --type embedding --prompt-tokens 20
ragtune budget --type embedding --embedding-model openai/text-embedding-3-large
```

---

### 6. Reranking Budget Loader

**Source:** Cohere, Voyage AI pricing pages

**Pricing models:**
- **Per-query (Cohere):** $X per 1,000 queries, up to 100 docs each
- **Per-token (Voyage):** $X per 1M tokens (query × docs + doc tokens)

**Pre-configured models:**
| Model | Pricing | Max Docs |
|-------|---------|----------|
| cohere/rerank-v4-pro | $2.50/1k queries | 100 |
| cohere/rerank-v4-fast | $2.00/1k queries | 100 |
| voyage/rerank-2.5 | $0.05/1M tokens | 100 |
| voyage/rerank-2.5-lite | $0.02/1M tokens | 100 |

**Usage:**
```bash
ragtune budget --type reranking --reranking-model cohere/rerank-v4-pro --queries 10 --docs 50
ragtune budget --type reranking --reranking-model voyage/rerank-2.5 --queries 10 --docs 50
```

---

## Configuration

### BudgetConfig Fields

All 34 fields are configurable via YAML, environment variables, or CLI:

| Field | Default | Description | Source |
|-------|---------|-------------|--------|
| `gpu_type` | "A100-80GB" | GPU hardware type | User config |
| `gpu_count` | 1 | Number of GPUs | User config |
| `gpu_hourly_rate` | 0.0 | GPU hourly rate ($, 0=use GPU_SPECS) | User config |
| `region` | "us-east" | Cloud region | User config |
| `electricity_cost_per_kwh` | 0.12 | Electricity price ($/kWh) | EIA 2024 |
| `carbon_intensity_g_per_kwh` | 450 | Grid carbon intensity | Ember 2025 |
| `pue` | 1.15 | Power Usage Effectiveness | Uptime Institute 2024 |
| `model_name` | "cross-encoder/..." | Model name | User config |
| `model_architecture` | "dense" | Model architecture | User config |
| `active_params_b` | 0.1 | Active parameters (billions) | User config |
| `total_params_b` | 0.1 | Total parameters (billions) | User config |
| `quantization` | "fp16" | Quantization format | User config |
| `max_batch_size` | 256 | Maximum batch size | User config |
| `latency_slo_ms` | 500 | Latency SLO (ms) | User config |
| `offered_rps` | 10.0 | Request rate (req/s) | User config |
| `tensor_parallel` | 1 | Tensor parallelism | User config |
| `pipeline_parallel` | 1 | Pipeline parallelism | User config |
| `kv_overhead_per_token_s` | 0.00014 | KV cache overhead (s/token) | arXiv 2606.11690 |
| `lam_sat_fallback` | 15.0 | Saturation knee fallback | Calibrated |
| `peak_utilization_threshold` | 0.9 | Batch size classification | Calibrated |
| `cache_saving_fraction` | 0.50 | Cache savings fraction | vLLM APC docs |
| `cache_hit_rate` | 0.0 | Statistical cache hit rate | Industry avg 35% |
| `default_tokens_per_doc` | 512 | Default tokens per document | Conservative |
| `escalation_gap_threshold` | 0.05 | Strategy escalation trigger | Calibrated |
| `gpu_power_idle_fraction` | 0.25 | GPU idle power fraction | NVIDIA power model |
| `gpu_power_active_fraction` | 0.75 | GPU active power swing | NVIDIA power model |

### YAML Configuration

```yaml
# configs/custom.yaml
gpu_type: "H100-NVL-96GB"
gpu_count: 2
pue: 1.10
model_name: "llama-3.1-8b"
offered_rps: 25.0
cache_hit_rate: 0.35
carbon_intensity_g_per_kwh: 450
```

```bash
ragtune budget --config configs/custom.yaml
```

### CLI Options

```bash
ragtune budget [OPTIONS]

Options:
  --type TEXT              Budget loader type (vllm, token, gpu_util, carbon, embedding, reranking)
  --prompt-tokens INT     Number of prompt/input tokens
  --completion-tokens INT Number of completion/output tokens
  --cached-tokens INT     Number of cached input tokens
  --gpu TEXT               GPU type (e.g., H100-NVL-96GB, A100-80GB)
  --gpu-count INT          Number of GPUs
  --gpu-rate FLOAT         GPU hourly rate override ($)
  --model TEXT             Model name (e.g., llama-3.1-8b)
  --rps FLOAT              Offered request rate (requests/sec)
  --slo INT                Latency SLO in milliseconds
  --region TEXT             Cloud region for carbon intensity
  --pue FLOAT              Power Usage Effectiveness (default 1.15)
  --cache-hit-rate FLOAT   Cache hit rate (0.0-1.0)
  --embedding-model TEXT   Embedding model (e.g., openai/text-embedding-3-small)
  --reranking-model TEXT   Reranking model (e.g., cohere/rerank-v4-pro)
  --queries INT             Number of queries (for reranking)
  --docs INT                Docs per query (for reranking)
  --config PATH             Path to YAML budget config
  --suggest                 Show optimization suggestions
  --verbose                 Show detailed breakdown
```

---

## Per-Component Cost Tracking

`BudgetResult` includes 5 per-component cost fields for pipeline visibility:

| Field | Description | Populated By |
|-------|-------------|--------------|
| `embedding_cost_usd` | Embedding API cost | EmbeddingBudgetLoader |
| `retrieval_cost_usd` | Retrieval cost | (BM25 is free, dense = embedding) |
| `reranking_cost_usd` | Reranking API cost | RerankingBudgetLoader |
| `reformulation_cost_usd` | Query reformulation cost | (via token loader) |
| `generation_cost_usd` | LLM generation cost | VLLMBudgetLoader |

---

## Cost Optimization

The optimizer analyzes `BudgetResult` and suggests improvements:

```python
from ragtune.budget.optimizer import suggest_optimizations

suggestions = suggest_optimizations(result)
for s in suggestions:
    print(f"[{s['priority']}] {s['category']}: {s['suggestion']}")
```

**Suggestion categories:**
- **caching**: Enable semantic caching (35% avg savings)
- **model_selection**: Right-size model for workload
- **batching**: Increase batch size for better utilization
- **quantization**: Use FP8 for ~2x throughput
- **parallelism**: Tensor parallelism for large models
- **latency**: Relax SLO or use faster model
- **carbon**: Run in cleaner grid region

---

## Cost History

Log budget calculations for historical analysis:

```python
from ragtune.budget.history import CostHistoryLogger

logger = CostHistoryLogger("cost_history.jsonl")
logger.log("vllm", config_dict, context_dict, result)

# Query
entries = logger.query(budget_type="vllm", since="2026-07-01")

# Summary
summary = logger.summary()
print(f"Total cost: ${summary['total_cost_usd']:.6f}")
```

---

## Cost Alerts

Monitor costs against thresholds:

```python
from ragtune.budget.alerts import check_alerts

alerts = check_alerts(result, {
    "max_cost_usd": 0.01,
    "max_carbon_kg": 0.001,
    "min_throughput_tok_s": 100,
    "min_gpu_utilization": 30.0,
})
```

---

## Controller Integration

Optional cost estimation per iteration in the RAGtuneController:

```python
from ragtune.budget.base import BudgetConfig
from ragtune.budget.loaders.vllm_budget import VLLMBudgetLoader

loader = VLLMBudgetLoader(BudgetConfig({
    "gpu_type": "H100-NVL-96GB",
    "model_name": "llama-3.1-8b",
}))

controller = RAGtuneController(
    ...,
    cost_loader=loader,
    cost_config={"prompt_tokens": 512, "completion_tokens": 256},
)

result = controller.run(query)
print(f"Total cost: ${result.final_budget_state['total_cost_usd']:.6f}")
```

---

## Verification

### Paper Accuracy

The vLLM loader matches arXiv 2606.11690 within ≤5.3%:

| λ (rps) | Our C_eff | Paper C_eff | Error |
|----------|-----------|-------------|-------|
| 1 | $7.57 | $7.60 | 0.3% |
| 5 | $1.52 | $1.51 | 0.3% |
| 10 | $0.76 | $0.80 | 5.3% |
| 25 | $0.36 | $0.37 | 3.5% |
| 50 | $0.32 | $0.32 | 0.1% |

### Source Citations

| Component | Source |
|-----------|--------|
| Throughput model | arXiv 2606.11690 (Patil, June 2026) |
| Carbon formula | IPCC Tier 1 methodology, GHG Protocol |
| Grid intensity | Ember Global Electricity Review 2025 |
| PUE values | Uptime Institute 2024 |
| GPU power model | NVIDIA GPU power management documentation |
| Embedding pricing | OpenAI, Cohere, Voyage AI pricing pages |
| Reranking pricing | Cohere, Voyage AI pricing pages |
| Token pricing | OpenAI API pricing (GPT-4o) |
| Cache savings | vLLM automatic-prefix-caching documentation |

---

## Tests

161 tests covering:
- BudgetResult arithmetic and breakdown merge
- All 6 loader types with edge cases
- Hardware specs (frozen dataclass, fallback, FP8 detection)
- Throughput model (empirical lookup, saturation, SLO enforcement)
- Factory creation (config, YAML, error handling)
- Cost optimizer suggestions
- Cost history logging
- Cost alerts
- BudgetConfig validation

```bash
python -m pytest tests/unit/budget/ -v
```
