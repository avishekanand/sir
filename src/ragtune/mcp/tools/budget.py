"""Budget tools: cost, energy and carbon estimation (ragtune.budget)."""

import dataclasses
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

from ragtune.budget import BudgetLoaderFactory
from ragtune.budget.base import BudgetConfig
from ragtune.budget.main import format_report
from ragtune.mcp._common import DESTRUCTIVE, READ_ONLY, WRITES, add_tools, to_jsonable
from ragtune.mcp.state import ServerState

# What each loader reads from `context` and from `config.extra`.
LOADER_INPUTS = {
    "vllm": {"context": ["prompt_tokens", "completion_tokens", "cached_tokens"], "extra": []},
    "token": {"context": ["prompt_tokens", "completion_tokens", "cached_tokens"],
              "extra": ["input_rate", "output_rate", "cached_rate"]},
    "gpu_util": {"context": ["runtime_s", "gpu_util_pct", "prompt_tokens", "completion_tokens"], "extra": []},
    "carbon": {"context": ["runtime_s", "gpu_util_pct", "prompt_tokens", "completion_tokens"], "extra": []},
    "embedding": {"context": ["tokens", "prompt_tokens", "completion_tokens"],
                  "extra": ["embedding_model", "embedding_price_per_million"]},
    "reranking": {"context": ["queries", "docs_per_query", "query_tokens", "doc_tokens_per_doc"],
                  "extra": ["reranking_model"]},
}
RESULT_FIELDS = ("cost_usd", "cost_per_million_tokens", "energy_kwh", "carbon_kg", "total_tokens",
                 "throughput_tok_s", "gpu_utilization", "latency_slo_met")


def _reference_tables() -> Dict[str, Any]:
    from ragtune.budget import hardware, throughput
    from ragtune.budget.loaders import carbon_budget, embedding_budget, reranking_budget, token_budget

    return {
        "gpus": {name: dataclasses.asdict(spec) for name, spec in hardware.GPU_SPECS.items()},
        "model_profiles": {name: dict(zip(("total_params_b", "active_params_b", "architecture"), p))
                           for name, p in throughput.MODEL_PROFILES.items()},
        "calibrated_theta_max": {"|".join(k): v for k, v in throughput.CALIBRATED_THETA_MAX.items()},
        "saturation_knee": throughput.LAM_SAT_TABLE,
        "quantization_bytes_per_param": throughput.QUANT_FACTORS,
        "regional_carbon_intensity": carbon_budget.REGIONAL_INTENSITY,
        "embedding_rates": embedding_budget.EMBEDDING_RATES,
        "reranking_rates": reranking_budget.RERANKING_RATES,
        "token_rates": token_budget.DEFAULT_RATES,
        "budget_config_defaults": BudgetConfig({}).to_dict(),
        "loaders": {key: {"doc": (cls.__doc__ or "").strip().split("\n")[0], **LOADER_INPUTS.get(key, {})}
                    for key, cls in BudgetLoaderFactory._REGISTRY.items()},
    }


def register(mcp: MCPServer, state: ServerState) -> None:
    def make_loader(budget_type: str, config: Optional[Dict[str, Any]], config_path: Optional[str]):
        path = str(state.resolve(config_path, must_exist=True)) if config_path else None
        return BudgetLoaderFactory.create(budget_type, config=config, config_path=path)

    def estimate_cost(
        budget_type: str = "vllm",
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggest: bool = False,
        thresholds: Optional[Dict[str, float]] = None,
        log_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """`ragtune budget`: cost, energy and carbon of one operation.

        budget_type: vllm, token, gpu_util, carbon, embedding, reranking.
        config holds BudgetConfig fields (gpu_type, model_name, offered_rps,
        region, ...; loader-specific keys under "extra") and overrides
        config_path. context holds per-request inputs; budget_reference lists
        both per loader. suggest adds optimization tips; thresholds (e.g.
        {"max_cost_usd": 0.01}) adds alerts; log_to appends to a JSONL history.
        """
        loader = make_loader(budget_type, config, config_path)
        ctx = dict(context or {})
        result = loader.calculate(ctx)
        out: Dict[str, Any] = {"budget_type": budget_type, "result": to_jsonable(result),
                               "report": format_report(result, budget_type)}
        config_errors = loader.config.validate()
        if config_errors:
            out["config_errors"] = config_errors
        if suggest:
            from ragtune.budget.optimizer import suggest_optimizations

            out["suggestions"] = suggest_optimizations(result, loader.config.to_dict())
        if thresholds is not None:
            from ragtune.budget.alerts import check_alerts

            out["alerts"] = check_alerts(result, thresholds)
        if log_to:
            from ragtune.budget.history import CostHistoryLogger

            target = state.resolve(log_to)
            target.parent.mkdir(parents=True, exist_ok=True)
            CostHistoryLogger(str(target)).log(budget_type, to_jsonable(loader.config.to_dict()), ctx, result)
            out["logged_to"] = state.relative(target)
        return out

    def compare_costs(
        variants: List[Dict[str, Any]],
        budget_type: str = "vllm",
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        sort_by: str = "cost_usd",
    ) -> Dict[str, Any]:
        """Sweep configurations, e.g. GPUs, regions or request rates, and rank them.

        Each variant is {"label"?, "config"?: overrides, "context"?: overrides}
        applied on top of config / context. Example: variants=[{"config":
        {"gpu_type": "H100-NVL-96GB"}}, {"config": {"gpu_type": "L4-24GB"}}].
        """
        if sort_by not in RESULT_FIELDS:
            raise ValueError(f"sort_by must be one of {list(RESULT_FIELDS)}")
        rows = []
        for i, variant in enumerate(variants):
            unknown = set(variant) - {"label", "config", "context"}
            if unknown:
                raise ValueError(f"variant {i} has unknown keys {sorted(unknown)}; use label, config, context")
            merged = {**(config or {}), **variant.get("config", {})}
            result = make_loader(budget_type, merged or None, config_path).calculate(
                {**(context or {}), **variant.get("context", {})})
            rows.append({"label": variant.get("label", f"variant_{i}"),
                         **{f: getattr(result, f) for f in RESULT_FIELDS}})
        rows.sort(key=lambda r: r[sort_by])
        return {"budget_type": budget_type, "sorted_by": sort_by, "rows": rows}

    def validate_budget_config(config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Check BudgetConfig values and flag keys BudgetConfig would silently ignore."""
        loader = make_loader("vllm", config, config_path)
        if config_path:
            import yaml

            config = {**(yaml.safe_load(state.resolve(config_path).read_text()) or {}), **(config or {})}
        known = set(BudgetConfig({}).to_dict())
        errors = loader.config.validate()
        return {"valid": not errors, "errors": errors,
                "ignored_keys": sorted(set(config or {}) - known),
                "resolved": to_jsonable(loader.config.to_dict())}

    def budget_reference(table: Optional[str] = None) -> Dict[str, Any]:
        """Reference data behind the estimators: GPU specs, model profiles, empirical
        throughput, saturation knees, quantization, regional carbon intensity,
        embedding/reranking/token pricing, BudgetConfig defaults, and each loader's inputs."""
        tables = _reference_tables()
        if table is None:
            return to_jsonable(tables)
        if table not in tables:
            raise ValueError(f"Unknown table {table!r}. Available: {sorted(tables)}")
        return {table: to_jsonable(tables[table])}

    def estimate_hardware(
        gpu_type: str = "A100-80GB",
        gpu_count: int = 1,
        utilization: float = 0.5,
        runtime_s: float = 1.0,
        pue: float = 1.15,
        carbon_intensity_g_per_kwh: Optional[float] = None,
        region: Optional[str] = None,
        idle_fraction: float = 0.25,
        active_fraction: float = 0.75,
        cpu_tdp_w: int = 0,
        cpu_cores: float = 1,
        cpu_total_cores: int = 0,
        cpu_utilization: float = 0.5,
        memory_power_w: float = 0.0,
        network_power_w: float = 0.0,
    ) -> Dict[str, Any]:
        """Power draw, energy and carbon for a GPU (and optional CPU) workload.

        Carbon intensity comes from carbon_intensity_g_per_kwh, else region
        (see budget_reference table='regional_carbon_intensity'), else the
        global average.
        """
        from ragtune.budget import hardware
        from ragtune.budget.loaders.carbon_budget import REGIONAL_INTENSITY

        if region is not None and region not in REGIONAL_INTENSITY:
            raise ValueError(f"Unknown region {region!r}. Available: {sorted(REGIONAL_INTENSITY)}")
        intensity = (carbon_intensity_g_per_kwh if carbon_intensity_g_per_kwh is not None
                     else REGIONAL_INTENSITY[region or "global-average"])
        gpu_w = hardware.estimate_gpu_power(gpu_type, gpu_count, utilization, idle_fraction, active_fraction)
        cpu_w = hardware.estimate_cpu_power(cpu_tdp_w, cpu_cores, utilization=cpu_utilization,
                                            total_cores=cpu_total_cores)
        total_w = hardware.estimate_total_system_power(gpu_w, cpu_w, memory_power_w, network_power_w)
        energy = hardware.estimate_energy_kwh(total_w, runtime_s, pue)
        return {"gpu": dataclasses.asdict(hardware.get_gpu_spec(gpu_type)),
                "gpu_known": gpu_type in hardware.GPU_SPECS,
                "gpu_power_w": gpu_w, "cpu_power_w": cpu_w, "total_power_w": total_w,
                "energy_kwh": energy, "carbon_intensity_g_per_kwh": intensity,
                "carbon_kg": hardware.estimate_carbon_kg(energy, intensity)}

    def estimate_throughput(
        gpu_type: str = "A100-80GB",
        model_name: str = "llama-3.1-8b",
        quantization: str = "fp16",
        offered_rps: float = 10.0,
        latency_slo_ms: int = 500,
        output_tokens: int = 256,
        total_params_b: float = 0.1,
        active_params_b: float = 0.1,
        architecture: str = "dense",
        tensor_parallel: int = 1,
        max_batch_size: int = 256,
        kv_overhead_per_token_s: float = 0.00014,
        lam_sat_fallback: float = 15.0,
        peak_utilization_threshold: float = 0.9,
    ) -> Dict[str, Any]:
        """Peak and achieved inference throughput (arXiv 2606.11690 model) and VRAM fit.

        Known model names (budget_reference table='model_profiles') override
        the *_params_b / architecture arguments.
        """
        from ragtune.budget import hardware, throughput

        total_b, active_b, arch = throughput.get_model_profile(model_name, total_params_b, active_params_b, architecture)
        peak = throughput.estimate_peak_throughput(gpu_type, model_name, quantization, total_b, active_b, arch,
                                                   tensor_parallel, max_batch_size, kv_overhead_per_token_s)
        achieved, batch = throughput.estimate_actual_throughput(
            gpu_type, model_name, quantization, offered_rps, latency_slo_ms, output_tokens, total_b, active_b,
            arch, tensor_parallel, max_batch_size, kv_overhead_per_token_s, lam_sat_fallback,
            peak_utilization_threshold)
        vram = throughput.estimate_weight_vram(total_b, quantization)
        return {"model_profile": {"total_params_b": total_b, "active_params_b": active_b, "architecture": arch},
                "peak_source": ("empirical" if (gpu_type, model_name, quantization) in throughput.CALIBRATED_THETA_MAX
                                else "analytical"),
                "peak_tps": peak, "achieved_tps": achieved, "achieved_batch": batch,
                "utilization": achieved / peak if peak else 0.0,
                "saturation_knee_rps": throughput.get_saturation_knee(active_b, fallback=lam_sat_fallback),
                "weight_vram_gb": vram,
                "fits_in_vram": vram <= hardware.get_gpu_spec(gpu_type).vram_gb * tensor_parallel}

    def cost_history(
        path: str = "cost_history.jsonl",
        budget_type: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
        summary: bool = False,
    ) -> Dict[str, Any]:
        """Read a cost-history JSONL written by estimate_cost(log_to=...): entries, or totals with summary=True.

        since is an ISO timestamp, e.g. "2026-10-01".
        """
        from ragtune.budget.history import CostHistoryLogger

        logger = CostHistoryLogger(str(state.resolve(path)))
        if summary:
            return {"summary": logger.summary(budget_type=budget_type)}
        return {"entries": logger.query(budget_type=budget_type, since=since, limit=limit)}

    def clear_cost_history(path: str = "cost_history.jsonl") -> Dict[str, Any]:
        """Delete a cost-history JSONL file."""
        from ragtune.budget.history import CostHistoryLogger

        target = state.resolve(path, must_exist=True)
        CostHistoryLogger(str(target)).clear()
        return {"cleared": state.relative(target)}

    add_tools(mcp, READ_ONLY, compare_costs, validate_budget_config, budget_reference,
              estimate_hardware, estimate_throughput, cost_history)
    add_tools(mcp, WRITES, estimate_cost)
    add_tools(mcp, DESTRUCTIVE, clear_cost_history)
