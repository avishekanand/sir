"""
FLOPs-and-dollar cost model for tuning trials.

The FLOP estimator follows Peng et al., "Efficiency-Effectiveness Reranking
FLOPs for LLM-based Rerankers" (arXiv:2507.06223).  For an encoder-only
cross-encoder nothing is generated, so n_opt = 0 and only the context term
survives:

    C_pair(n_ctx) = 2*N_enc*n_ctx + 4*n_layer*n_ctx^2*d_attn
    N_enc         = 2*d_model*n_layer*(2*d_attn + d_ff)

Dollars rather than FLOPs are the Pareto objective because FLOPs cannot be
computed for a closed API model — the parameter count is unpublished — so the
vendor's price is the only available proxy for their compute.  See
specs/spec-cost-model.md for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Tuple

MS_PER_HOUR = 3_600_000.0
# sentence-transformers CrossEncoder.predict() default batch_size. A rerank call
# of N documents is executed in batches of at most this, so the effective GPU
# batch is min(N, SBERT_BATCH_CAP).
SBERT_BATCH_CAP = 32
FLOPS_PER_PETAFLOP = 1e15

# USD per million tokens, (input, output).  Keyed by bare model id; provider
# prefixes such as "anthropic/" are stripped before lookup.
TOKEN_PRICES_USD_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
}


class CostModelError(ValueError):
    """Raised when cost cannot be computed. Never silently return zero."""


def normalise_model_id(model: str) -> str:
    """Strip a litellm provider prefix: 'anthropic/claude-opus-5' -> 'claude-opus-5'."""
    return model.split("/")[-1].strip()


# ── FLOPs ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EncoderGeometry:
    """Transformer encoder shape, enough to estimate forward-pass FLOPs."""

    n_layer: int
    d_model: int
    d_ff: int

    @property
    def d_attn(self) -> int:
        return self.d_model

    @property
    def n_enc_params(self) -> float:
        """E2R-FLOPs parameter approximation (excludes embeddings)."""
        return 2.0 * self.d_model * self.n_layer * (2.0 * self.d_attn + self.d_ff)

    def flops_per_pair(self, n_ctx: int) -> float:
        """FLOPs for one (query, document) forward pass at this context length."""
        if n_ctx <= 0:
            raise CostModelError(f"n_ctx must be positive, got {n_ctx}")
        linear = 2.0 * self.n_enc_params * n_ctx
        quadratic = 4.0 * self.n_layer * (n_ctx ** 2) * self.d_attn
        return linear + quadratic


@lru_cache(maxsize=32)
def encoder_geometry(hf_name: str) -> EncoderGeometry:
    """Read geometry from the HF config. Downloads config.json only, not weights."""
    try:
        from transformers import AutoConfig
    except ImportError as exc:  # pragma: no cover - transformers is a hard dep
        raise CostModelError("transformers is required to read model geometry") from exc
    try:
        cfg = AutoConfig.from_pretrained(hf_name)
    except Exception as exc:
        raise CostModelError(f"cannot read geometry for {hf_name!r}: {exc}") from exc
    return EncoderGeometry(
        n_layer=int(cfg.num_hidden_layers),
        d_model=int(cfg.hidden_size),
        d_ff=int(cfg.intermediate_size),
    )


# ── Usage record ──────────────────────────────────────────────────────────────

@dataclass
class QueryUsage:
    """What one query actually consumed. Produced from a CostTracker snapshot."""

    rerank_docs: float = 0.0
    reranker_model: Optional[str] = None
    # Effective GPU batch = scheduler_batch_size: rerank() hands the scheduler's
    # doc list straight to predict(), so this is a TUNED knob (range 1-20) and
    # swings per-doc cost ~3.9x. Pricing it as a constant would reintroduce
    # exactly the blind spot this model exists to remove.
    rerank_batch_size: Optional[int] = None
    retrieval_ms: float = 0.0
    # model id -> (input_tokens, output_tokens), summed over calls for that model
    llm_tokens: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # Per-call record [(model, n_docs), ...] from CostTracker.rerank_calls. The
    # scheduler issues several calls of differing size per query and each one's
    # size sets its own per-document rate, so when this is present it takes
    # precedence over the scalar fields above.
    rerank_calls: Optional[List[Tuple[str, int]]] = None

    @classmethod
    def from_budget_state(cls, state: Mapping[str, object], retrieval_ms: float = 0.0):
        """Build from ControllerOutput.final_budget_state (CostTracker.snapshot())."""
        calls = [(str(m), int(n)) for m, n in (state.get("rerank_calls") or [])]
        tokens = {
            str(m): (int(v[0]), int(v[1]))
            for m, v in (state.get("llm_usage") or {}).items()
        }
        # `consumed["rerank_docs"]` counts DENIED attempts too — CostTracker
        # .try_consume accumulates before checking the limit, so a config that
        # asks for more reranking than its budget allows inflates that counter
        # without the reranker ever running. rerank_calls records only calls
        # actually issued, so it is what we charge for.
        docs_actually_reranked = float(sum(n for _, n in calls))
        return cls(
            rerank_docs=docs_actually_reranked,
            reranker_model=calls[0][0] if calls else None,
            retrieval_ms=retrieval_ms,
            llm_tokens=tokens,
            rerank_calls=calls or None,
        )

    @property
    def used_api_model(self) -> bool:
        return bool(self.llm_tokens)


# ── Cost model ────────────────────────────────────────────────────────────────

@dataclass
class CostModelConfig:
    """Calibrated constants. Versioned in config/cost_model.yaml, not hardcoded."""

    n_ctx: int
    ce_ms_per_doc: Mapping[str, float]
    gpu_usd_per_hour: float = 0.80
    cpu_usd_per_hour: float = 0.0
    token_prices: Mapping[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(TOKEN_PRICES_USD_PER_MTOK)
    )


class CostModel:
    """Turns a QueryUsage into FLOPs and dollars."""

    def __init__(self, config: CostModelConfig):
        self.config = config

    # -- components ----------------------------------------------------------

    def _ce_ms_per_doc(self, model: str, batch_size: int) -> float:
        """
        Measured milliseconds per reranked document at this batch size.

        Interpolates piecewise-linearly over the calibrated batch-size grid and
        clamps outside it, because the measured curve flattens by bs=10-20 and
        extrapolating a falling trend past the grid would under-price.
        """
        key = normalise_model_id(model)
        table = self.config.ce_ms_per_doc
        grid = None
        for candidate in (model, key):
            if candidate in table:
                grid = table[candidate]
                break
        if grid is None:
            raise CostModelError(
                f"no measured ms/doc for reranker {model!r}; "
                f"calibrate it (scripts/calibrate_cost_model.py) or add it to "
                f"cost_model.yaml. Known: {sorted(table)}"
            )
        if isinstance(grid, (int, float)):
            raise CostModelError(
                f"ce_ms_per_doc[{model!r}] is a scalar; it must be a "
                f"{{batch_size: ms}} grid. Re-run scripts/calibrate_cost_model.py"
            )

        points = sorted((int(b), float(v)) for b, v in grid.items())
        if batch_size <= points[0][0]:
            return points[0][1]
        if batch_size >= points[-1][0]:
            return points[-1][1]
        for (b0, v0), (b1, v1) in zip(points, points[1:]):
            if b0 <= batch_size <= b1:
                frac = (batch_size - b0) / (b1 - b0)
                return v0 + frac * (v1 - v0)
        raise CostModelError(f"could not interpolate batch_size={batch_size}")

    def _token_price(self, model: str) -> Tuple[float, float]:
        key = normalise_model_id(model)
        table = self.config.token_prices
        for candidate in (model, key):
            if candidate in table:
                return tuple(table[candidate])  # type: ignore[return-value]
        raise CostModelError(
            f"no token price for LLM {model!r}; add it to cost_model.yaml. "
            f"Known: {sorted(table)}"
        )

    def gpu_usd(self, usage: QueryUsage) -> float:
        if usage.rerank_calls:
            ms = sum(
                n * self._ce_ms_per_doc(model, min(n, SBERT_BATCH_CAP))
                for model, n in usage.rerank_calls
            )
            return ms * self.config.gpu_usd_per_hour / MS_PER_HOUR
        if usage.rerank_docs <= 0:
            return 0.0  # budget exhausted before reranking — a legitimate zero
        if usage.reranker_model is None:
            raise CostModelError(
                f"{usage.rerank_docs} docs reranked but reranker_model is None"
            )
        if usage.rerank_batch_size is None:
            raise CostModelError(
                f"{usage.rerank_docs} docs reranked but rerank_batch_size is None; "
                "per-doc cost varies ~3.9x across the tuned 1-20 range, so it "
                "cannot be defaulted"
            )
        ms = usage.rerank_docs * self._ce_ms_per_doc(
            usage.reranker_model, usage.rerank_batch_size
        )
        return ms * self.config.gpu_usd_per_hour / MS_PER_HOUR

    def cpu_usd(self, usage: QueryUsage) -> float:
        return usage.retrieval_ms * self.config.cpu_usd_per_hour / MS_PER_HOUR

    def api_usd(self, usage: QueryUsage) -> float:
        total = 0.0
        for model, (tok_in, tok_out) in usage.llm_tokens.items():
            p_in, p_out = self._token_price(model)
            total += (tok_in * p_in + tok_out * p_out) / 1e6
        return total

    # -- public API ----------------------------------------------------------

    def cost_usd_per_query(self, usage: QueryUsage) -> float:
        """Objective 2 (minimize). Spans local compute and API spend."""
        return self.gpu_usd(usage) + self.cpu_usd(usage) + self.api_usd(usage)

    def flops_per_query(self, usage: QueryUsage) -> Optional[float]:
        """
        Local forward-pass FLOPs. None when an API model participated, because a
        closed model's FLOPs are not computable and a partial sum must never be
        presented as total compute.
        """
        if usage.used_api_model:
            return None
        if usage.rerank_calls:
            # Batch size changes wall-clock, not arithmetic, so FLOPs sum plainly.
            return sum(
                n * encoder_geometry(
                    normalise_model_id_to_hf(model)
                ).flops_per_pair(self.config.n_ctx)
                for model, n in usage.rerank_calls
            )
        if usage.rerank_docs <= 0:
            return 0.0
        if usage.reranker_model is None:
            raise CostModelError(
                f"{usage.rerank_docs} docs reranked but reranker_model is None"
            )
        geom = encoder_geometry(normalise_model_id_to_hf(usage.reranker_model))
        return usage.rerank_docs * geom.flops_per_pair(self.config.n_ctx)

    # -- reporting metrics (E2R-FLOPs section 4) -----------------------------

    def rpp(self, ndcg: float, usage: QueryUsage) -> Optional[float]:
        """Ranking metric per PetaFLOP. None when FLOPs are not computable."""
        flops = self.flops_per_query(usage)
        if not flops:
            return None
        return ndcg / (flops / FLOPS_PER_PETAFLOP)

    def qpp(self, usage: QueryUsage) -> Optional[float]:
        """Queries per PetaFLOP. None when FLOPs are not computable."""
        flops = self.flops_per_query(usage)
        if not flops:
            return None
        return 1.0 / (flops / FLOPS_PER_PETAFLOP)


def normalise_model_id_to_hf(model: str) -> str:
    """
    Reranker ids reach us either as a bare HF repo path
    ('cross-encoder/ms-marco-MiniLM-L-6-v2') or prefixed by the registry
    ('cross-encoder:ms-marco-MiniLM-L-6-v2'). Normalise to the HF form.
    """
    if ":" in model:
        family, _, name = model.partition(":")
        return name if "/" in name else f"{family}/{name}"
    return model


def load_cost_model(path: Optional[str] = None, dataset: Optional[str] = None) -> CostModel:
    """Build a CostModel from config/cost_model.yaml."""
    import pathlib

    import yaml

    if path is None:
        path = str(pathlib.Path(__file__).resolve().parents[1] / "config" / "cost_model.yaml")
    data = yaml.safe_load(pathlib.Path(path).read_text()) or {}

    n_ctx_table = data.get("n_ctx", {})
    if dataset is not None and dataset in n_ctx_table:
        n_ctx = int(n_ctx_table[dataset])
    elif "default" in n_ctx_table:
        n_ctx = int(n_ctx_table["default"])
    else:
        raise CostModelError(
            f"no n_ctx for dataset {dataset!r} and no default in {path}; "
            "run scripts/calibrate_cost_model.py"
        )

    prices = {k: tuple(v) for k, v in (data.get("token_prices") or {}).items()}
    return CostModel(
        CostModelConfig(
            n_ctx=n_ctx,
            ce_ms_per_doc=dict(data.get("ce_ms_per_doc") or {}),
            gpu_usd_per_hour=float(data.get("gpu_usd_per_hour", 0.80)),
            cpu_usd_per_hour=float(data.get("cpu_usd_per_hour", 0.0)),
            token_prices=prices or dict(TOKEN_PRICES_USD_PER_MTOK),
        )
    )
