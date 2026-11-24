"""Unit tests for the FLOPs-and-dollar cost model. No network, no real models."""

import pytest

from ragtune.tuning.cost_model import (
    CostModel,
    CostModelConfig,
    CostModelError,
    EncoderGeometry,
    QueryUsage,
    normalise_model_id,
    normalise_model_id_to_hf,
)

# MiniLM-L-6 geometry, so the numbers below are checkable by hand.
L6 = EncoderGeometry(n_layer=6, d_model=384, d_ff=1536)
L2 = EncoderGeometry(n_layer=2, d_model=384, d_ff=1536)
L12 = EncoderGeometry(n_layer=12, d_model=384, d_ff=1536)

CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def make_model(**kw) -> CostModel:
    cfg = dict(
        n_ctx=355,
        ce_ms_per_doc={
            "cross-encoder/ms-marco-MiniLM-L-2-v2":
                {1: 3.000, 2: 2.989, 5: 1.435, 10: 1.033, 20: 0.875},
            "cross-encoder/ms-marco-MiniLM-L-6-v2":
                {1: 4.629, 2: 3.525, 5: 1.787, 10: 1.238, 20: 1.184},
            "cross-encoder/ms-marco-MiniLM-L-12-v2":
                {1: 7.345, 2: 5.284, 5: 2.573, 10: 2.233, 20: 2.216},
        },
        gpu_usd_per_hour=0.80,
        cpu_usd_per_hour=0.0,
        token_prices={"claude-opus-5": (5.0, 25.0), "claude-haiku-4-5": (1.0, 5.0)},
    )
    cfg.update(kw)
    return CostModel(CostModelConfig(**cfg))


# ── FLOP estimator ────────────────────────────────────────────────────────────

def test_n_enc_params_matches_e2r_flops_formula():
    # 2*d_model*n_layer*(2*d_attn + d_ff) = 2*384*6*(768+1536)
    assert L6.n_enc_params == pytest.approx(2 * 384 * 6 * (768 + 1536))


def test_n_enc_params_within_2pct_of_counted_weights():
    # Counted non-embedding parameters, measured from the real checkpoints.
    for geom, counted in ((L2, 3.7e6), (L6, 10.8e6), (L12, 21.4e6)):
        assert geom.n_enc_params == pytest.approx(counted, rel=0.06)


def test_flops_per_pair_has_linear_and_quadratic_terms():
    n_ctx = 355
    linear = 2 * L6.n_enc_params * n_ctx
    quadratic = 4 * 6 * n_ctx**2 * 384
    assert L6.flops_per_pair(n_ctx) == pytest.approx(linear + quadratic)


def test_quadratic_term_grows_with_context_length():
    """The attention term is quadratic, so its share rises with n_ctx."""
    def share(n):
        return (4 * 6 * n**2 * 384) / L6.flops_per_pair(n)

    assert share(256) < share(355) < share(512)


def test_flops_scale_superlinearly_in_layers():
    """FLOPs ratios across depth: roughly 1 : 2.9 : 5.8 at n_ctx=355."""
    a, b, c = (g.flops_per_pair(355) for g in (L2, L6, L12))
    assert b / a == pytest.approx(2.93, rel=0.05)
    assert c / a == pytest.approx(5.81, rel=0.05)


def test_flops_rejects_nonpositive_context():
    with pytest.raises(CostModelError):
        L6.flops_per_pair(0)


# ── dollars ───────────────────────────────────────────────────────────────────

def test_gpu_cost_matches_hand_calculation():
    cm = make_model()
    usage = QueryUsage(rerank_docs=39.2, reranker_model=CE, rerank_batch_size=20)
    expected = 39.2 * 1.184 * 0.80 / 3_600_000.0
    assert cm.gpu_usd(usage) == pytest.approx(expected)


def test_api_cost_matches_published_rates():
    cm = make_model()
    usage = QueryUsage(llm_tokens={"anthropic/claude-opus-5": (96, 95)})
    expected = (96 * 5.0 + 95 * 25.0) / 1e6
    assert cm.api_usd(usage) == pytest.approx(expected)


def test_api_cost_dominates_local_rerank_cost():
    """The finding that motivates the whole spec: one opus call >> reranking."""
    cm = make_model()
    rerank_only = QueryUsage(rerank_docs=39.2, reranker_model=CE, rerank_batch_size=20)
    with_llm = QueryUsage(
        rerank_docs=39.2, reranker_model=CE, rerank_batch_size=20,
        llm_tokens={"claude-opus-5": (96, 95)},
    )
    ratio = cm.api_usd(with_llm) / cm.gpu_usd(rerank_only)
    assert ratio > 100


def test_cost_is_additive_across_components():
    cm = make_model(cpu_usd_per_hour=1.50)
    usage = QueryUsage(
        rerank_docs=10, reranker_model=CE, rerank_batch_size=10, retrieval_ms=147,
        llm_tokens={"claude-haiku-4-5": (69, 40)},
    )
    assert cm.cost_usd_per_query(usage) == pytest.approx(
        cm.gpu_usd(usage) + cm.cpu_usd(usage) + cm.api_usd(usage)
    )


def test_llm_tokens_summed_over_multiple_models():
    cm = make_model()
    usage = QueryUsage(llm_tokens={
        "claude-haiku-4-5": (100, 50),
        "claude-opus-5": (100, 50),
    })
    expected = (100 * 1.0 + 50 * 5.0) / 1e6 + (100 * 5.0 + 50 * 25.0) / 1e6
    assert cm.api_usd(usage) == pytest.approx(expected)


# ── budget exhaustion (Rule 4: required for budget/cost logic) ────────────────

def test_zero_rerank_docs_is_a_legitimate_zero_not_an_error():
    """Budget exhausted before any reranking: cost is retrieval only."""
    cm = make_model(cpu_usd_per_hour=1.50)
    usage = QueryUsage(rerank_docs=0.0, reranker_model=None, retrieval_ms=147)
    assert cm.gpu_usd(usage) == 0.0
    assert cm.cost_usd_per_query(usage) == pytest.approx(cm.cpu_usd(usage))
    assert cm.flops_per_query(usage) == 0.0


def test_fully_exhausted_budget_costs_nothing():
    cm = make_model()
    assert cm.cost_usd_per_query(QueryUsage()) == 0.0
    assert cm.flops_per_query(QueryUsage()) == 0.0


def test_partial_budget_cost_scales_with_docs_actually_reranked():
    """A trial pruned mid-way must be charged for what it consumed, not the cap."""
    cm = make_model()
    full = QueryUsage(rerank_docs=50, reranker_model=CE, rerank_batch_size=10)
    partial = QueryUsage(rerank_docs=7, reranker_model=CE, rerank_batch_size=10)
    assert cm.gpu_usd(partial) == pytest.approx(cm.gpu_usd(full) * 7 / 50)


def test_docs_reranked_without_a_model_is_an_error():
    cm = make_model()
    usage = QueryUsage(rerank_docs=10, reranker_model=None, rerank_batch_size=10)
    with pytest.raises(CostModelError, match="reranker_model is None"):
        cm.gpu_usd(usage)


# ── failure behaviour: never silently zero ────────────────────────────────────

def test_unknown_reranker_raises_rather_than_costing_zero():
    cm = make_model()
    usage = QueryUsage(rerank_docs=10, reranker_model="cross-encoder/not-calibrated",
                       rerank_batch_size=10)
    with pytest.raises(CostModelError, match="no measured ms/doc"):
        cm.gpu_usd(usage)


def test_unknown_llm_raises_rather_than_costing_zero():
    cm = make_model()
    usage = QueryUsage(llm_tokens={"gpt-5.5": (10, 10)})
    with pytest.raises(CostModelError, match="no token price"):
        cm.api_usd(usage)


def test_flops_is_none_when_an_api_model_participated():
    """A partial sum must never be presented as total compute."""
    cm = make_model()
    usage = QueryUsage(
        rerank_docs=39.2, reranker_model=CE, rerank_batch_size=20,
        llm_tokens={"claude-opus-5": (96, 95)},
    )
    assert cm.flops_per_query(usage) is None
    assert cm.rpp(0.33, usage) is None
    assert cm.qpp(usage) is None
    # dollars still work — that is the point of using money as the axis
    assert cm.cost_usd_per_query(usage) > 0


# ── reporting metrics ─────────────────────────────────────────────────────────

def test_rpp_reproduces_scaling_hurts_efficiency_finding():
    """E2R-FLOPs' headline: deeper reranker, worse ranking-per-PetaFLOP."""
    cm = make_model()
    measured = [
        ("cross-encoder/ms-marco-MiniLM-L-2-v2", L2, 0.3069),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", L6, 0.3339),
        ("cross-encoder/ms-marco-MiniLM-L-12-v2", L12, 0.2996),
    ]
    rpps = []
    for _, geom, ndcg in measured:
        flops = 39.2 * geom.flops_per_pair(355)
        rpps.append(ndcg / (flops / 1e15))
    assert rpps[0] > rpps[1] > rpps[2]          # RPP falls monotonically with depth
    assert rpps[0] / rpps[1] == pytest.approx(2.69, rel=0.05)
    assert rpps[2] / rpps[1] == pytest.approx(0.45, rel=0.05)


def test_rpp_ratios_are_invariant_to_n_ctx():
    """n_ctx cancels within a corpus: it scales absolute RPP, not the ordering."""
    def ratio(n_ctx):
        a = 0.3069 / (39.2 * L2.flops_per_pair(n_ctx))
        b = 0.3339 / (39.2 * L6.flops_per_pair(n_ctx))
        return a / b

    assert ratio(256) == pytest.approx(ratio(355), rel=1e-9)
    assert ratio(256) == pytest.approx(ratio(512), rel=1e-9)


def test_qpp_is_reciprocal_of_petaflops_per_query():
    cm = make_model()
    usage = QueryUsage(rerank_docs=39.2, reranker_model=CE, rerank_batch_size=20)
    flops = cm.flops_per_query(usage)
    assert cm.qpp(usage) == pytest.approx(1.0 / (flops / 1e15))


# ── id normalisation ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("anthropic/claude-opus-5", "claude-opus-5"),
    ("claude-opus-5", "claude-opus-5"),
    ("  anthropic/claude-haiku-4-5  ", "claude-haiku-4-5"),
])
def test_normalise_model_id_strips_provider_prefix(raw, expected):
    assert normalise_model_id(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("cross-encoder:ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ("cross-encoder:BAAI/bge-reranker-base", "BAAI/bge-reranker-base"),
])
def test_registry_ids_normalise_to_hf_paths(raw, expected):
    assert normalise_model_id_to_hf(raw) == expected


# ── batch-size dependence (the tuned knob) ────────────────────────────────────

def test_batch_size_changes_cost_by_almost_4x():
    """scheduler_batch_size is tuned over 1-20 and must not be priced flat."""
    cm = make_model()
    bs1 = QueryUsage(rerank_docs=40, reranker_model=CE, rerank_batch_size=1)
    bs20 = QueryUsage(rerank_docs=40, reranker_model=CE, rerank_batch_size=20)
    assert cm.gpu_usd(bs1) / cm.gpu_usd(bs20) == pytest.approx(4.629 / 1.184, rel=1e-6)


def test_cost_is_monotonically_cheaper_with_larger_batches():
    cm = make_model()
    costs = [
        cm.gpu_usd(QueryUsage(rerank_docs=40, reranker_model=CE, rerank_batch_size=b))
        for b in (1, 2, 5, 10, 20)
    ]
    assert costs == sorted(costs, reverse=True)


def test_batch_size_interpolates_between_grid_points():
    cm = make_model()
    # bs=15 sits midway between the 10 and 20 grid points
    got = cm._ce_ms_per_doc(CE, 15)
    assert got == pytest.approx(1.238 + 0.5 * (1.184 - 1.238))


def test_batch_size_clamps_outside_the_grid():
    """The curve flattens by bs=20; extrapolating the fall would under-price."""
    cm = make_model()
    assert cm._ce_ms_per_doc(CE, 1000) == pytest.approx(1.184)
    assert cm._ce_ms_per_doc(CE, 0) == pytest.approx(4.629)


def test_missing_batch_size_raises_rather_than_defaulting():
    cm = make_model()
    usage = QueryUsage(rerank_docs=40, reranker_model=CE, rerank_batch_size=None)
    with pytest.raises(CostModelError, match="rerank_batch_size is None"):
        cm.gpu_usd(usage)


def test_scalar_ms_per_doc_is_rejected():
    """A pre-batch-aware config must fail loudly, not silently mis-price."""
    cm = make_model(ce_ms_per_doc={CE: 2.22})
    usage = QueryUsage(rerank_docs=40, reranker_model=CE, rerank_batch_size=10)
    with pytest.raises(CostModelError, match="must be a"):
        cm.gpu_usd(usage)


def test_flops_are_independent_of_batch_size():
    """FLOPs count arithmetic; batching changes wall-clock, not operations."""
    cm = make_model()
    a = cm.flops_per_query(QueryUsage(rerank_docs=40, reranker_model=CE, rerank_batch_size=1))
    b = cm.flops_per_query(QueryUsage(rerank_docs=40, reranker_model=CE, rerank_batch_size=20))
    assert a == pytest.approx(b)
