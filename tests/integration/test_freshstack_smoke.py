"""
Integration smoke test for ``scripts/benchmark_freshstack.py``.

Gives reviewers a stable target (Rule 04) and pins the three review fixes that
are easy to silently regress:

  B1 — metrics are read by exact key, never by substring match with a silent
       fallback to an arbitrary value.
  B2 — ``rerank_docs`` is the only binding budget dimension; tokens and
       latency sit at a ceiling so they cannot truncate a scenario.
  B4 — every scenario is evaluated at the same depth, so Recall@50 compares
       like with like against the no-rerank baseline.

The benchmark is a standalone script rather than an importable module, so it
is loaded by path. It imports langchain/FAISS at module scope, hence the
``importorskip`` guards below: this test runs wherever
``pip install -e '.[benchmarks]'`` has been done and skips cleanly elsewhere.

Nothing here touches the network or downloads a dataset — ``load_domain()``,
the embedding model and the freshstack evaluator are all left alone.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("langchain_community")
pytest.importorskip("langchain_huggingface")
pytest.importorskip("langchain_core")
pytest.importorskip("rich")
pytest.importorskip("pandas")

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_freshstack.py"


@pytest.fixture(scope="module")
def bfs():
    """The benchmark script, imported by path."""
    spec = importlib.util.spec_from_file_location("benchmark_freshstack", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def retrieval_order():
    return [f"d{i}" for i in range(50)]


# ---------------------------------------------------------------------------
# B4 — equal evaluation depth across scenarios
# ---------------------------------------------------------------------------


def test_pad_to_depth_equalises_evaluation_depth(bfs, retrieval_order):
    """A 10-doc reranked result must still be evaluated over 50 docs."""
    head = ["d7", "d3", "d1", "d9", "d2", "d0", "d5", "d4", "d8", "d6"]
    padded = bfs.pad_to_depth(head, retrieval_order, 50)
    assert len(padded) == 50


def test_pad_to_depth_keeps_reranked_head_above_retrieval_tail(bfs, retrieval_order):
    """
    Padding must not disturb metrics measured no deeper than the budget:
    reranked docs keep their order and outrank every padded doc.
    """
    head = ["d7", "d3", "d1", "d9", "d2", "d0", "d5", "d4", "d8", "d6"]
    padded = bfs.pad_to_depth(head, retrieval_order, 50)

    by_rank = sorted(padded, key=lambda doc_id: -padded[doc_id])
    assert by_rank[: len(head)] == head

    worst_reranked = min(padded[doc_id] for doc_id in head)
    best_padded = max(padded[doc_id] for doc_id in by_rank[len(head) :])
    assert worst_reranked > best_padded

    # the tail follows retrieval order
    assert by_rank[len(head) :] == [d for d in retrieval_order if d not in head]


def test_pad_to_depth_is_noop_at_full_depth(bfs, retrieval_order):
    """The assembler normally already returns full depth; padding must not alter it."""
    padded = bfs.pad_to_depth(retrieval_order, retrieval_order, 50)
    assert list(padded) == retrieval_order


def test_pad_to_depth_does_not_duplicate_documents(bfs, retrieval_order):
    """Reranked docs also appear in the retrieval order; they must not be re-added."""
    padded = bfs.pad_to_depth(["d0", "d1"], retrieval_order, 50)
    assert len(padded) == len(set(padded)) == 50


def test_pad_to_depth_tolerates_short_retrieval_list(bfs):
    """Fewer candidates than the target depth must not raise."""
    assert list(bfs.pad_to_depth(["d0"], ["d0", "d1", "d2"], 50)) == ["d0", "d1", "d2"]


def test_pad_to_depth_tolerates_missing_retrieval_order(bfs):
    """An unknown query id yields an empty order; the head must survive."""
    assert list(bfs.pad_to_depth(["d1", "d2"], [], 50)) == ["d1", "d2"]


# ---------------------------------------------------------------------------
# B1 — exact metric keys
# ---------------------------------------------------------------------------


def test_metric_keys_match_freshstack_spelling(bfs):
    """
    freshstack.retrieval.metrics builds keys as f"alpha-nDCG@{k}",
    f"Coverage@{k}" and f"Recall@{k}". The alpha-nDCG spelling is hyphenated
    with a lowercase 'n' — an underscored guess silently missed before.
    """
    assert bfs.ALPHA_NDCG_KEY == "alpha-nDCG@10"
    assert bfs.COVERAGE_KEY == "Coverage@20"
    assert bfs.RECALL_KEY == "Recall@50"


def test_metric_reads_exact_key(bfs):
    metrics = {"alpha-nDCG@10": 0.42, "alpha-nDCG@20": 0.51}
    assert bfs._metric(metrics, "alpha-nDCG@10") == 0.42


def test_metric_raises_instead_of_silently_falling_back(bfs):
    """
    The old helper returned `next(iter(d.values()))` when no key matched,
    yielding a wrong-but-plausible number. alpha-nDCG genuinely has no @50
    entry (pyndeval caps k at 20), so this path was reachable.
    """
    metrics = {"alpha-nDCG@10": 0.42, "alpha-nDCG@20": 0.51}
    with pytest.raises(KeyError) as excinfo:
        bfs._metric(metrics, "alpha-nDCG@50")
    assert "available keys" in str(excinfo.value)


# ---------------------------------------------------------------------------
# B1/B2 — scenarios must stay distinguishable and budget-bound on rerank_docs
# ---------------------------------------------------------------------------


@pytest.fixture
def scenarios(bfs, monkeypatch):
    """
    The three configured scenarios, with SimilarityEstimator stubbed out.

    `SimilarityEstimator.__init__` constructs a SentenceTransformer, which
    downloads and loads a model. These tests assert on the wiring — budgets,
    scheduler batch sizes, assembler depth — so the real estimator is not
    needed and would make the test require a model download.
    """

    class StubSimilarityEstimator:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(bfs, "SimilarityEstimator", StubSimilarityEstimator)
    return bfs.build_scenarios(retriever=object())


def test_scenarios_are_distinctly_configured(scenarios):
    """
    All three reranking scenarios once reported bit-identical metrics. Guard
    the cause: each must differ in at least one of budget / batch size /
    estimator.
    """
    assert len(scenarios) == 3

    signatures = [
        (
            ctrl.budget.limits["rerank_docs"],
            ctrl.scheduler.batch_size,
            type(ctrl.estimator),
        )
        for _, ctrl in scenarios
    ]
    assert len(set(signatures)) == 3, signatures


def test_scenarios_do_not_share_a_reranker_with_mutable_gold_state(scenarios):
    """
    `_OracleReranker.set_gold()` mutates shared state between queries. The
    module-level `_reranker` is intentionally shared, but each scenario must
    pick it up by reference so `set_gold` applies — a per-scenario copy would
    silently evaluate against stale gold.
    """
    rerankers = {id(ctrl.reranker) for _, ctrl in scenarios}
    assert len(rerankers) == 1


def test_rerank_docs_is_the_only_binding_budget(scenarios):
    """
    B2: tokens and latency_ms silently capped every scenario at 10 docs. They
    must stay far above anything a scenario reaches so rerank_docs is the sole
    study variable.
    """
    for name, ctrl in scenarios:
        limits = ctrl.budget.limits
        assert limits["tokens"] >= 100_000, name
        assert limits["latency_ms"] >= 600_000, name
        assert limits["rerank_docs"] <= 20, name


def test_assembler_returns_full_evaluation_depth(scenarios, bfs):
    """
    B4: the assembler emits the reranked head followed by the retrieval tail.
    max_docs must stay at CANDIDATES_TOP_K — the GreedyAssembler default of 10
    is what made Recall@50 an unfair comparison in the first place.
    """
    for name, ctrl in scenarios:
        assert ctrl.assembler.max_docs == bfs.CANDIDATES_TOP_K, name
