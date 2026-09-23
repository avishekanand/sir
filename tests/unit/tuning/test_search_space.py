"""Unit tests for RAGtuneSearchSpace — no real models, no API calls."""
import pytest
from optuna.trial import FixedTrial

from ragtune.tuning.search_space import RAGtuneSearchSpace


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_params(space: RAGtuneSearchSpace) -> dict:
    """Full param dict covering every key — use to build FixedTrials that
    override specific parent choices without omitting conditional sub-params."""
    return {
        # Discrete component selection
        "reranker_type": space.reranker_types[0],   # "noop"
        "reformulator_type": "identity",
        "estimator_type": "baseline",
        "scheduler_type": "active-learning",
        "feedback_type": "none",
        # Always-active numerical
        "original_query_depth": 10,
        "depth_per_reformulation": 5,
        "max_pool_size": 50,
        "near_duplicate_threshold": 0.8,
        "assembler_max_docs": 10,
        "budget_rerank_docs": 30,
        "budget_reformulations": 1,
        "scheduler_batch_size": 5,
        # Conditional — only suggested when parent is active
        "gd_llm_limit": 3,
        "gd_ce_limit": 10,
        "ce_model": space.ce_models[0],
        "monot5_model": space.monot5_models[0],
        "monot5_batch_size": "16",
        "reformulator_model": space.reformulator_models[0],
        "reformulator_n_variants": 3,
        "similarity_model": space.similarity_models[0],
        "min_reranked_for_regression": 3,
        "budget_stop_token_threshold": 0.9,
    }


def _make_trial(space: RAGtuneSearchSpace) -> FixedTrial:
    """Build a FixedTrial for the default inactive path (noop/identity/baseline/active-learning/none)."""
    return FixedTrial(_all_params(space))


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSearchSpaceCardinality:
    def test_positive(self):
        ss = RAGtuneSearchSpace()
        assert ss.get_cardinality() > 0

    def test_restricted_menu_reduces_cardinality(self):
        full = RAGtuneSearchSpace()
        restricted = RAGtuneSearchSpace(reranker_types=["noop"])
        assert restricted.get_cardinality() < full.get_cardinality()


class TestSampleReturnsAllKeys:
    ALWAYS_PRESENT = {
        "reranker_type", "reformulator_type", "estimator_type",
        "scheduler_type", "feedback_type",
        "original_query_depth", "depth_per_reformulation", "max_pool_size",
        "near_duplicate_threshold", "assembler_max_docs",
        "budget_rerank_docs", "budget_reformulations",
        "scheduler_batch_size",
    }

    def test_always_active_keys_present(self):
        ss = RAGtuneSearchSpace()
        params = ss.sample(_make_trial(ss))
        assert self.ALWAYS_PRESENT <= set(params.keys())

    def test_noop_reranker_has_no_model_keys(self):
        ss = RAGtuneSearchSpace()
        params = ss.sample(_make_trial(ss))  # reranker_type="noop"
        assert "ce_model" not in params
        assert "monot5_model" not in params
        assert "monot5_batch_size" not in params

    def test_cross_encoder_includes_ce_model(self):
        ss = RAGtuneSearchSpace()
        trial = FixedTrial({**_all_params(ss), "reranker_type": "cross-encoder"})
        params = ss.sample(trial)
        assert "ce_model" in params
        assert "monot5_model" not in params

    def test_monot5_includes_model_and_batch(self):
        ss = RAGtuneSearchSpace()
        trial = FixedTrial({**_all_params(ss), "reranker_type": "monot5"})
        params = ss.sample(trial)
        assert "monot5_model" in params
        assert "monot5_batch_size" in params
        assert "ce_model" not in params

    def test_graceful_degradation_includes_gd_limits(self):
        ss = RAGtuneSearchSpace()
        trial = FixedTrial({**_all_params(ss), "scheduler_type": "graceful-degradation"})
        params = ss.sample(trial)
        assert "gd_llm_limit" in params
        assert "gd_ce_limit" in params

    def test_active_learning_has_no_gd_limits(self):
        ss = RAGtuneSearchSpace()
        params = ss.sample(_make_trial(ss))  # scheduler_type="active-learning"
        assert "gd_llm_limit" not in params
        assert "gd_ce_limit" not in params

    def test_similarity_estimator_includes_model(self):
        ss = RAGtuneSearchSpace()
        trial = FixedTrial({**_all_params(ss), "estimator_type": "similarity"})
        params = ss.sample(trial)
        assert "similarity_model" in params

    def test_budget_stop_feedback_includes_threshold(self):
        ss = RAGtuneSearchSpace()
        trial = FixedTrial({**_all_params(ss), "feedback_type": "budget-stop"})
        params = ss.sample(trial)
        assert "budget_stop_token_threshold" in params

    def test_none_feedback_has_no_threshold(self):
        ss = RAGtuneSearchSpace()
        params = ss.sample(_make_trial(ss))  # feedback_type="none"
        assert "budget_stop_token_threshold" not in params


class TestRetrievalOverrides:
    def test_keys_match_controller_config_paths(self):
        ss = RAGtuneSearchSpace()
        trial = _make_trial(ss)
        params = ss.sample(trial)
        overrides = ss.to_retrieval_overrides(params)
        assert set(overrides) == {
            "retrieval.original_query_depth",
            "retrieval.depth_per_reformulation",
            "retrieval.max_pool_size",
            "retrieval.near_duplicate_threshold",
        }

    def test_values_match_sampled_params(self):
        ss = RAGtuneSearchSpace()
        trial = _make_trial(ss)
        params = ss.sample(trial)
        overrides = ss.to_retrieval_overrides(params)
        assert overrides["retrieval.original_query_depth"] == params["original_query_depth"]
        assert overrides["retrieval.max_pool_size"] == params["max_pool_size"]


class TestBuildController:
    """build_controller should succeed for every reranker/reformulator/estimator type
    that does NOT require model downloads (noop, identity, baseline)."""

    def _noop_params(self, ss: RAGtuneSearchSpace) -> dict:
        trial = _make_trial(ss)
        return ss.sample(trial)

    def test_noop_reranker_identity_reformulator(self):
        import ragtune.components  # noqa — populate registry
        from ragtune.components.retrievers import InMemoryRetriever
        from ragtune.core.types import ScoredDocument

        docs = [ScoredDocument(id=f"d{i}", content=f"doc {i}", score=float(i)) for i in range(5)]
        retriever = InMemoryRetriever(docs)

        ss = RAGtuneSearchSpace(
            reranker_types=["noop"],
            reformulator_types=["identity"],
            estimator_types=["baseline"],
            scheduler_types=["active-learning"],
            feedback_types=["none"],
        )
        params = self._noop_params(ss)
        controller = ss.build_controller(params, retriever)
        assert controller is not None

    def test_unknown_reranker_raises(self):
        import ragtune.components  # noqa
        from ragtune.components.retrievers import InMemoryRetriever
        from ragtune.core.types import ScoredDocument

        docs = [ScoredDocument(id="d0", content="doc", score=1.0)]
        retriever = InMemoryRetriever(docs)

        ss = RAGtuneSearchSpace(reranker_types=["noop"])
        params = {
            "reranker_type": "nonexistent-reranker",
            "reformulator_type": "identity",
            "estimator_type": "baseline",
            "scheduler_type": "active-learning",
            "feedback_type": "none",
            "original_query_depth": 10,
            "depth_per_reformulation": 5,
            "max_pool_size": 50,
            "near_duplicate_threshold": 0.8,
            "assembler_max_docs": 10,
            "budget_rerank_docs": 30,
            "budget_reformulations": 1,
            "scheduler_batch_size": 5,
            "gd_llm_limit": 3,
            "gd_ce_limit": 10,
            "ce_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "monot5_model": "castorini/monot5-base-msmarco",
            "monot5_batch_size": "16",
            "reformulator_model": "gpt-4o-mini",
            "reformulator_n_variants": 3,
            "similarity_model": "all-MiniLM-L6-v2",
            "min_reranked_for_regression": 3,
            "budget_stop_token_threshold": 0.9,
        }
        with pytest.raises(ValueError, match="not in registry"):
            ss.build_controller(params, retriever)

    def test_graceful_degradation_scheduler_params_wired(self):
        import ragtune.components  # noqa
        from ragtune.components.retrievers import InMemoryRetriever
        from ragtune.core.types import ScoredDocument
        from ragtune.components.schedulers import GracefulDegradationScheduler

        docs = [ScoredDocument(id=f"d{i}", content=f"doc {i}", score=float(i)) for i in range(3)]
        retriever = InMemoryRetriever(docs)

        ss = RAGtuneSearchSpace(
            reranker_types=["noop"],
            reformulator_types=["identity"],
            estimator_types=["baseline"],
            scheduler_types=["graceful-degradation"],
            feedback_types=["none"],
        )
        trial = FixedTrial({**_all_params(ss), "scheduler_type": "graceful-degradation"})
        params = ss.sample(trial)
        params["gd_llm_limit"] = 7
        params["gd_ce_limit"] = 15

        controller = ss.build_controller(params, retriever)
        assert isinstance(controller.scheduler, GracefulDegradationScheduler)
        assert controller.scheduler.llm_limit == 7
        assert controller.scheduler.cross_encoder_limit == 15
