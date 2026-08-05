from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# ── Component menu ────────────────────────────────────────────────────────────
# Only types that can be instantiated without mandatory API keys or heavy
# model downloads are enabled by default.  Override via search_space_overrides
# in the study YAML to expand or restrict the menu.

RERANKER_TYPES = ["noop", "cross-encoder", "monot5"]
REFORMULATOR_TYPES = ["identity", "llm_rewrite", "reformir"]
ESTIMATOR_TYPES = ["baseline", "utility", "similarity", "reformir"]
SCHEDULER_TYPES = ["active-learning", "graceful-degradation"]
FEEDBACK_TYPES = ["none", "budget-stop"]

CE_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]
MONOT5_MODELS = [
    "castorini/monot5-base-msmarco",
    "castorini/monot5-large-msmarco",
]
REFORMULATOR_MODELS = ["gpt-4o-mini", "gpt-4o"]
SIMILARITY_MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]


class RAGtuneSearchSpace(BaseModel):
    """
    Defines the joint search space over RAGtune pipeline configurations.

    All list fields are menus of allowed values.  Numerical fields are
    (low, high) tuples passed to Optuna's suggest_int / suggest_float.

    Conditional parameters (e.g. ce_model only matters when reranker_type
    == "cross-encoder") are always sampled — inactive values are silently
    ignored by to_pipeline_config().  This is the standard Optuna approach
    with multivariate TPE.
    """

    # ── Discrete menus ────────────────────────────────────────────────────────
    reranker_types: List[str] = Field(default_factory=lambda: list(RERANKER_TYPES))
    reformulator_types: List[str] = Field(default_factory=lambda: list(REFORMULATOR_TYPES))
    estimator_types: List[str] = Field(default_factory=lambda: list(ESTIMATOR_TYPES))
    scheduler_types: List[str] = Field(default_factory=lambda: list(SCHEDULER_TYPES))
    feedback_types: List[str] = Field(default_factory=lambda: list(FEEDBACK_TYPES))

    ce_models: List[str] = Field(default_factory=lambda: list(CE_MODELS))
    monot5_models: List[str] = Field(default_factory=lambda: list(MONOT5_MODELS))
    reformulator_models: List[str] = Field(default_factory=lambda: list(REFORMULATOR_MODELS))
    similarity_models: List[str] = Field(default_factory=lambda: list(SIMILARITY_MODELS))

    # ── Numerical ranges (low, high) ──────────────────────────────────────────
    original_query_depth_range: Tuple[int, int] = (5, 50)
    depth_per_reformulation_range: Tuple[int, int] = (1, 20)
    max_pool_size_range: Tuple[int, int] = (10, 200)
    near_duplicate_threshold_range: Tuple[float, float] = (0.5, 0.95)
    scheduler_batch_size_range: Tuple[int, int] = (1, 20)
    assembler_max_docs_range: Tuple[int, int] = (3, 20)
    budget_rerank_docs_range: Tuple[int, int] = (5, 150)
    budget_reformulations_range: Tuple[int, int] = (0, 5)
    gd_llm_limit_range: Tuple[int, int] = (1, 10)
    gd_ce_limit_range: Tuple[int, int] = (1, 30)
    monot5_batch_sizes: List[int] = Field(default_factory=lambda: [4, 8, 16, 32])
    reformulator_n_variants_range: Tuple[int, int] = (1, 8)
    min_reranked_for_regression_range: Tuple[int, int] = (1, 10)
    budget_stop_token_threshold_range: Tuple[float, float] = (0.7, 0.99)

    def get_cardinality(self) -> int:
        return (
            len(self.reranker_types)
            * len(self.reformulator_types)
            * len(self.estimator_types)
            * len(self.scheduler_types)
            * len(self.feedback_types)
        )

    def sample(self, trial: Any) -> Dict[str, Any]:
        """
        Draw one configuration from the search space using Optuna's suggest API.
        All parameters are sampled unconditionally; inactive ones are ignored in
        to_pipeline_config().
        """
        params: Dict[str, Any] = {}

        # Discrete component selection
        params["reranker_type"] = trial.suggest_categorical("reranker_type", self.reranker_types)
        params["reformulator_type"] = trial.suggest_categorical("reformulator_type", self.reformulator_types)
        params["estimator_type"] = trial.suggest_categorical("estimator_type", self.estimator_types)
        params["scheduler_type"] = trial.suggest_categorical("scheduler_type", self.scheduler_types)
        params["feedback_type"] = trial.suggest_categorical("feedback_type", self.feedback_types)

        # Always-active retrieval / pool params
        params["original_query_depth"] = trial.suggest_int(
            "original_query_depth", *self.original_query_depth_range, log=True
        )
        params["depth_per_reformulation"] = trial.suggest_int(
            "depth_per_reformulation", *self.depth_per_reformulation_range, log=True
        )
        params["max_pool_size"] = trial.suggest_int(
            "max_pool_size", *self.max_pool_size_range, log=True
        )
        params["near_duplicate_threshold"] = trial.suggest_float(
            "near_duplicate_threshold", *self.near_duplicate_threshold_range
        )

        # Assembler
        params["assembler_max_docs"] = trial.suggest_int(
            "assembler_max_docs", *self.assembler_max_docs_range
        )

        # Budget
        params["budget_rerank_docs"] = trial.suggest_int(
            "budget_rerank_docs", *self.budget_rerank_docs_range, log=True
        )
        params["budget_reformulations"] = trial.suggest_int(
            "budget_reformulations", *self.budget_reformulations_range
        )

        # Scheduler sub-params
        params["scheduler_batch_size"] = trial.suggest_int(
            "scheduler_batch_size", *self.scheduler_batch_size_range, log=True
        )
        params["gd_llm_limit"] = trial.suggest_int(
            "gd_llm_limit", *self.gd_llm_limit_range
        )
        params["gd_ce_limit"] = trial.suggest_int(
            "gd_ce_limit", *self.gd_ce_limit_range
        )

        # Reranker sub-params
        params["ce_model"] = trial.suggest_categorical("ce_model", self.ce_models)
        params["monot5_model"] = trial.suggest_categorical("monot5_model", self.monot5_models)
        params["monot5_batch_size"] = trial.suggest_categorical(
            "monot5_batch_size", [str(b) for b in self.monot5_batch_sizes]
        )

        # Reformulator sub-params
        params["reformulator_model"] = trial.suggest_categorical(
            "reformulator_model", self.reformulator_models
        )
        params["reformulator_n_variants"] = trial.suggest_int(
            "reformulator_n_variants", *self.reformulator_n_variants_range
        )

        # Estimator sub-params
        params["similarity_model"] = trial.suggest_categorical(
            "similarity_model", self.similarity_models
        )
        params["min_reranked_for_regression"] = trial.suggest_int(
            "min_reranked_for_regression", *self.min_reranked_for_regression_range
        )

        # Feedback sub-params
        params["budget_stop_token_threshold"] = trial.suggest_float(
            "budget_stop_token_threshold", *self.budget_stop_token_threshold_range
        )

        return params

    def to_retrieval_overrides(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the subset of params that override the global config singleton.
        Applied via config.set() before each controller.run() call.
        """
        return {
            "retrieval.original_query_depth": params["original_query_depth"],
            "retrieval.depth_per_reformulation": params["depth_per_reformulation"],
            "retrieval.max_pool_size": params["max_pool_size"],
            "retrieval.near_duplicate_threshold": params["near_duplicate_threshold"],
        }

    def build_controller(self, params: Dict[str, Any], fixed_retriever: Any) -> Any:
        """
        Instantiate a RAGtuneController from sampled params.

        fixed_retriever is pre-built and shared across all trials (e.g. a
        PyTerrier BM25 retriever backed by a pre-built index).  Only the
        reranking / scheduling / estimation pipeline is tuned.
        """
        import ragtune.components  # noqa — ensures registry decorators fire
        from ragtune.registry import registry
        from ragtune.core.controller import RAGtuneController
        from ragtune.core.budget import CostBudget
        from ragtune.components.assemblers import GreedyAssembler

        reranker = self._build_reranker(params, registry)
        reformulator = self._build_reformulator(params, registry)
        estimator = self._build_estimator(params, registry)
        scheduler = self._build_scheduler(params, registry)
        assembler = GreedyAssembler(max_docs=params["assembler_max_docs"])
        feedback = self._build_feedback(params, registry)

        budget = CostBudget(limits={
            "rerank_docs": float(params["budget_rerank_docs"]),
            "reformulations": float(params["budget_reformulations"]),
            "retrieval_calls": 20.0,
            "tokens": 1_000_000.0,  # generous; token budget is not an objective here
            "latency_ms": 120_000.0,  # 2-min per-query ceiling during tuning
        })

        return RAGtuneController(
            retriever=fixed_retriever,
            reformulator=reformulator,
            reranker=reranker,
            assembler=assembler,
            scheduler=scheduler,
            estimator=estimator,
            budget=budget,
            feedback=feedback,
        )

    # ── Private builders ──────────────────────────────────────────────────────

    def _build_reranker(self, params: Dict[str, Any], registry: Any) -> Any:
        rtype = params["reranker_type"]
        cls = registry.get_reranker(rtype)
        if cls is None:
            raise ValueError(f"Reranker '{rtype}' not in registry")
        kwargs: Dict[str, Any] = {}
        if rtype == "cross-encoder":
            kwargs["model_name"] = params["ce_model"]
        elif rtype == "monot5":
            kwargs["model_name"] = params["monot5_model"]
            kwargs["batch_size"] = int(params["monot5_batch_size"])
        elif rtype == "llm":
            kwargs["model_name"] = params.get("llm_reranker_model", "gpt-4o-mini")
        return cls(**kwargs)

    def _build_reformulator(self, params: Dict[str, Any], registry: Any) -> Any:
        rtype = params["reformulator_type"]
        cls = registry.get_reformulator(rtype)
        if cls is None:
            raise ValueError(f"Reformulator '{rtype}' not in registry")
        kwargs: Dict[str, Any] = {}
        if rtype == "llm_rewrite":
            kwargs["model_name"] = params["reformulator_model"]
        elif rtype == "reformir":
            kwargs["model"] = params["reformulator_model"]
            kwargs["n_variants"] = params["reformulator_n_variants"]
        return cls(**kwargs)

    def _build_estimator(self, params: Dict[str, Any], registry: Any) -> Any:
        etype = params["estimator_type"]
        cls = registry.get_estimator(etype)
        if cls is None:
            raise ValueError(f"Estimator '{etype}' not in registry")
        kwargs: Dict[str, Any] = {}
        if etype == "similarity":
            kwargs["model_name"] = params["similarity_model"]
        elif etype == "reformir":
            kwargs["min_reranked_for_regression"] = params["min_reranked_for_regression"]
        return cls(**kwargs)

    def _build_scheduler(self, params: Dict[str, Any], registry: Any) -> Any:
        stype = params["scheduler_type"]
        cls = registry.get_scheduler(stype)
        if cls is None:
            raise ValueError(f"Scheduler '{stype}' not in registry")
        batch_size = params["scheduler_batch_size"]
        if stype == "active-learning":
            return cls(batch_size=batch_size)
        elif stype == "graceful-degradation":
            return cls(
                batch_size=batch_size,
                llm_limit=params["gd_llm_limit"],
                cross_encoder_limit=params["gd_ce_limit"],
            )
        return cls()

    def _build_feedback(self, params: Dict[str, Any], registry: Any) -> Optional[Any]:
        ftype = params["feedback_type"]
        if ftype == "none":
            return None
        cls = registry.get_feedback(ftype)
        if cls is None:
            return None
        kwargs: Dict[str, Any] = {}
        if ftype == "budget-stop":
            kwargs["token_threshold"] = params["budget_stop_token_threshold"]
        elif ftype == "reformir-convergence":
            kwargs["convergence_threshold"] = params.get("reformir_convergence_threshold", 0.01)
        return cls(**kwargs)

    def to_pipeline_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat params dict to a PipelineConfig-compatible nested dict."""
        reranker_type = params.get("reranker_type", "noop")
        reranker_params: Dict[str, Any] = {}
        if reranker_type == "cross-encoder":
            reranker_params["model_name"] = params.get("ce_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        elif reranker_type == "monot5":
            reranker_params["model_name"] = params.get("monot5_model", "castorini/monot5-base-msmarco")
            reranker_params["batch_size"] = int(params.get("monot5_batch_size", 16))
        elif reranker_type == "llm":
            reranker_params["model_name"] = params.get("llm_reranker_model", "gpt-4o-mini")

        reformulator_type = params.get("reformulator_type", "identity")
        reformulator_params: Dict[str, Any] = {}
        if reformulator_type == "llm_rewrite":
            reformulator_params["model_name"] = params.get("reformulator_model", "gpt-4o-mini")
        elif reformulator_type == "reformir":
            reformulator_params["model"] = params.get("reformulator_model", "gpt-4o-mini")
            reformulator_params["n_variants"] = params.get("reformulator_n_variants", 5)

        estimator_type = params.get("estimator_type", "baseline")
        estimator_params: Dict[str, Any] = {}
        if estimator_type == "similarity":
            estimator_params["model_name"] = params.get("similarity_model", "all-MiniLM-L6-v2")
        elif estimator_type == "reformir":
            estimator_params["min_reranked_for_regression"] = params.get("min_reranked_for_regression", 3)

        scheduler_type = params.get("scheduler_type", "graceful-degradation")
        scheduler_params: Dict[str, Any] = {"batch_size": params.get("scheduler_batch_size", 5)}
        if scheduler_type == "graceful-degradation":
            scheduler_params["llm_limit"] = params.get("gd_llm_limit", 3)
            scheduler_params["cross_encoder_limit"] = params.get("gd_ce_limit", 10)

        feedback_type = params.get("feedback_type", "none")
        feedback_cfg: Optional[Dict[str, Any]] = None
        if feedback_type != "none":
            fb_params: Dict[str, Any] = {}
            if feedback_type == "budget-stop":
                fb_params["token_threshold"] = params.get("budget_stop_token_threshold", 0.9)
            feedback_cfg = {"type": feedback_type, "params": fb_params}

        pipeline: Dict[str, Any] = {
            "name": "ragtune-pareto-config",
            "components": {
                "retriever": {"type": "pyterrier"},
                "reformulator": {"type": reformulator_type, "params": reformulator_params},
                "reranker": {"type": reranker_type, "params": reranker_params},
                "estimator": {"type": estimator_type, "params": estimator_params},
                "scheduler": {"type": scheduler_type, "params": scheduler_params},
                "assembler": {
                    "type": "greedy",
                    "params": {"max_docs": params.get("assembler_max_docs", 10)},
                },
            },
            "budget": {
                "limits": {
                    "rerank_docs": params.get("budget_rerank_docs", 50),
                    "reformulations": params.get("budget_reformulations", 1),
                    "retrieval_calls": 20,
                    "tokens": 100000,
                    "latency_ms": 30000,
                }
            },
        }

        if feedback_cfg:
            pipeline["feedback"] = feedback_cfg

        return pipeline
