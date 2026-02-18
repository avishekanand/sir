import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from ragtune.registry import registry
from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.core.interfaces import BaseRetriever, BaseReranker, BaseReformulator, BaseAssembler, BaseScheduler, BaseEstimator

class ConfigLoader:
    """
    Loads RAGtune pipeline configuration from YAML and instantiates components.
    """
    
    @staticmethod
    def load_config(path: Path) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def create_controller(cls, config: Dict[str, Any], budget_overrides: Optional[Dict[str, float]] = None) -> RAGtuneController:
        pipeline_conf = config.get("pipeline", {})
        components_conf = pipeline_conf.get("components", {})
        budget_conf = pipeline_conf.get("budget", {})
        feedback_conf = pipeline_conf.get("feedback")

        limits = budget_conf.get("limits", {})
        if budget_overrides:
            limits.update(budget_overrides)

        budget = CostBudget(limits=limits)

        from ragtune.core.interfaces import BaseRetriever, BaseReranker, BaseReformulator, BaseAssembler, BaseScheduler, BaseEstimator, BaseFeedback
        
        def create_component(category: str, conf: Any):
            if isinstance(conf, list):
                # Special case for estimators: wrap in CompositeEstimator
                if category == "estimator":
                    sub_estimators = [create_component("estimator", c) for c in conf]
                    from ragtune.components.estimators import CompositeEstimator
                    return CompositeEstimator(estimators=sub_estimators)
                return [create_component(category, c) for c in conf]

            if not isinstance(conf, dict):
                return conf

            comp_type = conf.get("type", "default")
            params = conf.get("params", {})
            
            getter_map = {
                "retriever": registry.get_retriever,
                "reranker": registry.get_reranker,
                "reformulator": registry.get_reformulator,
                "assembler": registry.get_assembler,
                "scheduler": registry.get_scheduler,
                "estimator": registry.get_estimator,
                "feedback": registry.get_feedback,
            }
            
            getter = getter_map.get(category)
            if not getter:
                raise ValueError(f"Unknown component category: {category}")
            
            comp_cls = getter(comp_type)
            if not comp_cls:
                available = list(registry.list_all().get(category, {}).keys())
                raise ValueError(f"Component type '{comp_type}' not found for category '{category}'. Available: {available}")
            
            # Recursively handle any params that might be component configs
            processed_params = {}
            for k, v in params.items():
                if isinstance(v, dict) and "type" in v:
                    # Heuristic: if it's a dict with 'type', it might be a sub-component
                    # This is a bit vague, but for now we'll assume it's a component if it matches a category
                    # or if the parent component expects it.
                    # For RAGtune v0.56, we mainly care about the top-level lists.
                    processed_params[k] = v
                else:
                    processed_params[k] = v

            return comp_cls(**processed_params)

        # Instantiate implementation
        retriever = create_component("retriever", components_conf.get("retriever", {"type": "bm25"}))
        reranker = create_component("reranker", components_conf.get("reranker", {"type": "noop"}))
        reformulator = create_component("reformulator", components_conf.get("reformulator", {"type": "noop"}))
        assembler = create_component("assembler", components_conf.get("assembler", {"type": "greedy"}))
        scheduler = create_component("scheduler", components_conf.get("scheduler", {"type": "graceful-degradation"}))
        estimator = create_component("estimator", components_conf.get("estimator", {"type": "baseline"}))

        # Feedback
        feedback = None
        if feedback_conf:
            feedback = create_component("feedback", feedback_conf)

        return RAGtuneController(
            retriever=retriever,
            reformulator=reformulator,
            reranker=reranker,
            assembler=assembler,
            scheduler=scheduler,
            estimator=estimator,
            budget=budget,
            feedback=feedback
        )
