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

        if budget_overrides:
            budget_conf.update(budget_overrides)

        # Instantiate Budget
        # Handle legacy vs new budget format if needed, but for CLI we prefer the dict format
        # CostBudget accepts **kwargs so passing the dict directly should work if keys match
        # But wait, CostBudget now expects 'limits' dict or kwargs mapping to it via validator.
        # Let's pass it as kwargs for simplicity if the keys match simple names
        budget = CostBudget(limits=budget_conf)

        # Helper to instantiate component from registry
        def create_component(category: str, conf: Dict[str, Any], base_class):
            comp_type = conf.get("type", "default") # Ensure default type is handled if registry has it
            params = conf.get("params", {})
            
            # Get class from registry
            # We need to map category to registry getter
            getter_map = {
                "retriever": registry.get_retriever,
                "reranker": registry.get_reranker,
                "reformulator": registry.get_reformulator,
                "assembler": registry.get_assembler,
                "scheduler": registry.get_scheduler,
                "estimator": registry.get_estimator,
            }
            
            getter = getter_map.get(category)
            if not getter:
                raise ValueError(f"Unknown component category: {category}")
            
            comp_cls = getter(comp_type)
            if not comp_cls:
                raise ValueError(f"Component type '{comp_type}' not found in registry for category '{category}'. Available: {list(registry.list_all()[category].keys())}")
            
            # Instantiate
            return comp_cls(**params)

        # Instantiate implementation
        retriever = create_component("retriever", components_conf.get("retriever", {"type": "noop"}), BaseRetriever)
        reranker = create_component("reranker", components_conf.get("reranker", {"type": "noop"}), BaseReranker)
        reformulator = create_component("reformulator", components_conf.get("reformulator", {"type": "noop"}), BaseReformulator)
        assembler = create_component("assembler", components_conf.get("assembler", {"type": "greedy"}), BaseAssembler)
        scheduler = create_component("scheduler", components_conf.get("scheduler", {"type": "graceful-degradation"}), BaseScheduler)
        estimator = create_component("estimator", components_conf.get("estimator", {"type": "baseline"}), BaseEstimator)

        return RAGtuneController(
            retriever=retriever,
            reformulator=reformulator,
            reranker=reranker,
            assembler=assembler,
            scheduler=scheduler,
            estimator=estimator,
            budget=budget
        )
