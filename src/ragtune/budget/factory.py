"""
Budget Loader Factory
======================
Registry-based factory for creating budget loaders.

Usage:
    factory = BudgetLoaderFactory()
    loader = factory.create("vllm", config=budget_config)
    result = loader.calculate(context)
"""

import os
import yaml
from typing import Dict, Any, Optional

from ragtune.budget.base import BaseBudgetLoader, BudgetConfig


class BudgetLoaderFactory:
    """Factory that creates budget loaders by registry key.

    Follows the same pattern as DataLoaderFactory and IndexFactory —
    loaders register themselves via a decorator-style mechanism.
    """

    _REGISTRY: Dict[str, type] = {}

    @classmethod
    def register(cls, key: str):
        """Decorator to register a budget loader class."""

        def wrapper(loader_cls):
            cls._REGISTRY[key] = loader_cls
            loader_cls.key = key
            return loader_cls

        return wrapper

    @classmethod
    def create(
        cls,
        budget_type: str = "vllm",
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ) -> BaseBudgetLoader:
        """Create a budget loader by type.

        Args:
            budget_type: "vllm", "token", "gpu_util", "carbon", "electricity"
            config: Optional dict of config values (overrides YAML)
            config_path: Optional path to YAML config file

        Returns:
            BaseBudgetLoader instance
        """
        # Load config from YAML if path given, then overlay explicit config
        # dict values on top (config overrides YAML, matching the docstring).
        budget_config = None
        if config_path:
            with open(config_path) as f:
                yaml_cfg = yaml.safe_load(f)
                if config:
                    merged = dict(yaml_cfg)
                    merged.update(config)
                    budget_config = BudgetConfig(merged)
                else:
                    budget_config = BudgetConfig(yaml_cfg)
        elif config:
            budget_config = BudgetConfig(config)

        loader_cls = cls._REGISTRY.get(budget_type)
        if loader_cls is None:
            available = ", ".join(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown budget type: {budget_type!r}. Available: {available}"
            )
        return loader_cls(config=budget_config)

    @classmethod
    def from_env(cls) -> BaseBudgetLoader:
        """Create budget loader from BUDGET_TYPE env var.

        Config is loaded from BUDGET_CONFIG_PATH env var or default YAML.
        """
        budget_type = os.environ.get("BUDGET_TYPE", "vllm")
        config_path = os.environ.get(
            "BUDGET_CONFIG_PATH",
            os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
        )
        return cls.create(budget_type, config_path=config_path)

    @classmethod
    def list_types(cls) -> list:
        return list(cls._REGISTRY.keys())
