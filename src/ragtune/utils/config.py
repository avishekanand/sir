import yaml
from typing import Any, Dict
from pathlib import Path

class ConfigLoader:
    _instance = None
    _config = {}
    _prompts = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_all()
        return cls._instance

    def _load_all(self):
        base_path = Path(__file__).parent.parent / "config"
        
        # Load defaults
        defaults_path = base_path / "defaults.yaml"
        if defaults_path.exists():
            with open(defaults_path, "r") as f:
                self._config = yaml.safe_load(f) or {}

        # Load prompts
        prompts_path = base_path / "prompts.yaml"
        if prompts_path.exists():
            with open(prompts_path, "r") as f:
                self._prompts = yaml.safe_load(f) or {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a value from config using dot notation (e.g., 'retrieval.pool_multiplier')"""
        return self._get_recursive(self._config, key_path, default)

    def get_prompt(self, key_path: str, default: Any = None) -> Any:
        """Get a prompt from prompts using dot notation"""
        return self._get_recursive(self._prompts, key_path, default)

    def _get_recursive(self, data: Dict, key_path: str, default: Any) -> Any:
        keys = key_path.split(".")
        val = data
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

# Global instance
config = ConfigLoader()
