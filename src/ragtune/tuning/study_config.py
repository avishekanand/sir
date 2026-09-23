from __future__ import annotations

from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    name: str = "trec-covid"
    split: str = "test"
    irds_id: Optional[str] = None  # e.g. "irds:beir/trec-covid/test"; inferred from name if absent


class TuningStudyConfig(BaseModel):
    name: str
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    # Trial budget
    n_trials: int = 200
    n_startup_trials: int = 50       # random exploration before TPE engages
    n_parallel_workers: int = 1      # sequential by default (global config is not thread-safe)
    n_eval_queries: int = 200
    seed: int = 42

    # Persistence
    storage_url: Optional[str] = None  # None → in-memory; "sqlite:///path.db" → persistent

    # Pruner thresholds
    max_mean_rerank_docs: float = 50.0   # CostPruner: abort if projected mean > this
    max_trial_seconds: float = 120.0     # RuntimePruner: abort if projected wall time > this
    pareto_warmup_trials: int = 30       # ParetoPruner: don't prune before this many completed trials

    # Output
    output_dir: str = "tuning_results"

    # Optional overrides for search space (e.g. restrict reranker_types to a subset)
    search_space_overrides: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def clamp_startup_trials(self) -> "TuningStudyConfig":
        self.n_startup_trials = min(self.n_startup_trials, self.n_trials // 4)
        return self

    @classmethod
    def from_yaml(cls, path: str) -> TuningStudyConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
