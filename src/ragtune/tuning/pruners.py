from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class TuningPruner(ABC):
    """
    Evaluates whether a trial-in-progress should be aborted early.

    report() is called after each query is evaluated.  should_prune()
    is checked immediately after; if it returns True the evaluator raises
    optuna.TrialPruned.
    """

    @abstractmethod
    def report(
        self,
        *,
        step: int,
        n_total: int,
        obj1: float,
        obj2: float,
        elapsed_ms: float,
    ) -> None:
        """
        step      — number of queries evaluated so far (1-indexed)
        n_total   — total queries planned for this trial
        obj1      — running mean of NDCG@10 (maximize)
        obj2      — running mean of rerank_docs consumed (minimize)
        elapsed_ms — cumulative wall time so far
        """

    @abstractmethod
    def should_prune(self) -> bool:
        """Return True to abort this trial."""

    def reset(self) -> None:
        """Called at the start of each new trial so the pruner can clear state."""


class CostPruner(TuningPruner):
    """
    Aborts a trial when the running mean of rerank_docs per query exceeds
    max_mean_rerank_docs.  Intended to kill configurations with very
    expensive rerankers before they exhaust the entire evaluation set.
    """

    def __init__(self, max_mean_rerank_docs: float, warmup_steps: int = 3):
        self.max_mean_rerank_docs = max_mean_rerank_docs
        self.warmup_steps = warmup_steps
        self._step = 0
        self._total_obj2 = 0.0

    def reset(self) -> None:
        self._step = 0
        self._total_obj2 = 0.0

    def report(self, *, step: int, n_total: int, obj1: float, obj2: float, elapsed_ms: float) -> None:
        # obj2 passed here is the per-query value, not a running mean
        self._step = step
        self._total_obj2 += obj2

    def should_prune(self) -> bool:
        if self._step < self.warmup_steps:
            return False
        mean_cost = self._total_obj2 / self._step
        return mean_cost > self.max_mean_rerank_docs


class RuntimePruner(TuningPruner):
    """
    Projects total evaluation time based on current pace and aborts if
    the projection exceeds max_trial_seconds.
    """

    def __init__(self, max_trial_seconds: float, warmup_steps: int = 3):
        self.max_trial_seconds = max_trial_seconds
        self.warmup_steps = warmup_steps
        self._step = 0
        self._n_total = 0
        self._elapsed_ms = 0.0

    def reset(self) -> None:
        self._step = 0
        self._n_total = 0
        self._elapsed_ms = 0.0

    def report(self, *, step: int, n_total: int, obj1: float, obj2: float, elapsed_ms: float) -> None:
        self._step = step
        self._n_total = n_total
        self._elapsed_ms = elapsed_ms

    def should_prune(self) -> bool:
        if self._step < self.warmup_steps or self._step == 0:
            return False
        return (self._elapsed_ms / 1000.0) > self.max_trial_seconds


class ParetoPruner(TuningPruner):
    """
    Prunes a trial when its optimistic best-case outcome is dominated by
    the current Pareto front.

    "Optimistic best-case" uses confidence bounds:
      - obj1 upper bound = mean_ndcg + z * std_err(ndcg)   (binomial CI)
      - obj2 lower bound = mean_cost - z * std_err(cost)    (conservative)

    If every Pareto-front point dominates this optimistic point — i.e.,
    is at least as good on quality AND strictly better on cost — the trial
    cannot improve the front and is pruned.

    The pruner is dormant until warmup_trials completed trials exist, so
    that the Pareto front estimate is reliable.
    """

    def __init__(
        self,
        study: object,
        warmup_trials: int = 30,
        zscore: float = 1.645,  # 90% one-sided CI
        min_steps_before_prune: int = 5,
    ):
        self.study = study
        self.warmup_trials = warmup_trials
        self.zscore = zscore
        self.min_steps_before_prune = min_steps_before_prune

        self._step = 0
        self._obj1_sum = 0.0
        self._obj1_sq_sum = 0.0
        self._obj2_sum = 0.0
        self._obj2_sq_sum = 0.0
        self._fired = False  # latch: once True, always prune

    def reset(self) -> None:
        self._step = 0
        self._obj1_sum = 0.0
        self._obj1_sq_sum = 0.0
        self._obj2_sum = 0.0
        self._obj2_sq_sum = 0.0
        self._fired = False

    def report(self, *, step: int, n_total: int, obj1: float, obj2: float, elapsed_ms: float) -> None:
        self._step = step
        self._obj1_sum += obj1
        self._obj1_sq_sum += obj1 * obj1
        self._obj2_sum += obj2
        self._obj2_sq_sum += obj2 * obj2

    def should_prune(self) -> bool:
        if self._fired:
            return True

        if self._step < self.min_steps_before_prune:
            return False

        try:
            import optuna
            completed = [
                t for t in self.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
            ]
        except Exception:
            return False

        if len(completed) < self.warmup_trials:
            return False

        pareto = _pareto_front(completed)
        if not pareto:
            return False

        # Confidence bounds on current trial's running means
        n = self._step
        mean_obj1 = self._obj1_sum / n
        mean_obj2 = self._obj2_sum / n

        # NDCG is bounded in [0, 1], use binomial-style interval
        obj1_conf = self.zscore * math.sqrt(max(mean_obj1 * (1.0 - mean_obj1), 0.0) / n)
        optimistic_obj1 = min(mean_obj1 + obj1_conf, 1.0)

        # Cost: use sample std dev for confidence
        var_obj2 = max(self._obj2_sq_sum / n - mean_obj2 ** 2, 0.0)
        obj2_conf = self.zscore * math.sqrt(var_obj2 / n) if n > 1 else 0.0
        optimistic_obj2 = max(mean_obj2 - obj2_conf, 0.0)

        # Dominated iff every Pareto point is weakly better on obj1 AND weakly better on obj2
        dominated = any(
            p_obj1 >= optimistic_obj1 and p_obj2 <= optimistic_obj2
            for (p_obj1, p_obj2) in pareto
        )
        if dominated:
            self._fired = True
        return dominated


def _pareto_front(trials: list) -> List[Tuple[float, float]]:
    """Return (obj1, obj2) pairs that are non-dominated (maximize obj1, minimize obj2)."""
    points = [(t.values[0], t.values[1]) for t in trials]
    front = []
    for p in points:
        if not any(q[0] >= p[0] and q[1] <= p[1] and q != p for q in points):
            front.append(p)
    return front
