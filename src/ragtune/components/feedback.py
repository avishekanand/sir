from typing import Dict, Any, Optional, Tuple
from ragtune.core.interfaces import BaseFeedback
from ragtune.registry import registry

@registry.feedback("budget-stop")
class BudgetStopFeedback(BaseFeedback):
    """Stops when budget is nearly exhausted or a threshold is met."""
    def __init__(self, token_threshold: float = 0.9):
        self.token_threshold = token_threshold

    def should_stop(self, metrics: Dict[str, Any], budget: Any, estimates: Dict[str, float]) -> Tuple[bool, str]:
        # Simple logic: stop if we've used more than X% of tokens
        # Note: metrics is usually result of CandidatePool.get_metrics()
        # budget is RemainingBudgetView
        
        # This is a placeholder for more complex logic
        if budget.remaining_tokens < 100:
            return True, "Critical token budget remaining"
            
        return False, ""


@registry.feedback("reformir-convergence")
class ReformIRConvergenceFeedback(BaseFeedback):
    """
    Stops the reranking loop when ReformIR's learned source weights have converged:
    the maximum weight change across all sources between consecutive iterations
    falls below convergence_threshold. Requires ReformIREstimator (or any estimator
    that puts "reformir_weights" into EstimatorOutput.metadata).
    """
    def __init__(self, convergence_threshold: float = 0.01):
        self.convergence_threshold = convergence_threshold
        self._prev_weights: Optional[Dict[str, float]] = None

    def should_stop(self, metrics: Dict[str, Any], budget: Any, estimates: Dict[str, Any]) -> Tuple[bool, str]:
        current_weights = estimates.get("reformir_weights")
        if current_weights is None or self._prev_weights is None:
            self._prev_weights = current_weights
            return False, ""

        all_keys = set(current_weights) | set(self._prev_weights)
        delta = max(
            abs(current_weights.get(k, 0.0) - self._prev_weights.get(k, 0.0))
            for k in all_keys
        )
        self._prev_weights = current_weights

        if delta < self.convergence_threshold:
            return True, f"ReformIR weights converged (max_delta={delta:.4f})"
        return False, ""
