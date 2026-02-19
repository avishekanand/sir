from typing import Dict, Any, Tuple
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
