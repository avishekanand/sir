from typing import List
from ragtune.core.interfaces import BaseReformulator
from ragtune.core.budget import CostTracker

class IdentityReformulator(BaseReformulator):
    """Pass-through reformulator that returns the original query."""
    def generate(self, query: str, tracker: CostTracker) -> List[str]:
        if tracker.try_consume_reformulation():
            return [query]
        return []
