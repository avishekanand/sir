from typing import List
from ragtune.core.interfaces import BaseReformulator
from ragtune.core.types import RAGtuneContext
from ragtune.registry import registry

@registry.reformulator("identity")
class IdentityReformulator(BaseReformulator):
    """Pass-through reformulator that returns the original query."""
    def generate(self, context: RAGtuneContext) -> List[str]:
        if context.tracker.try_consume_reformulation():
            return [context.query]
        return []
