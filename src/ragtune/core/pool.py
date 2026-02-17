from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ragtune.core.types import IllegalTransitionError, ItemState

class PoolItem(BaseModel):
    doc_id: str                      # Primary Key
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    state: ItemState = ItemState.CANDIDATE
    
    # Provenance
    sources: Dict[str, float] = Field(default_factory=dict) # {strategy: score}
    initial_rank: int = 0                                    # From first retriever
    
    # Iterative State
    priority_value: float = 0.0      # Set by Estimator
    reranker_score: Optional[float] = None
    reranker_strategy: Optional[str] = None

    def final_score(self) -> float:
        """
        Precedence: Reranker > Estimator > Retrieval Baseline.
        """
        if self.reranker_score is not None:
            return self.reranker_score
        if self.priority_value > 0:
            return self.priority_value
        return max(self.sources.values()) if self.sources else 0.0

class CandidatePool:
    ALLOWED_TRANSITIONS = {
        ItemState.CANDIDATE: {ItemState.IN_FLIGHT, ItemState.DROPPED},
        ItemState.IN_FLIGHT: {ItemState.RERANKED, ItemState.DROPPED},
        ItemState.RERANKED: {ItemState.DROPPED},
        ItemState.DROPPED: set()
    }

    def __init__(self, items: List[PoolItem]):
        self._items: Dict[str, PoolItem] = {it.doc_id: it for it in items}

    def transition(self, doc_ids: List[str], target: ItemState):
        """Validates current state before moving to target."""
        for did in doc_ids:
            item = self._items.get(did)
            if not item:
                continue
            
            if target not in self.ALLOWED_TRANSITIONS.get(item.state, set()):
                raise IllegalTransitionError(did, str(item.state), str(target))
            
            item.state = target

    def update_scores(self, scores: Dict[str, float], strategy: str, expected_ids: Optional[List[str]] = None):
        """Updates scores only for docs in IN_FLIGHT, then moves to RERANKED."""
        for did, score in scores.items():
            item = self._items.get(did)
            if not item:
                continue
            if item.state != ItemState.IN_FLIGHT:
                raise IllegalTransitionError(did, str(item.state), "reranked")
            
            item.reranker_score = score
            item.reranker_strategy = strategy
            item.state = ItemState.RERANKED
            
        if expected_ids:
            for did in expected_ids:
                item = self._items.get(did)
                if item and item.state == ItemState.IN_FLIGHT:
                    item.state = ItemState.DROPPED

    def apply_priorities(self, priorities: Dict[str, float]):
        """Sets priority_value for candidates."""
        for did, val in priorities.items():
            item = self._items.get(did)
            if item and item.state == ItemState.CANDIDATE:
                item.priority_value = val

    def get_eligible(self) -> List[PoolItem]:
        """Returns items in the CANDIDATE state."""
        return [it for it in self._items.values() if it.state == ItemState.CANDIDATE]

    def get_items(self, doc_ids: List[str]) -> List[PoolItem]:
        """Returns items by ID mapping."""
        return [self._items[did] for did in doc_ids if did in self._items]

    def get_active_items(self) -> List[PoolItem]:
        """Returns docs that are CANDIDATE or RERANKED (excludes IN_FLIGHT/DROPPED)."""
        return [it for it in self._items.values() 
                if it.state in (ItemState.CANDIDATE, ItemState.RERANKED)]

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())
