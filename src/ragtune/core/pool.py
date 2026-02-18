from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ragtune.core.types import IllegalTransitionError, ItemState, ScoredDocument

class PoolItem(BaseModel):
    doc_id: str                      # Primary Key
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    state: ItemState = ItemState.CANDIDATE
    
    # Provenance
    sources: Dict[str, float] = Field(default_factory=dict) # {source: score}
    initial_rank: int = 999                                  # Min rank across all sources
    appearances_count: int = 1                               # Number of retrieval rounds it appeared in
    
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

    def __init__(self, items: Optional[List[PoolItem]] = None):
        self._items: Dict[str, PoolItem] = {it.doc_id: it for it in items} if items else {}

    def add_items(self, docs: List[ScoredDocument], source: str):
        """Adds or updates items in the pool from a retrieval round."""
        for rank, doc in enumerate(docs):
            if doc.id in self._items:
                item = self._items[doc.id]
                item.sources[source] = doc.score
                item.initial_rank = min(item.initial_rank, rank)
                item.appearances_count += 1
            else:
                item = PoolItem(
                    doc_id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata or {},
                    sources={source: doc.score},
                    initial_rank=rank,
                    appearances_count=1
                )
                self._items[doc.id] = item

    def enforce_cap(self, max_size: int):
        """Deterministic pruning to keep pool size within limits."""
        if len(self._items) <= max_size:
            return
            
        # Sort by max source score (final_score before estimator) DESC, then doc_id ASC
        sorted_items = sorted(
            self._items.values(),
            key=lambda x: (-x.final_score(), x.doc_id)
        )
        
        self._items = {it.doc_id: it for it in sorted_items[:max_size]}

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

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates retrieval efficiency and overlap metrics."""
        total = len(self._items)
        if total == 0:
            return {}
            
        original_ids = {it.doc_id for it in self._items.values() if "original" in it.sources}
        rewrite_ids = {it.doc_id for it in self._items.values() if any(s.startswith("rewrite_") for s in it.sources)}
        
        original_only = original_ids - rewrite_ids
        rewrite_only = rewrite_ids - original_ids
        overlap = original_ids & rewrite_ids
        
        return {
            "total_unique_docs": total,
            "original_count": len(original_ids),
            "rewrite_count": len(rewrite_ids),
            "original_only_count": len(original_only),
            "rewrite_only_count": len(rewrite_only),
            "overlap_count": len(overlap),
            "rewrite_utility_ratio": len(rewrite_only) / total if total > 0 else 0
        }

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())
