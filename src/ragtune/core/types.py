import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class RerankStrategy(str, Enum):
    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"
    IDENTITY = "identity"


class ScoredDocument(BaseModel):
    """Atomic unit of content."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0           # Current active score
    original_score: float = 0.0  # From retriever
    reranker_score: Optional[float] = None
    fusion_score: Optional[float] = None
    token_count: int = 0

class ReformulationResult(BaseModel):
    """Output of one retrieval path."""
    original_query: str
    reformulated_query: str
    strategy: str = "identity"
    candidates: List[ScoredDocument] = Field(default_factory=list)

class TraceEvent(BaseModel):
    """Log entry for decision debugging."""
    timestamp: float = Field(default_factory=time.time)
    component: str
    action: str      # e.g. "consume", "skip", "error"
    details: Dict[str, Any]

class ControllerTrace(BaseModel):
    """Full execution history."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    events: List[TraceEvent] = Field(default_factory=list)

    def add(self, component: str, action: str, **kwargs):
        self.events.append(TraceEvent(component=component, action=action, details=kwargs))

class BatchProposal(BaseModel):
    """The Scheduler's command for the next iteration."""
    document_indices: List[int]    # Indices in the master pool to process next
    strategy: RerankStrategy       # Which model to use
    estimated_utility: float = 0.0 # Why we chose this batch (for debugging)

class ControllerOutput(BaseModel):
    """Final artifact returned to user."""
    query: str
    documents: List[ScoredDocument]
    trace: ControllerTrace
    final_budget_state: Dict[str, Any]
