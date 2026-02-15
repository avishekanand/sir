from abc import ABC, abstractmethod
from typing import List, Optional
from ragtune.core.types import ScoredDocument, ReformulationResult, BatchProposal
from ragtune.core.budget import CostTracker

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        pass

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        pass

class BaseReformulator(ABC):
    @abstractmethod
    def generate(self, query: str, tracker: CostTracker) -> List[str]:
        pass

class BaseAssembler(ABC):
    @abstractmethod
    def assemble(self, candidates: List[ScoredDocument], tracker: CostTracker) -> List[ScoredDocument]:
        pass

class BaseScheduler(ABC):
    @abstractmethod
    def propose_next_batch(
        self,
        pool: List[ScoredDocument],           # All candidates (ranked & unranked)
        processed_indices: List[int],         # Which ones we've already done
        tracker: CostTracker                  # Current budget status
    ) -> Optional[BatchProposal]:             # Return None to stop iteration
        """
        Look at the pool. Look at what we've already reranked.
        Decide which indices to rerank next.
        """
        pass
