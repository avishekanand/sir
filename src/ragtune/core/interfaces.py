from abc import ABC, abstractmethod
from typing import List, Optional
from ragtune.core.types import ScoredDocument, ReformulationResult, BatchProposal
from ragtune.core.budget import CostTracker

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        pass

    async def aretrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        return self.retrieve(query, top_k)

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        pass

    async def arerank(self, documents: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        return self.rerank(documents, query)

class BaseReformulator(ABC):
    @abstractmethod
    def generate(self, query: str, tracker: CostTracker) -> List[str]:
        pass

    async def agenerate(self, query: str, tracker: CostTracker) -> List[str]:
        return self.generate(query, tracker)

class BaseAssembler(ABC):
    @abstractmethod
    def assemble(self, candidates: List[ScoredDocument], tracker: CostTracker) -> List[ScoredDocument]:
        pass

    async def aassemble(self, candidates: List[ScoredDocument], tracker: CostTracker) -> List[ScoredDocument]:
        return self.assemble(candidates, tracker)

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

    async def apropose_next_batch(
        self,
        pool: List[ScoredDocument],
        processed_indices: List[int],
        tracker: CostTracker
    ) -> Optional[BatchProposal]:
        return self.propose_next_batch(pool, processed_indices, tracker)
