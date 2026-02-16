from abc import ABC, abstractmethod
from typing import List, Optional
from ragtune.core.types import ScoredDocument, ReformulationResult, BatchProposal, RAGtuneContext

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        pass

    async def aretrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        return self.retrieve(context, top_k)

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, documents: List[ScoredDocument], context: RAGtuneContext, strategy: Optional[str] = None) -> List[ScoredDocument]:
        pass

    async def arerank(self, documents: List[ScoredDocument], context: RAGtuneContext, strategy: Optional[str] = None) -> List[ScoredDocument]:
        return self.rerank(documents, context, strategy)

class BaseReformulator(ABC):
    @abstractmethod
    def generate(self, context: RAGtuneContext) -> List[str]:
        pass

    async def agenerate(self, context: RAGtuneContext) -> List[str]:
        return self.generate(context)

class BaseAssembler(ABC):
    @abstractmethod
    def assemble(self, candidates: List[ScoredDocument], context: RAGtuneContext) -> List[ScoredDocument]:
        pass

    async def aassemble(self, candidates: List[ScoredDocument], context: RAGtuneContext) -> List[ScoredDocument]:
        return self.assemble(candidates, context)

class BaseScheduler(ABC):
    @abstractmethod
    def propose_next_batch(
        self,
        pool: List[ScoredDocument],
        processed_indices: List[int],
        context: RAGtuneContext
    ) -> Optional[BatchProposal]:
        pass

    async def apropose_next_batch(
        self,
        pool: List[ScoredDocument],
        processed_indices: List[int],
        context: RAGtuneContext
    ) -> Optional[BatchProposal]:
        return self.propose_next_batch(pool, processed_indices, context)
