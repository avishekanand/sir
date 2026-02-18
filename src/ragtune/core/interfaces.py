from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from ragtune.core.types import ScoredDocument, ReformulationResult, BatchProposal, RAGtuneContext, RemainingBudgetView, EstimatorOutput
from ragtune.core.pool import CandidatePool, PoolItem

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        pass

    async def aretrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        return self.retrieve(context, top_k)

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, documents: List[PoolItem], context: RAGtuneContext, strategy: Optional[str] = None) -> Dict[str, float]:
        """Returns {doc_id: score}."""
        pass

    async def arerank(self, documents: List[PoolItem], context: RAGtuneContext, strategy: Optional[str] = None) -> Dict[str, float]:
        return self.rerank(documents, context, strategy)

class BaseReformulator(ABC):
    @abstractmethod
    def generate(self, context: RAGtuneContext) -> List[str]:
        pass

    async def agenerate(self, context: RAGtuneContext) -> List[str]:
        return self.generate(context)

class BaseAssembler(ABC):
    @abstractmethod
    def assemble(self, candidates: List[PoolItem], context: RAGtuneContext) -> List[ScoredDocument]:
        pass

    async def aassemble(self, candidates: List[PoolItem], context: RAGtuneContext) -> List[ScoredDocument]:
        return self.assemble(candidates, context)

class BaseEstimator(ABC):
    @abstractmethod
    def value(self, pool: CandidatePool, context: RAGtuneContext) -> Dict[str, EstimatorOutput]:
        """Calculates priority and other metrics for all eligible items."""
        pass

    def needs_reformulation(self, context: RAGtuneContext, current_pool: CandidatePool) -> bool:
        """Determines if query reformulation is needed based on current results."""
        return True

class BaseScheduler(ABC):
    @abstractmethod
    def select_batch(
        self,
        pool: CandidatePool,
        budget: RemainingBudgetView
    ) -> Optional[BatchProposal]:
        pass

    async def aselect_batch(
        self,
        pool: CandidatePool,
        budget: RemainingBudgetView
    ) -> Optional[BatchProposal]:
        return self.select_batch(pool, budget)

class BaseIndexer(ABC):
    """Base interface for indexing frameworks."""
    @abstractmethod
    def build(self, collection_path: str, format: str, fields: Dict[str, str], **params) -> bool:
        pass

    @abstractmethod
    def exists(self, index_path: str) -> bool:
        pass

class BaseFeedback(ABC):
    """Base interface for feedback/stop policies."""
    @abstractmethod
    def should_stop(self, state: Dict[str, Any], budget: Any, estimates: Dict[str, float]) -> Tuple[bool, str]:
        pass
