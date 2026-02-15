import pytest
import sys
import os

# Add src to sys.path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ragtune.core.types import ScoredDocument, RerankStrategy
from ragtune.core.budget import CostBudget
from ragtune.components.retrievers import InMemoryRetriever
from ragtune.core.interfaces import BaseReranker, BaseReformulator, BaseAssembler, BaseScheduler

class FakeReranker(BaseReranker):
    def rerank(self, documents: list[ScoredDocument], query: str) -> list[ScoredDocument]:
        # Simple simulated reranking: add a small score boost
        results = []
        for doc in documents:
            new_score = (doc.reranker_score or doc.score) + 0.1
            results.append(doc.model_copy(update={
                "score": new_score,
                "reranker_score": new_score
            }))
        return results

class FakeReformulator(BaseReformulator):
    def generate(self, query, tracker):
        if tracker.try_consume_reformulation():
            return [query]
        return []

class FakeAssembler(BaseAssembler):
    def assemble(self, candidates, tracker):
        # Sort by score and take first N that fit budget
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        result = []
        for doc in sorted_candidates:
            if tracker.try_consume_tokens(doc.token_count):
                result.append(doc)
        return result

class FakeScheduler(BaseScheduler):
    """Simple scheduler that proposes batches of size 1 until budget or pool is exhausted."""
    def __init__(self, batch_size=1):
        self.batch_size = batch_size

    def propose_next_batch(self, pool, processed_indices, tracker):
        if len(processed_indices) >= len(pool):
            return None
        
        # In this fake, we just pick the next unprocessed index
        next_indices = []
        for i in range(len(pool)):
            if i not in processed_indices:
                next_indices.append(i)
                if len(next_indices) >= self.batch_size:
                    break
        
        if not next_indices:
            return None

        from ragtune.core.types import BatchProposal
        return BatchProposal(
            document_indices=next_indices,
            strategy=RerankStrategy.CROSS_ENCODER
        )

@pytest.fixture
def doc_pool():
    return [
        ScoredDocument(id=f"doc_{i}", content=f"text {i}", token_count=10, score=0.1*i)
        for i in range(10)
    ]

@pytest.fixture
def fake_retriever(doc_pool):
    return InMemoryRetriever(doc_pool)

@pytest.fixture
def fake_reranker():
    return FakeReranker()

@pytest.fixture
def fake_reformulator():
    return FakeReformulator()

@pytest.fixture
def fake_assembler():
    return FakeAssembler()

@pytest.fixture
def fake_scheduler():
    return FakeScheduler()
