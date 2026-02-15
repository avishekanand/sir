import sys
import os
import time
import random
import numpy as np
from unittest.mock import MagicMock

# Mock ST to avoid download in benchmark
mock_st = MagicMock()
sys.modules["sentence_transformers"] = mock_st

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from ragtune.core.types import ScoredDocument, RerankStrategy
from ragtune.core.budget import CostBudget
from ragtune.core.controller import RAGtuneController
from ragtune.components.retrievers import InMemoryRetriever
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.rerankers import BaseReranker
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import UtilityEstimator, SimilarityEstimator

class PerfectReranker(BaseReranker):
    """Signals 'Golden' docs with 0.95 and others with 0.2."""
    def __init__(self, golden_ids):
        self.golden_ids = set(golden_ids)

    def rerank(self, documents, query):
        results = []
        for doc in documents:
            score = 0.95 if doc.id in self.golden_ids else 0.2
            results.append(doc.model_copy(update={"score": score, "reranker_score": score}))
        return results

def setup_benchmark_pool(num_docs=100, num_golden=5):
    """
    Creates a pool where golden docs are semantically similar but don't share metadata.
    This simulates a case where metadata-based boosting (v0.1) fails but similarity (v0.2) wins.
    """
    documents = []
    # Golden docs are semantically close to each other
    golden_ids = [f"golden_{i}" for i in range(num_golden)]
    
    # We will mock the embeddings in the estimator for these IDs
    embeddings = {}
    
    for i in range(num_docs):
        is_golden = i < num_golden
        doc_id = f"golden_{i}" if is_golden else f"noise_{i}"
        
        # Meta is randomized so metadata-boosting doesn't help
        documents.append(ScoredDocument(
            id=doc_id,
            content=f"Content for {doc_id}",
            metadata={"random_tag": str(random.random())},
            score=random.uniform(0.1, 0.3), # Low initial score
            token_count=100
        ))
        
        # Simulating embeddings:
        # Golden docs are [1.0, 0.x]
        # Noise docs are [0.0, 1.0]
        if is_golden:
            embeddings[doc_id] = np.array([1.0, random.uniform(0, 0.1)])
        else:
            embeddings[doc_id] = np.array([0.0, 1.0])
            
    return documents, golden_ids, embeddings

def run_bench(estimator_type="utility"):
    docs, golden_ids, embeddings_map = setup_benchmark_pool()
    retriever = InMemoryRetriever(docs)
    class FullRetriever(InMemoryRetriever):
        def retrieve(self, query, top_k): return self.docs
    retriever = FullRetriever(docs)

    # Setup Estimator
    if estimator_type == "similarity":
        estimator = SimilarityEstimator()
        # Mock the internal model to return our simulated embeddings
        mock_model = MagicMock()
        def mock_encode(texts, **kwargs):
            # Map text back to doc id (fragile but fine for bench)
            # Content is "Content for [id]"
            ids = [t.replace("Content for ", "") for t in texts]
            return np.array([embeddings_map[did] for did in ids])
        
        mock_model.encode.side_effect = mock_encode
        estimator.model = mock_model
    else:
        estimator = UtilityEstimator()

    scheduler = ActiveLearningScheduler(batch_size=2, estimator=estimator)
    controller = RAGtuneController(
        retriever=retriever,
        reformulator=IdentityReformulator(),
        reranker=PerfectReranker(golden_ids),
        assembler=GreedyAssembler(),
        scheduler=scheduler,
        budget=CostBudget(max_reranker_docs=100)
    )

    output = controller.run("Find Golden")
    
    # Metrics
    rerank_rounds = [e for e in output.trace.events if e.action == "rerank_batch"]
    
    # Find round where all golden found
    found_count = 0
    round_to_complete = -1
    for i, r in enumerate(rerank_rounds):
        batch_ids = r.details['doc_ids']
        found_count += sum(1 for did in batch_ids if did in golden_ids)
        if found_count >= len(golden_ids):
            round_to_complete = i + 1
            break
            
    return round_to_complete, len(rerank_rounds)

if __name__ == "__main__":
    print("=== RAGtune Intelligence Benchmark: Semantic Gain ===")
    print("Scenario: Golden docs are semantically similar but share NO metadata.")
    
    ut_rounds, ut_total = run_bench("utility")
    print(f"\n[v0.1 Metadata Baseline]")
    print(f"- Rounds to find all 5 docs: {ut_rounds if ut_rounds > 0 else 'Never'}")
    
    sim_rounds, sim_total = run_bench("similarity")
    print(f"\n[v0.2 Semantic Intelligence]")
    print(f"- Rounds to find all 5 docs: {sim_rounds}")
    
    if sim_rounds < ut_rounds:
        speedup = (ut_rounds / sim_rounds)
        print(f"\n✅ SUCCESS: SimilarityEstimator is {speedup:.1f}x faster at finding hidden relevance.")
    else:
        print("\n❌ FAILURE: SimilarityEstimator did not show gain.")
