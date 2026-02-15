import sys
import os
import time
import random

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

class BenchmarkReranker(BaseReranker):
    """Simulates a reranker that recognizes a specific 'golden' section."""
    def __init__(self, golden_section: str):
        self.golden_section = golden_section

    def rerank(self, documents: list[ScoredDocument], query: str) -> list[ScoredDocument]:
        results = []
        for doc in documents:
            # High score if in golden section, low otherwise
            score = 0.95 if doc.metadata.get("section") == self.golden_section else 0.2
            results.append(doc.model_copy(update={"score": score, "reranker_score": score}))
        return results

def run_benchmark(num_docs=100, batch_size=2):
    print(f"=== RAGtune Benchmarking: Loop Efficiency ===")
    print(f"Goal: Find all Golden Documents in a pool of {num_docs} docs efficiently.")
    print(f"Scenario: Golden docs are clustered in 'Section_4'.")
    
    # 1. Setup Pool
    # We place 10 golden docs in Section_4 (indices 40-49), and 90 noise docs elsewhere.
    documents = []
    for i in range(num_docs):
        section_id = i // 10
        section_name = f"Section_{section_id}"
        documents.append(ScoredDocument(
            id=f"doc_{i}",
            content=f"Content for doc {i} in {section_name}",
            metadata={"section": section_name},
            score=random.uniform(0.1, 0.4), # Low initial scores
            token_count=100
        ))
    
    # Shuffle the pool so the golden docs are scattered
    random.shuffle(documents)
    
    # 2. Setup System
    # Force retriever to return everything for the benchmark
    class FullRetriever(InMemoryRetriever):
        def retrieve(self, query, top_k): return self.docs
    retriever = FullRetriever(documents)
    
    reranker = BenchmarkReranker(golden_section="Section_4")
    scheduler = ActiveLearningScheduler(batch_size=batch_size)
    
    controller = RAGtuneController(
        retriever=retriever,
        reformulator=IdentityReformulator(),
        reranker=reranker,
        assembler=GreedyAssembler(),
        scheduler=scheduler,
        budget=CostBudget(max_reranker_docs=num_docs)
    )
    
    # 3. Execution
    start_time = time.time()
    output = controller.run("Find Golden")
    end_time = time.time()
    
    # 4. Metrics
    rerank_rounds = [e for e in output.trace.events if e.action == "rerank_batch"]
    
    # Analyze rounds
    golden_docs_found = 0
    total_golden = 10
    first_golden_round = -1
    last_golden_round = -1
    
    pool_lookup = {d.id: d for d in documents}
    
    print("\nBatch History:")
    for i, round_event in enumerate(rerank_rounds):
        batch_ids = round_event.details['doc_ids']
        batch_golden = sum(1 for did in batch_ids if pool_lookup[did].metadata['section'] == "Section_4")
        
        if batch_golden > 0:
            if first_golden_round == -1: first_golden_round = i + 1
            last_golden_round = i + 1
            golden_docs_found += batch_golden
            
        print(f"Round {i+1:2}: Batch={batch_ids} | Golden={batch_golden} | Utility={round_event.details['utility']:.2f}")

    print(f"\nFinal Statistics:")
    print(f"- Total Docs in Pool: {num_docs}")
    print(f"- Total Golden Docs: {total_golden}")
    print(f"- First Golden Found At Round: {first_golden_round}")
    print(f"- All Golden Found By Round: {last_golden_round} (Out of {len(rerank_rounds)})")
    print(f"- Efficiency Gain: All golden docs found in {last_golden_round*batch_size} rerank attempts vs {num_docs} total docs.")
    print(f"- Latency: {(end_time - start_time)*1000:.2f}ms")

if __name__ == "__main__":
    run_benchmark()
