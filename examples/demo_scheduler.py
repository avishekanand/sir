import sys
import os

# Add src to path to import ragtune
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ragtune.core.types import ScoredDocument
from ragtune.core.budget import CostBudget
from ragtune.core.controller import RAGtuneController
from ragtune.components.retrievers import InMemoryRetriever
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import GreedyTieredScheduler

def run_scheduler_demo():
    print("=== RAGtune Adaptive Scheduler Demo ===")
    
    # 1. Create document pool
    documents = [
        ScoredDocument(id="doc_1", content="RAGtune is a modular RAG framework.", token_count=10),
        ScoredDocument(id="doc_2", content="It allows for budget-driven orchestration.", token_count=10),
        ScoredDocument(id="doc_3", content="Users can plug in different retrievers.", token_count=10),
        ScoredDocument(id="doc_4", content="The controller handles iterative reranking.", token_count=10),
        ScoredDocument(id="doc_5", content="Middleware architecture is cool.", token_count=10),
    ]
    
    # 2. Initialize components
    retriever = InMemoryRetriever(documents)
    reformulator = IdentityReformulator()
    reranker = SimulatedReranker()
    assembler = GreedyAssembler()
    # Fast top 5, Thorough top 2
    scheduler = GreedyTieredScheduler(fast_top_k=5, thorough_top_k=2)
    
    # 3. Setup budget (Allow enough for multiple rerank rounds)
    budget = CostBudget(
        max_tokens=50, 
        max_reranker_docs=10, 
        max_reformulations=1, 
        max_latency_ms=2000.0
    )
    
    # 4. Initialize Controller
    controller = RAGtuneController(
        retriever=retriever,
        reformulator=reformulator,
        reranker=reranker,
        assembler=assembler,
        scheduler=scheduler,
        budget=budget
    )
    
    # 5. Run pipeline
    query = "reranking"
    print(f"Running query: '{query}' with budget: {budget.model_dump()}")
    
    output = controller.run(query)
    
    # 6. Inspect Output
    print("\n--- Final Documents ---")
    for i, doc in enumerate(output.documents):
        print(f"[{i+1}] ID: {doc.id} | Score: {doc.score:.2f} | Method: {doc.metadata.get('rerank_method')} | Content: {doc.content}")
    
    print("\n--- Trace Events ---")
    for event in output.trace.events:
        print(f"[{event.component}] {event.action}: {event.details}")
        
    print("\n--- Final Budget State ---")
    print(output.final_budget_state)

if __name__ == "__main__":
    run_scheduler_demo()
