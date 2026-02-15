import sys
import os

# Add src to path to import ragtune
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ragtune.core.types import ScoredDocument, RerankStrategy
from ragtune.core.budget import CostBudget
from ragtune.core.controller import RAGtuneController
from ragtune.components.retrievers import InMemoryRetriever
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler

def run_active_demo():
    print("=== RAGtune Active Learning Demo ===")
    
    # 1. Setup Documents
    # doc_1 (A, 0.5) - Winner
    # doc_2 (A, 0.4) - Will be boosted to 0.48
    # doc_3 (B, 0.45) - Static second choice
    documents = [
        ScoredDocument(id="doc_1", content="RAGtune works.", metadata={"section": "A"}, token_count=10, score=0.5),
        ScoredDocument(id="doc_2", content="Doc 2", metadata={"section": "A"}, token_count=10, score=0.4),
        ScoredDocument(id="doc_3", content="Doc 3", metadata={"section": "B"}, token_count=10, score=0.45),
    ]
    
    # 2. Setup Components
    class FullRetriever(InMemoryRetriever):
        def retrieve(self, query, top_k): return self.docs
            
    retriever = FullRetriever(documents)
    scheduler = ActiveLearningScheduler(batch_size=1, strategy=RerankStrategy.CROSS_ENCODER)
    
    controller = RAGtuneController(
        retriever=retriever,
        reformulator=IdentityReformulator(),
        reranker=SimulatedReranker(),
        assembler=GreedyAssembler(),
        scheduler=scheduler,
        budget=CostBudget(max_tokens=50, max_reranker_docs=3)
    )
    
    # 3. Run
    output = controller.run("RAGtune")
    
    # 4. Verify Order
    print("\n--- Execution Trace ---")
    rounds = []
    for event in output.trace.events:
        if event.component == "controller" and event.action == "rerank_batch":
            print(f"Round: {event.details['doc_ids']} | Utility: {event.details['utility']:.2f}")
            rounds.append(event.details['doc_ids'][0])
    
    print("\n--- Summary ---")
    print(f"Ranking Order: {rounds}")
    
    if rounds == ['doc_1', 'doc_2', 'doc_3']:
        print("SUCCESS: doc_2 was prioritized over doc_3 due to feedback from doc_1!")
    else:
        print(f"FAILURE: Expected ['doc_1', 'doc_2', 'doc_3'], got {rounds}")

if __name__ == "__main__":
    run_active_demo()
