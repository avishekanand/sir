import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ragtune.core.types import ScoredDocument, RerankStrategy
from ragtune.core.budget import CostBudget
from ragtune.core.controller import RAGtuneController
from ragtune.components.retrievers import InMemoryRetriever
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import UtilityEstimator
from ragtune.utils.console import print_header, print_step, print_documents, print_trace, print_budget, console

def run_active_demo():
    print_header("RAGtune Active Learning Demo")
    
    # 1. Setup Documents
    print_step("Initializing document pool with metadata clusters...")
    documents = [
        ScoredDocument(id="doc_1", content="RAGtune works.", metadata={"section": "A"}, token_count=10, score=0.5),
        ScoredDocument(id="doc_2", content="Supporting details for A", metadata={"section": "A"}, token_count=10, score=0.4),
        ScoredDocument(id="doc_3", content="Unrelated info in B", metadata={"section": "B"}, token_count=10, score=0.45),
    ]
    
    # 2. Setup Components
    retriever = InMemoryRetriever(documents)
    # Ensure retriever returns all docs for demo
    retriever.retrieve = lambda q, top_k: documents 

    controller = RAGtuneController(
        retriever=retriever,
        reformulator=IdentityReformulator(),
        reranker=SimulatedReranker(),
        assembler=GreedyAssembler(),
        scheduler=ActiveLearningScheduler(batch_size=1, strategy="cross_encoder"),
        estimator=UtilityEstimator(),
        budget=CostBudget.simple(tokens=50, docs=3)
    )
    
    # 3. Run
    print_step("Executing Adaptive Reranking Loop...")
    output = controller.run("RAGtune")
    
    # 4. Display results
    print_documents(output.documents)
    print_trace(output.trace.events)
    
    rounds = [e.details['doc_ids'][0] for e in output.trace.events if e.action == "rerank_batch"]
    
    console.print("\n[bold magenta]Adaptive Prioritization Summary[/bold magenta]")
    console.print(f"Reranking Sequence: [bold blue]{' -> '.join(rounds)}[/bold blue]")
    
    if rounds == ['doc_1', 'doc_2', 'doc_3']:
        console.print("\n[bold green]✅ SUCCESS: Active Learning worked![/bold green]")
        console.print("[dim]1. doc_1 was reranked first and confirmed relevance.[/dim]")
        console.print("[dim]2. UtilityEstimator boosted doc_2 because it shares metadata with doc_1.[/dim]")
        console.print("[dim]3. doc_2 leapfrogged doc_3 despite having lower initial score.[/dim]")
    else:
        console.print(f"\n[bold red]❌ FAILURE: Expected ['doc_1', 'doc_2', 'doc_3'], got {rounds}[/bold red]")

if __name__ == "__main__":
    run_active_demo()
