import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.core.types import ScoredDocument
from ragtune.components.retrievers import InMemoryRetriever
from ragtune.components.rerankers import NoOpReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.utils.console import print_header, print_step, print_documents, print_trace, print_budget

def run_quickstart():
    print_header("RAGtune Quickstart")

    # 1. Setup your knowledge base
    print_step("Setting up knowledge base...")
    documents = [
        ScoredDocument(id="1", content="RAGtune handles budget constraints.", token_count=10),
        ScoredDocument(id="2", content="It supports graceful degradation.", token_count=10),
        ScoredDocument(id="3", content="Traceability is a core feature.", token_count=10),
    ]

    # 2. Initialize components
    retriever = InMemoryRetriever(documents)
    controller = RAGtuneController(
        retriever=retriever,
        reformulator=IdentityReformulator(),
        reranker=NoOpReranker(),
        assembler=GreedyAssembler(),
        scheduler=ActiveLearningScheduler(batch_size=2),
        budget=CostBudget(max_tokens=25, max_reranker_docs=10)
    )

    # 3. Run a query
    query = "What is RAGtune?"
    print_step(f"Running query: [italic]'{query}'[/italic]")
    output = controller.run(query)

    # 4. Show results
    print_documents(output.documents)
    print_trace(output.trace.events)
    print_budget(output.final_budget_state)

    # 5. Run with a strict budget to see degradation
    print_header("Budget Enforcement Demo")
    print_step("Running with Strict Budget (0 tokens max)")
    strict_budget = CostBudget(max_tokens=0)
    output_strict = controller.run(query, override_budget=strict_budget)
    
    from rich.console import Console
    Console().print(f"Documents returned: [bold red]{len(output_strict.documents)}[/bold red] (Expected 0 due to budget)")

if __name__ == "__main__":
    run_quickstart()
