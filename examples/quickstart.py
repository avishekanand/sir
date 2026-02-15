from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.core.types import ScoredDocument
from ragtune.components.retrievers import InMemoryRetriever
from ragtune.components.rerankers import NoOpReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler

# 1. Setup your knowledge base
documents = [
    ScoredDocument(id="1", content="RAGtune handles budget constraints.", token_count=10),
    ScoredDocument(id="2", content="It supports graceful degradation.", token_count=10),
    ScoredDocument(id="3", content="Traceability is a core feature.", token_count=10),
]

# 2. Initialize components
retriever = InMemoryRetriever(documents)
reformulator = IdentityReformulator()
reranker = NoOpReranker()
assembler = GreedyAssembler()

# 3. Create the controller with a default budget
default_budget = CostBudget(max_tokens=25, max_reranker_docs=10)
controller = RAGtuneController(retriever, reformulator, reranker, assembler, default_budget)

# 4. Run a query
query = "What is RAGtune?"
output = controller.run(query)

print(f"Query: {output.query}")
print(f"Retrieved {len(output.documents)} documents:")
for doc in output.documents:
    print(f"- {doc.content} (Tokens: {doc.token_count})")

# 5. Inspect the trace to see what happened
print("\n--- Execution Trace ---")
for event in output.trace.events:
    print(f"[{event.component}] {event.action}: {event.details}")

# 6. Run with a strict budget to see degradation
print("\n--- Running with Strict Budget (0 tokens) ---")
strict_budget = CostBudget(max_tokens=0)
output_strict = controller.run(query, override_budget=strict_budget)
print(f"Documents returned: {len(output_strict.documents)}")
