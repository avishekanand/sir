import sys
import os
from typing import List

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Fix for macOS OpenMP duplication (common with FAISS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Mock sentence_transformers to avoid massive downloads for simple code verification
# If the user has it installed, they can remove this mock.
import numpy as np
from unittest.mock import MagicMock
if "sentence_transformers" not in sys.modules:
    mock_st = MagicMock()
    sys.modules["sentence_transformers"] = mock_st
    # Return a numpy array with dynamic length to satisfy FAISS validation
    mock_st.SentenceTransformer.return_value.encode.side_effect = lambda texts, **kwargs: np.zeros((len(texts), 384))

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.adapters.langchain import LangChainRetriever
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
except ImportError:
    print("Error: Please install dependencies: pip install faiss-cpu langchain-community langchain-huggingface")
    sys.exit(1)

def run_real_retriever_demo():
    print("=== RAGtune: Real Retriever Integration (LangChain + FAISS) ===\n")

    # 1. Create a minimal document collection
    raw_texts = [
        "RAGtune is a budget-aware middleware for RAG pipelines.",
        "The controller manages an iterative reranking loop.",
        "Active Learning is used to prioritize documents in the pool.",
        "FAISS is an efficient library for dense vector similarity search.",
        "LangChain provides a unified interface for various RAG components."
    ]
    docs = [Document(page_content=t, metadata={"id": str(i), "tokens": 10}) for i, t in enumerate(raw_texts)]

    # 2. Setup local vector store (Minimal RAG)
    print("Initializing FAISS index with local embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # 3. Wrap with RAGtune Adapter
    print("Wrapping FAISS with LangChainRetriever adapter...")
    retriever = LangChainRetriever(vectorstore.as_retriever(search_kwargs={"k": 5}))

    # 4. Configure RAGtune Loop
    # We'll use the ActiveLearningScheduler for batch-by-batch processing
    scheduler = ActiveLearningScheduler(batch_size=2)
    budget = CostBudget(max_tokens=30, max_reranker_docs=4)

    controller = RAGtuneController(
        retriever=retriever,
        reformulator=IdentityReformulator(),
        reranker=SimulatedReranker(),
        assembler=GreedyAssembler(),
        scheduler=scheduler,
        budget=budget
    )

    # 5. Run it!
    query = "How does RAGtune handle reranking?"
    print(f"\nQuerying: '{query}'")
    
    output = controller.run(query)

    # 6. Verbose Output
    print("\n--- Final Context Documents ---")
    for i, doc in enumerate(output.documents):
        print(f"[{i+1}] {doc.content} (Score: {doc.score:.2f})")

    print("\n--- Iterative Execution Trace ---")
    for event in output.trace.events:
        if event.action == "rerank_batch":
            print(f"Rerank Round: {event.details['doc_ids']} | Utility: {event.details['utility']:.2f}")
        elif "deny" in event.action:
             print(f"Budget Denied: {event.action} - {event.details.get('reason')}")

    print("\nFinal Budget Used:")
    print(output.final_budget_state)

if __name__ == "__main__":
    run_real_retriever_demo()
