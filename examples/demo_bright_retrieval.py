import sys
import os
import json
import numpy as np
from unittest.mock import MagicMock

# Fix for macOS OpenMP duplication (common with FAISS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Mock ST to avoid downloads in demo
mock_st = MagicMock()
sys.modules["sentence_transformers"] = mock_st
def mock_encode(texts, **kwargs):
    if isinstance(texts, str):
        texts = [texts]
    embeddings = np.zeros((len(texts), 384))
    for i, text in enumerate(texts):
        t = text.lower()
        if any(w in t for w in ["mitochondrial", "respiration", "metabolism", "biology"]):
            embeddings[i, 0] = 1.0
        elif any(w in t for w in ["algorithm", "kmp", "coding", "substring"]):
            embeddings[i, 1] = 1.0
        elif any(w in t for w in ["sequence", "theorem", "math", "convergence"]):
            embeddings[i, 2] = 1.0
        else:
            # Add a tiny bit of noise so norm isn't zero
            embeddings[i, 3] = 0.01
    return embeddings
mock_st.SentenceTransformer.return_value.encode.side_effect = mock_encode

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.adapters.langchain import LangChainRetriever
from ragtune.components.rerankers import SimulatedReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import SimilarityEstimator
from ragtune.utils.console import print_header, print_step, print_documents, print_trace, print_budget

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
except ImportError:
    print("Error: Missing dependencies. Run: pip install faiss-cpu langchain-community langchain-huggingface")
    sys.exit(1)

CORPUS_PATH = "data/bright_sample/corpus.json"
QUERIES_PATH = "data/bright_sample/queries.json"

def run_bright_demo():
    print_header("RAGtune BRIGHT Dataset Demo: Reasoning-Intensive Retrieval")

    # 1. Load Manual Sample
    with open(CORPUS_PATH, "r") as f:
        corpus = json.load(f)
    with open(QUERIES_PATH, "r") as f:
        queries = json.load(f)

    docs = [Document(page_content=c["content"], metadata={"id": c["doc_id"], "source": c["source"]}) for c in corpus]
    
    # 2. Setup Vector Store
    print_step(f"Indexing {len(docs)} documents into FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = LangChainRetriever(vectorstore.as_retriever(search_kwargs={"k": 5}))

    # 3. Configure Controller
    scheduler = ActiveLearningScheduler(
        batch_size=2, 
        estimator=SimilarityEstimator()
    )
    
    controller = RAGtuneController(
        retriever=retriever,
        reformulator=IdentityReformulator(),
        reranker=SimulatedReranker(),
        assembler=GreedyAssembler(),
        scheduler=scheduler,
        budget=CostBudget(max_reranker_docs=4)
    )

    # 4. Search
    test_q = queries[0] 
    print_step(f"Target Domain: [bold blue]{test_q['domain'].upper()}[/bold blue]")
    print_step(f"Reasoning Query: [italic]'{test_q['query']}'[/italic]")
    
    output = controller.run(test_q["query"])

    # 5. Show Results
    print_documents(output.documents, title="Reranked Results (Iterative Loop)")
    print_trace(output.trace.events)
    print_budget(output.final_budget_state)

if __name__ == "__main__":
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: {CORPUS_PATH} not found. Run scripts/sample_bright.py.")
        sys.exit(1)
    run_bright_demo()
