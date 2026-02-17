import sys
import os
import csv
import numpy as np
from unittest.mock import MagicMock

# Fix for macOS OpenMP duplication (common with FAISS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Mock ST to avoid downloads in demo
if "sentence_transformers" not in sys.modules:
    mock_st = MagicMock()
    sys.modules["sentence_transformers"] = mock_st
    mock_st.SentenceTransformer.return_value.encode.side_effect = lambda texts, **kwargs: np.random.rand(len(texts), 384)

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

CSV_PATH = "/Users/avishekanand/Projects/search-engine/data/documents/job_postings.csv"

def load_sample_docs(limit=200):
    print_step(f"Sampling first {limit} jobs from {CSV_PATH}...")
    documents = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if count >= limit: break
            if len(row) < 27: continue
            
            title = row[26]
            description = row[21]
            location = row[12]
            company = row[4]
            
            content = f"Title: {title}\nCompany: {company}\nLocation: {location}\nDetails: {description}"
            
            documents.append(Document(
                page_content=content, 
                metadata={"id": f"job_{count}", "source": company, "location": location}
            ))
            count += 1
    return documents

def run_real_data_demo():
    print_header("RAGtune Real Data Demo: Job Postings")

    # 1. Load Data
    docs = load_sample_docs(limit=200)
    
    # 2. Setup Vector Store
    print_step(f"Indexing {len(docs)} documents into FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = LangChainRetriever(vectorstore.as_retriever(search_kwargs={"k": 20}))

    # 3. Configure Controller with Intelligence (SimilarityEstimator)
    # This allows RAGtune to boost similar jobs if it finds a "winner"
    controller = RAGtuneController(
        retriever=retriever,
        reformulator=IdentityReformulator(),
        reranker=SimulatedReranker(),
        assembler=GreedyAssembler(),
        scheduler=ActiveLearningScheduler(batch_size=3),
        estimator=SimilarityEstimator(),
        budget=CostBudget.simple(docs=9) # Allow 3 rounds of 3 docs
    )

    # 4. Search!
    query = "Software development roles with remote options"
    print_step(f"Searching for: [italic]'{query}'[/italic]")
    
    output = controller.run(query)

    # 5. Show Results
    print_documents(output.documents[:5], title="Top Recommended Jobs (Budget-Aware)")
    print_trace(output.trace.events)
    print_budget(output.final_budget_state)

if __name__ == "__main__":
    run_real_data_demo()
