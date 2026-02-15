import os
import tempfile
import pandas as pd
from datasets import load_dataset
import pyterrier as pt

# Initialize PyTerrier if not already started
if not pt.started():
    pt.init()

from ragtune.core.controller import RAGtuneController
from ragtune.core.budget import CostBudget
from ragtune.adapters.pyterrier import PyTerrierRetriever, RAGtuneTransformer
from ragtune.components.rerankers import OllamaListwiseReranker
from ragtune.components.reformulators import IdentityReformulator
from ragtune.components.assemblers import GreedyAssembler
from ragtune.components.schedulers import ActiveLearningScheduler
from ragtune.components.estimators import SimilarityEstimator
from ragtune.utils.console import print_header, print_step

def run_pyterrier_demo():
    print_header("RAGtune + PyTerrier: BRIGHT Dataset Demo (Ollama LLM)")
    
    # 1. Load a tiny subset of BRIGHT (Biology)
    domain = "biology"
    print_step(f"Loading sample data from BRIGHT [{domain}]...")
    
    # Load 5 queries
    queries_ds = load_dataset('xlangai/BRIGHT', 'examples', split=domain, streaming=True)
    queries = []
    for i, q in enumerate(queries_ds):
        if i >= 2: break # Smaller set for LLM demo
        queries.append({"qid": f"q{i}", "query": q["query"]})
    queries_df = pd.DataFrame(queries)
    
    # Load a small corpus (100 documents)
    corpus_ds = load_dataset('xlangai/BRIGHT', 'documents', split=domain, streaming=True)
    docs = []
    for i, d in enumerate(corpus_ds):
        if i >= 50: break # Smaller corpus for speed
        docs.append({"docno": d["id"], "text": d["content"]})
    
    # 2. Index with PyTerrier
    print_step("Indexing documents with PyTerrier...")
    # Create a temporary directory for the index
    tmp_dir = tempfile.mkdtemp()
    index_path = os.path.join(tmp_dir, "index")
    
    # Use IterDictIndexer for easy indexing of dictionaries
    # We increase the docno length to 64 as BRIGHT IDs can be long
    indexer = pt.IterDictIndexer(index_path, overwrite=True, meta={'docno': 64})
    index_ref = indexer.index(docs)
    
    # 3. Setup retrieval pipeline
    print_step("Setting up RAGtune-boosted PyTerrier pipeline with Ollama...")
    # Standard BM25 retrieval
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", properties={"terrier.index.path": index_path})
    
    # RAGtune Retriever Adapter (wraps BM25)
    ragtune_retriever = PyTerrierRetriever(bm25)
    
    # RAGtune Controller
    controller = RAGtuneController(
        retriever=ragtune_retriever,
        reformulator=IdentityReformulator(),
        reranker=OllamaListwiseReranker(model_name="deepseek-r1:8b"), 
        assembler=GreedyAssembler(),
        scheduler=ActiveLearningScheduler(
            batch_size=2,
            estimator=SimilarityEstimator()
        ),
        budget=CostBudget(max_reranker_docs=4)
    )
    
    # 4. Integrate back into PyTerrier as a Transformer
    # This allows: retrieval >> ragtune_refinement
    ragtune_processor = RAGtuneTransformer(controller)
    
    # Create the full pipeline
    # Note: In this specific setup, RAGtuneTransformer uses the controller 
    # which ALREADY calls the retriever. So the pipeline is just the transformer.
    # However, RAGtuneTransformer can also take candidates from the previous step.
    # For this demo, let's run a query through the controller.
    
    print_step("Running sample queries...")
    for _, row in queries_df.iterrows():
        print(f"\nQuery: {row['query'][:100]}...")
        
        # Run via Controller
        output = controller.run(row['query'])
        
        print(f"Reranked Docs: {output.final_budget_state.get('rerank_docs', 0)}")
        print(f"Top results:")
        for i, doc in enumerate(output.documents[:3]):
            print(f"  {i+1}. [{doc.id}] (Score: {doc.score:.4f})")
            print(f"     Content snippet: {doc.content[:150]}...")
            
    print("\n" + "="*50)
    print("Demo completed successfully!")
    print(f"Temporary index at: {index_path}")
    print("="*50)

if __name__ == "__main__":
    run_pyterrier_demo()
