import os
import tempfile
import pandas as pd
import ir_datasets
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
from ragtune.utils.console import print_header, print_step, print_success

def run_ir_datasets_demo():
    dataset_id = "beir/trec-covid"
    print_header(f"RAGtune + PyTerrier: {dataset_id} Scaled Demo")
    
    # 1. Load data from ir_datasets
    print_step(f"Loading '{dataset_id}' (~171k documents)...")
    dataset = ir_datasets.load(dataset_id)
    
    # Take a larger sample of documents for indexing
    # We use a generator to avoid loading everything into memory
    def get_docs():
        for i, doc in enumerate(dataset.docs_iter()):
            if i >= 10000: break # Index 10k docs for the demo
            yield {"docno": doc.doc_id, "text": f"{doc.title} {doc.text}"}
    
    # 2. Index with PyTerrier
    print_step("Indexing 10,000 documents with PyTerrier...")
    tmp_dir = tempfile.mkdtemp()
    index_path = os.path.join(tmp_dir, "index")
    
    indexer = pt.IterDictIndexer(index_path, overwrite=True, meta={'docno': 64})
    index_ref = indexer.index(get_docs())
    
    # 3. Setup retrieval pipeline
    print_step("Setting up RAGtune + Ollama (DeepSeek) pipeline...")
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    
    ragtune_retriever = PyTerrierRetriever(bm25)
    
    controller = RAGtuneController(
        retriever=ragtune_retriever,
        reformulator=IdentityReformulator(),
        reranker=OllamaListwiseReranker(model_name="deepseek-r1:8b"), 
        assembler=GreedyAssembler(),
        scheduler=ActiveLearningScheduler(
            batch_size=2,
            estimator=SimilarityEstimator()
        ),
        budget=CostBudget(max_reranker_docs=4) # Rerank top 4 across 2 batches
    )
    
    # 4. Run sample queries
    print_step("Running sample queries with pretty telemetry...")
    
    from rich.table import Table
    from rich.console import Console
    from rich import box
    console = Console()
    
    # Get first 2 queries
    for i, q in enumerate(dataset.queries_iter()):
        if i >= 2: break
        
        console.print(f"\n[bold underline cyan]QUERY [{q.query_id}][/bold underline cyan]: [italic]{q.text}[/italic]")
        output = controller.run(q.text)
        
        # Display Budget Telemetry
        from ragtune.utils.console import print_budget
        print_budget(output.final_budget_state)
        
        # Create Pretty Results Table
        table = Table(
            title=f"Refined Results for Query {q.query_id}",
            header_style="bold white on blue",
            box=box.ROUNDED,
            show_footer=True
        )
        table.add_column("Rank", justify="center", style="dim")
        table.add_column("Document ID", style="bold yellow")
        table.add_column("Final Score", justify="right")
        table.add_column("Provenance", justify="left")
        table.add_column("Original Score", justify="right", style="dim")
        
        for idx, doc in enumerate(output.documents[:5]):
            is_reranked = doc.reranker_score is not None
            provenance = "[bold green]LLM Reranked[/bold green]" if is_reranked else "[dim]BM25 (Original)[/dim]"
            
            score_color = "bold green" if is_reranked else "white"
            orig_score = f"{doc.original_score:.2f}" if is_reranked else "-"
            
            table.add_row(
                str(idx+1),
                doc.id,
                f"[{score_color}]{doc.score:.4f}[/{score_color}]",
                provenance,
                orig_score
            )
            
        console.print(table)

    print_success("Scaled Demo Completed Successfully!")
    console.print(f"[dim]Index Location: {index_path}[/dim]")

if __name__ == "__main__":
    run_ir_datasets_demo()
