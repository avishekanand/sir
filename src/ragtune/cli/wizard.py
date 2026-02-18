import typer
import yaml
from pathlib import Path
from typing import Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from ragtune.registry import registry

console = Console()

def run_init_wizard() -> Dict[str, Any]:
    """
    Interactive wizard to construct a RAGtune v0.2 configuration.
    """
    console.print(Panel("[bold blue]RAGtune Pipeline Wizard[/bold blue]\nLet's configure your RAG pipeline step-by-step."))

    # --- Group 1: Collection & Indexing ---
    console.print("\n[bold]Group 1: Data Collection & Indexing[/bold]")
    collection_path = Prompt.ask("Path to your document collection", default="./data/bright_sample/corpus.json")
    collection_format = Prompt.ask(
        "Collection format", 
        choices=["json", "jsonl", "trectext", "parquet", "txtdir", "pdfdir"], 
        default="json"
    )
    
    id_field = "doc_id"
    text_field = "content"
    if collection_format in ["json", "jsonl", "parquet"]:
        id_field = Prompt.ask("Field name for Document ID", default="doc_id")
        text_field = Prompt.ask("Field name for Document Text", default="content")

    index_framework = Prompt.ask(
        "Indexing framework", 
        choices=["pyterrier", "anserini", "elasticsearch", "faiss"], 
        default="pyterrier"
    )
    index_type = Prompt.ask("Index type", choices=["sparse", "dense", "hybrid"], default="sparse")
    index_path = Prompt.ask("Where to save the index?", default=f"./index/{index_framework}_index")

    # --- Group 2: Reformulation ---
    console.print("\n[bold]Group 2: Query Reformulation[/bold]")
    enable_reformulation = Confirm.ask("Enable query reformulation (LLM-based rewriting)?", default=True)
    reformulator_config = {"type": "identity", "params": {}}
    if enable_reformulation:
        ref_type = Prompt.ask(
            "Reformulation type", 
            choices=["llm-diverse", "llm-coverage", "keyword-expansion"], 
            default="llm-diverse"
        )
        if "llm" in ref_type:
            llm_id = Prompt.ask("LLM Model (e.g. ollama/deepseek-r1:8b)", default="ollama/deepseek-r1:8b")
            num_rewrites = IntPrompt.ask("Number of variations to generate", default=3)
            reformulator_config = {
                "type": ref_type,
                "params": {
                    "model_name": llm_id,
                    "num_reformulations": num_rewrites
                }
            }
        else:
            reformulator_config = {"type": ref_type, "params": {}}

    # --- Group 3: Reranker ---
    console.print("\n[bold]Group 3: Reranking[/bold]")
    enable_rerank = Confirm.ask("Enable reranking?", default=True)
    reranker_config = {"type": "noop", "params": {}}
    if enable_rerank:
        rerank_type = Prompt.ask(
            "Reranker type", 
            choices=["cross-encoder", "ollama-listwise", "colbert"], 
            default="ollama-listwise"
        )
        rerank_docs = IntPrompt.ask("How many top documents to rerank?", default=10)
        
        params = {}
        if rerank_type == "cross-encoder":
            params["model_name"] = Prompt.ask("Cross-encoder model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
        elif rerank_type == "ollama-listwise":
            params["model_name"] = Prompt.ask("Ollama model for reranking", default="deepseek-r1:8b")
        
        reranker_config = {"type": rerank_type, "params": params}
    else:
        rerank_docs = 0

    # --- Group 4: Estimators & Assemblers ---
    console.print("\n[bold]Group 4: Strategy & Scoring[/bold]")
    estimator_type = Prompt.ask(
        "Estimator (Gating logic)", 
        choices=["baseline", "token_cost", "latency", "composite"], 
        default="baseline"
    )
    assembler_type = Prompt.ask(
        "Assembler (Result merging)", 
        choices=["default", "rrf", "concat"], 
        default="default"
    )

    # --- Group 5: Feedback & Budget ---
    console.print("\n[bold]Group 5: Constraints & Optimization[/bold]")
    feedback_type = Prompt.ask(
        "Feedback policy", 
        choices=["none", "budget_stop", "quality_threshold"], 
        default="none"
    )
    
    token_limit = IntPrompt.ask("Max tokens per query", default=5000)
    latency_limit = IntPrompt.ask("Max latency (ms) per query", default=5000)
    pipeline_name = Prompt.ask("Pipeline Name", default="My RAGtune Pipeline")

    # Construct the final config dict
    config = {
        "pipeline": {
            "name": pipeline_name,
            "data": {
                "collection_path": collection_path,
                "collection_format": collection_format,
                "id_field": id_field,
                "text_field": text_field
            },
            "index": {
                "framework": index_framework,
                "type": index_type,
                "params": {"index_path": index_path}
            },
            "components": {
                "retriever": {
                    "type": "bm25",
                    "params": {"index_path": index_path}
                },
                "reformulator": reformulator_config,
                "reranker": reranker_config,
                "assembler": {"type": assembler_type},
                "scheduler": {"type": "graceful-degradation"},
                "estimator": {"type": estimator_type}
            },
            "budget": {
                "limits": {
                    "tokens": token_limit,
                    "latency_ms": float(latency_limit),
                    "rerank_docs": rerank_docs,
                    "retrieval_calls": 5
                }
            }
        }
    }
    
    if feedback_type != "none":
        config["pipeline"]["feedback"] = {"type": feedback_type, "params": {}}

    return config
