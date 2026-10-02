"""User-selectable prompt templates that walk an agent through common RAGtune workflows."""

from mcp.server.mcpserver import MCPServer


def register(mcp: MCPServer) -> None:
    @mcp.prompt()
    def build_pipeline(goal: str) -> str:
        """Design, validate and try a RAGtune pipeline for a goal."""
        return f"""Build a RAGtune pipeline for this goal: {goal}

1. server_info: check which optional packages (pyterrier, faiss, sentence_transformers, litellm) are installed.
2. list_components: choose retriever, reranker, estimator, scheduler and assembler types that can run here.
3. config_template, then update_config to set components and pipeline.budget.limits.
4. validate_config until it reports no problems; visualize_config to show the flow.
5. create_pipeline (pass documents=[...] for a quick in-memory test, or index_retriever for a real index),
   then run_pipeline on 2-3 representative queries with include_trace=True.
6. Report the chosen config, the top documents, and the budget_state, and explain each choice."""

    @mcp.prompt()
    def benchmark_pipeline(benchmark: str, dataset: str) -> str:
        """Evaluate pipeline configurations on a benchmark dataset."""
        return f"""Benchmark RAGtune on {benchmark}/{dataset}.

1. list_benchmarks to confirm the name and options, then load_dataset(background=True) and poll job_status.
2. build_index (pyterrier BM25 first; a dense index if faiss/sentence_transformers are available).
3. evaluate_scenarios with the built-in scenarios (list_default_scenarios) and a small max_queries (e.g. 20),
   background=True, then poll job_status.
4. Present a table of NDCG/Recall per scenario next to its mean rerank_docs and latency,
   and say which scenario gives the best quality per unit of budget."""

    @mcp.prompt()
    def estimate_deployment_cost(workload: str) -> str:
        """Estimate cost, energy and carbon for a RAG deployment."""
        return f"""Estimate the cost of this RAG workload: {workload}

1. budget_reference to see GPUs, model profiles, pricing tables and each loader's inputs.
2. estimate_cost for the generation model (budget_type="vllm"), the reranker ("reranking" or "vllm"
   with a cross-encoder model_name), and embeddings ("embedding"), with suggest=True.
3. compare_costs across 2-4 GPU types and 2 regions; estimate_throughput to check the latency SLO.
4. Summarize cost per query and per 1M queries, energy and carbon, and the top optimization suggestions."""
