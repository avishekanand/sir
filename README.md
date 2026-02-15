# RAGtune (Version 0.5 - Developer Experience)

[**Release Log**](docs/RELEASE_LOG.md) | [**Roadmap**](docs/roadmap.md)

RAGtune is a budget-aware, iterative RAG middleware that treats **cost** and **latency** as first-class constraints.

## Architecture: The Active Loop

Unlike traditional linear RAG pipelines, RAGtune uses an **Active Learning** feedback loop to dynamically discover and prioritize relevant documents in real-time.

1.  **Retrieve**: Fetch a candidate pool.
2.  **Schedule**: Propose batches of documents to rerank based on predicted utility.
3.  **Rerank**: Use high-confidence models (Cross-Encoders, LLMs) to score batches.
4.  **Learn**: Feedback from scored docs boosts the priority of similar unranked docs (**Intelligence Layer**).
5.  **Assemble**: Truncate results into the final context based on token budget.

## Project Structure

- `src/ragtune/core/`: Orchestration logic, budget tracking, and interfaces.
- `src/ragtune/components/`: Pluggable retrievers, rerankers, schedulers, and **Similarity Estimators**.
- `src/ragtune/adapters/`: Adapters for external ecosystems (**LangChain**).
- `src/ragtune/utils/`: Shared utilities like the professional **CLI Console**.
- `tests/`: Unit and integration tests, plus performance benchmarks.

## New in v0.4: Scaled Integrations & Benchmarking

- **PyTerrier Adapter**: Full support for industrial IR pipelines.
- **Ollama Listwise Reranker**: Reasoning-aware ranking via local DeepSeek models.
- **Unified Benchmark Suite**: Professional IR metrics (nDCG@5, MRR) for benchmarking RAG efficiency.
- **Scaled Collection**: Support for indexing 170k+ doc collections (TREC-COVID).

## Quick Start

### 1. RAGtune CLI (New in v0.5)
Initialize, configure, and run pipelines directly from the terminal.

```bash
# Initialize a new config file
ragtune init

# List available components
ragtune list

# Run a pipeline
ragtune run ragtune_config.yaml --query "What is active learning?"
```

### 2. Unified Benchmarking Suite
```bash
make run-benchmarks
```

### 2. Scaled PyTerrier Demo (ir_datasets)
```bash
make run-scaled-terrier
```

### 3. Basic Async Quickstart
```bash
python3 examples/quickstart.py
```

### 4. Reasoning Demo (BRIGHT Dataset)
```bash
python3 examples/demo_pyterrier_bright.py
```

## Running Tests
```bash
pytest
```
