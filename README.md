# RAGtune (Version 0.3 - Production Readiness)

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

## New in v0.2: Intelligence & Real Data

- **Similarity Feedback**: Unranked documents are dynamically boosted based on semantic similarity to verified "winners".
- **LangChain Adapter**: Seamlessly connect to FAISS, Chroma, or any LangChain-compatible retriever.
- **Reasoning Benchmark**: Integrated **BRIGHT** dataset sample for testing complex reasoning queries.
- **Professional CLI**: Beautiful, color-coded tables and traces using `rich`.

## Quick Start

### 1. Simple Quickstart
```bash
python3 examples/quickstart.py
```

### 2. Real Data Demo (Job Postings)
```bash
python3 examples/demo_job_postings.py
```

### 3. Reasoning Demo (BRIGHT Dataset)
```bash
python3 examples/demo_bright_retrieval.py
```

### 4. Intelligence Gain Benchmark
```bash
python3 tests/benchmarks/intelligence_gain.py
```

## Running Tests
```bash
pytest
```
