# RAGtune

RAGtune is a budget-aware, iterative RAG middleware that treats **cost** and **latency** as first-class constraints.

## Architecture: The Active Loop

Unlike traditional linear RAG pipelines, RAGtune uses an **Active Learning** feedback loop to dynamically discover and prioritize relevant documents in real-time.

1.  **Retrieve**: Fetch a candidate pool.
2.  **Schedule**: Propose batches of documents to rerank based on predicted utility.
3.  **Rerank**: Use high-confidence models (Cross-Encoders, LLMs) to score batches.
4.  **Learn**: Feedback from scored docs boosts the priority of similar unranked docs.
5.  **Assemble**: Truncate results into the final context based on token budget.

## Project Structure

- `src/ragtune/core/`: Orchestration logic, budget tracking, and interfaces.
- `src/ragtune/components/`: Pluggable retrievers, rerankers, schedulers, and estimators.
- `tests/`: Unit and integration tests, plus performance benchmarks.

## Quick Start (Phase 1 Demo)

To see the iterative loop and adaptive scheduling in action:
```bash
python3 examples/demo_active_learning.py
```

To run the loop efficiency benchmark:
```bash
python3 tests/benchmarks/loop_efficiency.py
```

## Running Tests
```bash
pytest
```
