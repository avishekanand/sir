# RAGtune Release Log 📅

Detailed history of RAGtune versions and major milestones.

---

## v0.4 - Advanced Integrations & Benchmarking (Current)
*Focus: Scaling, IR Ecosystem Parity, and Rigorous Evaluation.*

- **PyTerrier Adapter**: Full integration with the PyTerrier ecosystem.
- **Ollama Listwise Reranker**: High-throughput relevance judging via local LLMs (DeepSeek).
- **Unified Benchmark Suite**: Automated harness for measuring costs/accuracy on BRIGHT and TREC-COVID.
- **Standard IR Metrics**: Support for nDCG@k and MRR in all evaluations.
- **Improved Telemetry**: Rich-based console output for real-time loop monitoring.

---

## v0.3 - Production Readiness 🚀
*Focus: Asynchronicity and Ecosystem Compatibility.*

- **Async Core**: Implementation of `arun()` for non-blocking execution.
- **LangChain Adapter**: Plug-and-play support for LangChain retrievers.
- **LlamaIndex Adapter**: Integration with LlamaIndex query engines.
- **Similarity Estimator**: First intelligent prioritization block based on embedding distance.
- **v1.0.0 Packaging**: Standardized `pyproject.toml` for library distribution.

---

## v0.2 - Intelligence Layer 🧠
*Focus: Active Learning and Real-Data Integration.*

- **Active Learning Scheduler**: The loop now dynamically selects batches based on predicted utility.
- **Cross-Encoder Rerankers**: Local reranking support via `sentence-transformers`.
- **HotpotQA Integration**: Verified the iterative loop against multi-hop reasoning datasets.
- **Traceability**: Implementation of `ControllerTrace` to log every decision in the feedback loop.

---

## v0.1 - The Foundation 🏗️
*Focus: Core Architecture and Budgeting Mechanism.*

- **Iterative Controller**: The core `while` loop implementation.
- **Cost Budgeting**: Request-scoped budget tracking (Tokens, Latency, Retrieval Count).
- **Core Interfaces**: Defined `BaseRetriever`, `BaseReranker`, and `BaseScheduler`.
- **Graceful Degradation**: Logic to return best-found results when budget is exhausted.
- **Dummy Components**: Proof-of-concept components for initial architectural verification.
