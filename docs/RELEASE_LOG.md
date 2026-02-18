---

## v0.57 - Visualization & Interactive Editing 🎨
*Focus: Traceability, Visual Comprehension, and Rapid Iteration.*

- **Pipeline Visualization**: ASCII-based flow diagrams for declarative configurations via `ragtune visualize`.
- **Interactive Editor**: CLI-based guided editing of components, parameters, and budgets with real-time diff preview.
- **Improved Feedback Loop**: Enhanced traceability for scheduler decisions in the iterative loop.

---

## v0.56 - Declarative Lifecycle & CLI Failsafes 🛡️
*Focus: Production Hardening, Lifecycle CLI, and Standardized Estimator Interfaces.*

- **Standardized Estimator Output**: All estimators now return `EstimatorOutput` (priority, predicted_quality, predicted_latency), enabling uniform processing in the controller.
- **CompositeEstimator**: Added support for weighted ensembles of multiple estimators.
- **CLI Lifecycle v0.2**: Formalized the `init -> index -> validate -> run` flow with the v0.2 configuration schema.
- **CLI Failsafes & Overrides**: Protection against accidental config overwrites and runtime overrides for data paths and budget limits.
- **Adapter Hardening**: Significant stability updates to the PyTerrier adapter, including absolute path resolution and modern `pt.terrier.Retriever` support.
- **100% Test Compliance**: Full sweep of 80 tests verified.

---

## v0.54 - Stateful Iterative Reranking (Core v0.5) 💎
*Focus: State Machine Enforcement, Robust Budgeting, and Testing.*

- **CandidatePool State Machine**: Strict enforcement of document transitions (`CANDIDATE -> IN_FLIGHT -> RERANKED`) to ensure data integrity.
- **CostBudget v2**: Standardized `CostObject` and immutable `RemainingBudgetView` for precise resource allocation across tokens, docs, calls, and latency.
- **Iterative Loop Orchestration**: `RAGtuneController` now manages the loop with formal state guards and robust exception recovery (automatic dropping of failed docs).
- **Comprehensive Test Suite**: 44 passing unit and integration tests covering the entire core engine and component contracts.
- **Repaired Examples**: Full audit and repair of all 10 demo scripts to match the Core 0.54 signatures.
- **Scoring Precedence**: Formalized scoring hierarchy (Reranker > Estimator > Retrieval) for the final ranked list.

---

## v0.5 - The Developer Experience Update 🛠️
*Focus: CLI, Configuration, and Extensibility.*

- **RAGtune CLI**: A new command-line interface (`ragtune`) to init, list, and run pipelines.
- **Component Registry**: `@reranker`, `@retriever` decorators for auto-discovery and YAML configuration.
- **Generic Cost Interface**: `CostTracker` now supports arbitrary cost types (e.g., "GPU_FLOPS"), decoupled from hardcoded token counts.
- **Unified Context**: Refactored all components to accept a unified `RAGtuneContext` object.

---

## v0.4 - Advanced Integrations & Benchmarking (Legacy)
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
