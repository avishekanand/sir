# RAGtune Core 0.56 Design Specification: Declarative Pipeline & CLI Lifecycle

This document formalizes the transition of RAGtune from a library-first approach to a **CLI-driven, declarative pipeline management** system. v0.56 introduces the v0.2 configuration schema and the operational tools to manage a pipeline's entire lifecycle.

## 1. Goal
The primary objective of v0.56 is to enable users to define, build, and run production-ready RAG pipelines entirely through a standardized configuration file and CLI. This version treats the pipeline as a **declarative asset** rather than a set of code calls.

## 2. Architecture: The Lifecycle Flow

RAGtune v0.56 enforces a clear four-stage lifecycle for every pipeline:

1.  **`init --wizard`**: Guided creation of the pipeline definition.
2.  **`index`**: Declarative data ingestion (based on the `data` and `index` sections).
3.  **`validate`**: Static check of component compatibility, registry presence, and path integrity.
4.  **`run`**: Execution of the budget-aware iterative loop on the validated asset.

## 3. Declarative Specification (v0.2)

The `ragtune_config.yaml` is now the single source of truth. The schema is expanded to include the "upstream" stages of the pipeline:

```yaml
pipeline:
  name: "My Declarative Pipeline"
  # 1. Data Source
  data:
    collection_path: "./data/docs.jsonl"
    collection_format: "jsonl"
  # 2. Ingestion Target
  index:
    framework: "pyterrier"
    params: { index_path: "./index" }
  # 3. Component Selection
  components:
    retriever: { type: "pyterrier" }
    reranker: { type: "ollama-listwise" }
    # ...
  # 4. Success Gating
  feedback: { type: "budget-stop" }
```

## 4. Operational Commands (CLI)

### A. `ragtune init --wizard`
An interactive prompt system that generates the v0.2 YAML. It ensures that the generated file is syntactically correct and includes all required sections for the full lifecycle.

### B. `ragtune index <config>`
A wrapper around `BaseIndexer`. It looks at the `data` and `index` sections of the config and builds the physical index (e.g., PyTerrier index, FAISS vector store). This replaces the need for external indexing scripts.

### C. `ragtune validate <config>`
Uses Pydantic (`models.py`) to perform:
-   **Schema Validation**: Ensures all keys and types match the v0.2 spec.
-   **Registry Check**: Verifies that chosen components (e.g., `pyterrier`, `active-learning`) are actually registered and available in the current environment.
-   **Path Check**: Ensures `collection_path` and `index_path` are consistent before execution.

## 5. Implementation: Config-Driven Orchestration

The `RAGtuneController` in v0.56 is instantiated via `ConfigLoader`, which handles the "binding" of the YAML to the code:
-   **Recursive Binding**: Lists in the YAML (e.g., multiple estimators) are automatically bound into `CompositeEstimator` types.
-   **Parameter Gating**: The controller now explicitly checks for `BaseFeedback` and `EstimatorOutput` structures to enable dynamic stopping.

## 6. Invariants
1.  **Configuration Sovereignty**: No pipeline state should exist outside the YAML configuration.
2.  **Validation Gate**: The `run` command should implicitly or explicitly favor a `validate` check.
3.  **Environment Parity**: The CLI ensures that the registry is fully loaded (adapters + components) before any operation.
