# RAGtune CLI Reference

The RAGtune CLI is the primary interface for managing the lifecycle of budget-aware RAG pipelines. It allows you to construct, index, validate, and execute pipelines declaratively.

## 🚀 Quickstart: The Guided Workflow

The easiest way to get started is using the interactive wizard:

```bash
# 1. Initialize your pipeline
ragtune init --wizard

# 2. Build the index (based on your config)
ragtune index ragtune_config.yaml

# 3. Run a query
ragtune run ragtune_config.yaml --query "What is budget-aware RAG?"
```

---

## 🛠 Command Reference

### `ragtune init`
Initializes a new `ragtune_config.yaml` file.

- **Options**:
  - `--output, -o`: Path to the output file (default: `ragtune_config.yaml`).
  - `--wizard, -w`: Run the interactive step-by-step constructor.

**Example**:
```bash
ragtune init --wizard
```

### `ragtune index`
Builds a retrieval index based on the `data` and `index` sections of your configuration.

- **Arguments**:
  - `CONFIG_PATH`: Path to your configuration file.

**Example**:
```bash
ragtune index my_pipeline.yaml
```

### `ragtune validate`
Performs static analysis on your configuration to ensure it is ready for execution.

- **Checks**:
  - Pydantic schema compliance.
  - Component registry presence (checks if `type` strings are valid).
  - Physical path integrity (data and index folders).
- **Options**:
  - `--allow-missing-index`: Skip index folder check (useful before running `index`).

**Example**:
```bash
ragtune validate my_pipeline.yaml
```

### `ragtune run`
Executes the RAG pipeline for a given query.

- **Arguments**:
  - `CONFIG_PATH`: Path to your configuration file.
- **Options**:
  - `--query, -q`: The user query string (Required).
  - `--verbose, -v`: Show the full iterative execution trace (Estimator scores, Scheduler decisions, etc.).
  - `--limit, -l`: Override budget limits at runtime (e.g., `-l tokens=1000 -l rerank_docs=5`).

**Example**:
```bash
ragtune run my_pipeline.yaml -q "How does RAGtune save cost?" --verbose
```

### `ragtune list`
Lists all registered components available in your current environment. Use this to find valid `type` strings for your configuration.

**Categories**:
- Retrievers
- Rerankers
- Reformulators
- Assemblers
- Schedulers
- Estimators
- Feedbacks
- Indexers

---

## 📂 Configuration (v0.2)

RAGtune uses a declarative YAML format. A minimal v0.2 config looks like this:

```yaml
pipeline:
  name: "Documentation Search"
  data:
    collection_path: "./data/docs.jsonl"
    collection_format: "jsonl"
  index:
    framework: "pyterrier"
    params: { index_path: "./index" }
  components:
    retriever: { type: "pyterrier" }
    reranker: { type: "ollama-listwise", params: { model_name: "deepseek-r1:8b" } }
    estimator: { type: "baseline" }
  budget:
    limits:
      tokens: 5000
      rerank_docs: 20
```

---

## 💡 Pro-Tips

### Overriding Budgets
You can run the same pipeline with different constraints without editing the YAML:
```bash
ragtune run config.yaml -q "query" --limit tokens=500 --limit rerank_docs=2
```

### Debugging with Verbose
If a document isn't appearing in your results, use `--verbose` to see if it was retrieved but dropped by the scheduler or filtered by the assembler's `min_score`.
