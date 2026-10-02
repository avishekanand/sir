# RAGtune MCP Server

`ragtune.mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server that lets an LLM agent (Claude Code, Claude Desktop, or any MCP client) use RAGtune through typed tools: inspect components, write and validate configs, run pipelines, load datasets, build indexes, evaluate, estimate cost, and run the repo's scripts and tests. Spec: `specs/mcp-server.md`.

## Setup

```bash
pip install -e ".[mcp]"        # adds mcp>=2.2 (Python >= 3.10)
ragtune-mcp --help              # or: python -m ragtune.mcp --help
```

**Claude Code** (run from the repo root):

```bash
claude mcp add ragtune -- "$(pwd)/.venv/bin/ragtune-mcp" --root "$(pwd)"
```

**Claude Desktop / any client using an `mcpServers` JSON config:**

```json
{
  "mcpServers": {
    "ragtune": {
      "command": "/abs/path/to/sir/.venv/bin/ragtune-mcp",
      "args": ["--root", "/abs/path/to/sir"]
    }
  }
}
```

**HTTP:** `ragtune-mcp --transport streamable-http --port 8000` serves `http://127.0.0.1:8000/mcp`.

## Conventions

- **Workspace root.** Relative paths resolve against `--root` (default: `$RAGTUNE_MCP_ROOT`, else the working directory). Paths outside it are rejected.
- **Errors explain the fix.** Failures come back as tool errors with the original message, e.g. `ValueError: No reranker named 'nope'. Available: ['cross-encoder', ...]`.
- **Handles.** Pipelines, datasets and jobs are referenced by ids (`pipeline-1`, `dataset-2`, `job-3`) that live as long as the server process.
- **Background work.** Slow tools accept `background=True` and return a `job_id`; poll `job_status`.
- **Output safety.** Under stdio the SDK diverts the process's stdout to stderr, so prints from RAGtune, Rich or the PyTerrier JVM never corrupt the protocol stream.

## Tools

### Discovery and jobs

| Tool | Purpose |
|---|---|
| `server_info` | versions, workspace root, which optional packages are installed, open handles |
| `list_components(category?)` | registered component names (the `type` strings in configs) with constructor parameters |
| `describe_component(category, name)` | full docstring, class path, source file, parameters |
| `get_settings(key?, prompts?)` / `set_setting(key, value)` | read or override runtime defaults (`config/defaults.yaml`) and prompt templates |
| `list_default_scenarios` | the 7 built-in benchmark scenarios |
| `job_status(job_id)` / `list_jobs` / `cancel_job(job_id)` | follow and stop background work |

### Configs

| Tool | Purpose |
|---|---|
| `config_template(output_path?)` | the `ragtune init` starter config, optionally written to a file |
| `read_config(config_path)` / `write_config(config_path, config)` | load or save YAML; writing reports validation problems without blocking |
| `update_config(updates, remove, config_path or config, output_path?)` | set/remove values by dotted path (`pipeline.components.reranker.type`), returns a unified diff; the non-interactive `ragtune visualize --edit` |
| `validate_config(config_path or config)` | `ragtune validate`: schema, registered types, index path |
| `visualize_config(config_path or config)` | the `ragtune visualize` ASCII diagram |

### Datasets

| Tool | Purpose |
|---|---|
| `list_benchmarks` | every benchmark `load_dataset` understands (BRIGHT, BEIR, FreshStack, ToolRet, SkillRet, SRA-Bench, CRUMB, OBLIQ, ir_datasets, HuggingFace, local files), datasets and options |
| `load_dataset(benchmark, dataset, ...)` | load and keep a split in memory as a `dataset_id` (`background=True` for downloads) |
| `list_datasets` / `drop_dataset` | loaded datasets and their sizes / free memory |
| `get_queries` / `get_documents` | page through queries (optionally with qrels; BRIGHT includes reasoning) and documents |
| `get_qrels(kind=...)` | `qrels`, or loader extras: `excluded_ids` (BRIGHT, OBLIQ), `nugget_qrels` (FreshStack) |
| `export_corpus` | write the corpus as JSONL for `ragtune index` or a config's `data` section |

Local files: `load_dataset(benchmark="local", options={"corpus_path": "data/corpus.jsonl", "queries_path": "data/queries.jsonl", "qrels_path": "data/qrels.tsv"})`. Queries are JSONL `{"id", "text"}`; qrels are BEIR TSV (`query-id`, `corpus-id`, `score` header) or JSONL.

### Indexing

| Tool | Purpose |
|---|---|
| `build_index(index_type, index_path, dataset_id or collection_path)` | build `pyterrier` (BM25), `faiss`, `numpy` or `flex` indexes; reuses an existing index unless `overwrite=True` |
| `build_index_from_config(config_path)` | `ragtune index`: the config's `data` + `index` sections |
| `index_status` | existence, files and recorded metadata |
| `search_index` | query an index directly; with `dataset_id`, hits include text; `backend` selects the flex retriever |

### Pipelines and evaluation

| Tool | Purpose |
|---|---|
| `create_pipeline(config_path or config, ...)` | build a controller once (models load here) and keep it as a `pipeline_id`; options: `limit_overrides`, inline `documents`, `index_retriever`, `cost_estimation`, `initial_top_k` |
| `run_pipeline(query, pipeline_id or config)` | `ragtune run`: ranked documents + final budget state, optional decision trace, per-run `limit_overrides` |
| `list_pipelines` / `close_pipeline` | open pipelines / free their models |
| `evaluate_run(qrels, results)` | NDCG, MAP, Recall, Precision at k, and MRR for any run |
| `evaluate_pipeline(dataset_id, pipeline_id or config)` | run a pipeline over a dataset's queries: metrics, mean budget usage, per-query errors |
| `evaluate_scenarios(dataset_id, index_retriever, scenarios?)` | benchmark several configs with one shared retriever (default: the 7 built-in scenarios) |

`index_retriever` (`{"index_type", "index_path", "indexer_params"?, "dataset_id"?, "backend"?}`) retrieves through any index from `build_index`, BM25 or dense, with document text taken from the dataset so rerankers see real content.

A typical evaluation session:

```text
load_dataset(benchmark="sra_bench", dataset="toolqa", background=True)  -> job_status -> dataset-1
build_index(index_type="pyterrier", index_path="indexes/toolqa", dataset_id="dataset-1")
evaluate_scenarios(dataset_id="dataset-1", max_queries=50, background=True,
                   index_retriever={"index_type": "pyterrier", "index_path": "indexes/toolqa", "dataset_id": "dataset-1"})
```

### Budget and cost

| Tool | Purpose |
|---|---|
| `estimate_cost(budget_type, config, config_path, context)` | `ragtune budget` for any loader (`vllm`, `token`, `gpu_util`, `carbon`, `embedding`, `reranking`); `suggest`, `thresholds` (alerts) and `log_to` (JSONL history) are optional |
| `compare_costs(variants, ...)` | sweep GPUs, regions, request rates, models and rank the results |
| `validate_budget_config` | value errors plus keys `BudgetConfig` would silently ignore (e.g. `embedding_model` outside `extra`) |
| `budget_reference(table?)` | GPU specs, model profiles, empirical throughput, regional carbon, embedding/reranking/token pricing, defaults, and each loader's inputs |
| `estimate_hardware` | GPU + CPU power, energy and carbon for a runtime |
| `estimate_throughput` | peak and achieved throughput, saturation knee, VRAM fit |
| `cost_history` / `clear_cost_history` | read (entries or summary) or delete a cost-history JSONL |

## Resources

| URI | Content |
|---|---|
| `ragtune://docs/{name}` | `docs/<name>.md`; `ragtune://docs/README` is the repo README |
| `ragtune://specs/{name}` | `specs/<name>.md` |
| `ragtune://config/defaults`, `ragtune://config/prompts` | runtime defaults and prompt templates |
| `ragtune://budget/default-config` | default cost-estimation config with source citations |
| `ragtune://config/template` | the `ragtune init` starter config |
| `ragtune://registry` | registered component names per category |
