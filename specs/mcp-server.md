This is the executable specification for the **RAGtune MCP server** (`ragtune.mcp`): a Model Context Protocol server that lets an LLM agent (Claude Code, Claude Desktop, any MCP client) use every capability of this repository through typed tools, without writing Python.

### **1. Goals**

1. **Complete coverage.** Every public capability (CLI commands, controller, registry, configs, data loaders, indexers, evaluator, budget/cost system, benchmark scripts, tests, docs) is reachable through a tool or resource. Section 6 is the checklist.
2. **Agent-friendly.** Typed JSON-schema inputs, structured JSON outputs, errors that explain how to fix the call, expensive objects (models, datasets) loaded once and referenced by handle.
3. **Safe by default.** File access is confined to a workspace root; only repo scripts and `pytest` can be executed.

### **2. Runtime**

- SDK: `mcp>=2.2,<3` (`MCPServer`), installed via the optional extra `pip install -e ".[mcp]"` (requires Python >= 3.10; the core package keeps `>=3.9`).
- Entry points: `ragtune-mcp` console script and `python -m ragtune.mcp`. Options: `--root PATH` (default: `$RAGTUNE_MCP_ROOT` or the current directory), `--transport stdio|streamable-http`, `--host`, `--port`.
- stdio safety: the SDK diverts fd 1 to stderr while serving, so prints from RAGtune, Rich or the PyTerrier JVM cannot corrupt the protocol stream.
- Sync tools run in worker threads (SDK default), so a long pipeline run never blocks other requests.

### **3. Conventions**

- **Errors.** Every tool converts exceptions to `ToolError("<Type>: <message>")`. The SDK hides the text of any other exception, and agents need it to self-correct (e.g. "Unknown component type 'foo'. Available: [...]").
- **Paths.** Relative paths resolve against the workspace root; a path that resolves outside it is rejected.
- **Handles.** `create_pipeline` returns `pipeline_id`, `load_dataset` returns `dataset_id`, background work returns `job_id`. Handles live for the server process.
- **Background work.** Tools that may take minutes (`load_dataset`, `build_index`, `evaluate_pipeline`, `evaluate_scenarios`) accept `background=True` and return a `job_id`; `job_status` returns the result when done. Subprocess jobs (`run_script`, `run_tests`, `run_cli`) are always background.
- **Annotations.** Read-only tools set `readOnlyHint`; tools that overwrite or delete set `destructiveHint`.
- **Config input.** Tools taking a pipeline config accept either `config_path` or an inline `config` (dict or YAML string).

### **4. Tools**

| Group | Tools |
|---|---|
| Discovery | `server_info`, `list_components`, `describe_component`, `list_benchmarks`, `get_settings`, `set_setting`, `list_default_scenarios` |
| Config | `config_template`, `read_config`, `write_config`, `update_config`, `validate_config`, `visualize_config` |
| Pipelines | `create_pipeline`, `run_pipeline`, `list_pipelines`, `close_pipeline` |
| Evaluation | `evaluate_run`, `evaluate_pipeline`, `evaluate_scenarios` |
| Data | `load_dataset`, `list_datasets`, `get_queries`, `get_documents`, `get_qrels`, `export_corpus`, `drop_dataset` |
| Indexing | `build_index`, `build_index_from_config`, `index_status`, `search_index` |
| Budget | `estimate_cost`, `compare_costs`, `validate_budget_config`, `budget_reference`, `estimate_hardware`, `estimate_throughput`, `cost_history` |
| Jobs | `list_scripts`, `run_script`, `run_tests`, `run_cli`, `job_status`, `list_jobs`, `cancel_job` |

Notable behaviors:
- `create_pipeline` accepts `limit_overrides` (`None` removes a limit), `documents` (inline corpus, injects the `in-memory` retriever), `index_retriever` (`{index_type, index_path, indexer_params, dataset_id}`: retrieval through any built index, with document text filled from a loaded dataset; this is the generalized retriever cassette from `scripts/run_tool_retrieval.py`), and `cost_estimation` (`{budget_type, config, config_path, cost_config}`: attaches a budget loader as the controller's `cost_loader`).
- `run_pipeline` takes a `pipeline_id`, or a config (it creates and registers the pipeline and returns its id for reuse). Optional per-run `limit_overrides` and `include_trace`.
- `evaluate_pipeline` ranks results like the benchmark runner (`1/(rank+1)`) and reports metrics plus mean budget usage per key.
- `load_dataset` routes `crumb`, `obliq`, `irds`, `hf` and `local` (JSON/JSONL corpus, JSONL queries, TSV/JSONL qrels) to their loaders, and everything else through `DataLoaderFactory` (BRIGHT, BEIR, FreshStack, ToolRet, SkillRet, SRA-Bench).
- `get_qrels(kind=...)` also serves loader extras: `excluded_ids` (BRIGHT, OBLIQ) and `nugget_qrels` (FreshStack).
- `run_script` accepts only files under `scripts/` or `examples/`; `run_tests` only paths under `tests/`; `run_cli` runs `python -m ragtune.cli.main` with stdin closed, so interactive prompts fail fast instead of hanging.

### **5. Resources and prompts**

- `ragtune://docs/{name}`, `ragtune://specs/{name}`: `docs/*.md`, `README.md`, `specs/*.md` by file stem.
- `ragtune://config/template`, `ragtune://config/defaults`, `ragtune://config/prompts`, `ragtune://budget/default-config`, `ragtune://registry`.
- Prompts: `build_pipeline(goal)`, `benchmark_pipeline(benchmark, dataset)`, `estimate_deployment_cost(workload)`.

### **6. Coverage checklist**

| Repository capability | MCP surface |
|---|---|
| `ragtune list` / `registry.list_all()` / `get_*` | `list_components`, `describe_component`, `ragtune://registry` |
| `ragtune init` (template), `--wizard` | `config_template`; the wizard's choices map to `update_config` |
| `ragtune validate` | `validate_config` |
| `ragtune visualize` (+ `--edit`) | `visualize_config` (+ `update_config` diff) |
| `ragtune run` (`--limit`, `--verbose`) | `run_pipeline` (`limit_overrides`, `include_trace`). `--collection-path` only rewrites `pipeline.data`, which `run` never reads; `update_config` covers it |
| `ragtune index` | `build_index_from_config` |
| `ragtune budget` (all flags) | `estimate_cost` (+ `suggest`, `thresholds`) |
| any CLI command verbatim | `run_cli` |
| `ConfigLoader.create_controller` / `create_controllers_from_env` / `_default_scenarios` | `create_pipeline`, `evaluate_scenarios`, `list_default_scenarios` |
| `RAGtuneController.run` (`override_budget`, `cost_loader`, `cost_config`, `initial_top_k`) | `run_pipeline`, `create_pipeline` |
| `utils.config` `get` / `set` / `get_prompt` | `get_settings`, `set_setting` |
| `PyTerrierRetriever(index_path)` | pipeline config (`type: pyterrier`) or `index_retriever` |
| All `BaseDataLoader` subclasses, `DataLoaderFactory`, `RetrieverDataset`, `BRIGHTMultiTaskLoader` | `load_dataset` (one call per task) + `get_*`, `export_corpus` |
| `get_excluded_ids()`, `load_nugget_qrels()`, `Query.reasoning` | `get_qrels(kind=...)`, `get_queries` |
| `IndexFactory.create` / `from_config`, `build_from_corpus` / `build`, `exists`, `search`, `FlexIndexer.get_retriever(backend)` | `build_index`, `build_index_from_config`, `index_status`, `search_index(backend=...)` |
| `RetrievalEvaluator.evaluate` / `evaluate_custom`, `evaluate_run` | `evaluate_run(metrics=...)`, `evaluate_pipeline` |
| `calculate_budget`, `format_report`, `BudgetLoaderFactory.create` / `list_types` | `estimate_cost`, `budget_reference` |
| `BudgetConfig.validate` / `to_dict` | `validate_budget_config` |
| `suggest_optimizations`, `check_alerts` | `estimate_cost(suggest=..., thresholds=...)`, `compare_costs` |
| `CostHistoryLogger` log / query / summary / clear | `estimate_cost(log_to=...)`, `cost_history` |
| `hardware.*` (GPU specs, GPU/CPU/system power, energy, carbon) | `estimate_hardware`, `budget_reference` |
| `throughput.*` (peak/achieved throughput, saturation knee, profiles, VRAM) | `estimate_throughput`, `budget_reference` |
| Loader pricing tables (embedding, reranking, token, regional carbon) | `budget_reference` |
| `scripts/*.py`, `scripts/evaluation/*.py`, `examples/*.py` | `list_scripts`, `run_script` |
| test suite | `run_tests` |
| `docs/`, `specs/`, `README.md`, YAML defaults and prompts | resources |

**Out of scope:** adapters that wrap live Python objects (`LangChainRetriever`, `LlamaIndexRetriever`, `RAGtuneLangChainAdapter`, `RAGtuneTransformer`, a PyTerrier transformer object): MCP arguments are JSON, so there is no object to pass. They stay available to Python callers.

### **7. Verification checklist**

- [ ] Every tool in section 4 is listed by an in-memory `Client(build_server(...))` and has an input schema.
- [ ] Each group has tests for success and for the documented error (bad component, path outside root, unknown handle, non-whitelisted script).
- [ ] An end-to-end test builds an index from an inline corpus, creates a pipeline over it, runs a query and evaluates it, with no network access.
- [ ] A stdio smoke test launches `python -m ragtune.mcp` as a subprocess and completes `initialize` + `tools/list`.
