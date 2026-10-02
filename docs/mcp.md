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

## Resources

| URI | Content |
|---|---|
| `ragtune://docs/{name}` | `docs/<name>.md`; `ragtune://docs/README` is the repo README |
| `ragtune://specs/{name}` | `specs/<name>.md` |
| `ragtune://config/defaults`, `ragtune://config/prompts` | runtime defaults and prompt templates |
| `ragtune://budget/default-config` | default cost-estimation config with source citations |
| `ragtune://registry` | registered component names per category |
