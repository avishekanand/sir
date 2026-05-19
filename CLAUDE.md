# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install (recommended: uv for reproducible builds)
uv sync                              # Install from lock file (CPU torch)
uv sync --extra test                 # Include test dependencies

# Alternative: pip (no lock file — versions may drift)
pip install -e .
pip install -e ".[test]"

# GPU users: install CUDA torch first, then sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv sync

# Run tests
pytest                        # Full suite
pytest tests/unit             # Unit tests only
pytest tests/integration      # Integration tests

# Run demos via Makefile
make run-terrier              # PyTerrier BRIGHT demo
make run-langchain            # LangChain demo
make run-active-learning      # Active learning demo
make run-benchmarks           # Benchmark suite
make run SCRIPT=examples/quickstart.py  # Any script

# CLI usage
ragtune init                  # Create config template
ragtune list                  # Show registered components
ragtune validate config.yaml  # Validate config
ragtune run config.yaml -q "query"          # Execute pipeline
ragtune run config.yaml -q "query" --verbose  # With trace
```

## Architecture

RAGtune is budget-aware iterative RAG middleware. The core loop:

```
1. REFORMULATION: Initial retrieval + optional LLM query rewriting
2. ITERATIVE LOOP (while budget remains):
   Estimator.value() → Scheduler.select_batch() → Reranker.rerank() → Pool.update()
3. ASSEMBLY: GreedyAssembler selects final docs within token budget
```

**Key abstractions:**
- `RAGtuneController` (`core/controller.py`) - Orchestrates the loop
- `CandidatePool` (`core/pool.py`) - Document state machine: CANDIDATE → IN_FLIGHT → RERANKED
- `CostTracker` (`core/budget.py`) - Enforces limits (tokens, rerank_docs, latency_ms, etc.)
- `RAGtuneContext` (`core/types.py`) - Unified context passed to all components

**Registry system:**
```python
from ragtune.registry import registry

@registry.retriever("my-retriever")
class MyRetriever(BaseRetriever):
    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        ...
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/ragtune/core/` | Controller, budget, pool, types, interfaces |
| `src/ragtune/components/` | Retrievers, rerankers, schedulers, estimators, assemblers |
| `src/ragtune/adapters/` | LangChain, LlamaIndex, PyTerrier integrations |
| `src/ragtune/cli/` | Typer-based CLI (main.py, config_loader.py) |
| `src/ragtune/config/` | defaults.yaml, prompts.yaml, Pydantic models |
| `tests/unit/`, `tests/integration/` | Test suites mirroring src/ structure |
| `examples/` | Demo scripts for various scenarios |

## Configuration

Declarative YAML configs define the pipeline:

```yaml
pipeline:
  budget:
    limits:
      tokens: 4000
      rerank_docs: 50
      latency_ms: 2000
  components:
    retriever: { type: "pyterrier" }
    reranker: { type: "cross-encoder" }
    scheduler: { type: "active-learning", params: { batch_size: 5 } }
    estimator: { type: "similarity" }
    assembler: { type: "greedy" }
```

ConfigLoader singleton (`src/ragtune/utils/config.py`) provides access via dot notation: `config.get("retrieval.num_reformulations")`

## Testing Patterns

- Fake components for unit tests (see `tests/unit/core/test_controller_loop.py`)
- Fixtures in `tests/conftest.py`
- Test files mirror module paths: `src/ragtune/core/pool.py` → `tests/unit/core/test_pool.py`

## PR & Code Review Conventions

When opening or reviewing PRs in this repo, enforce these rules (full text in the project wiki page "PR & Code Review Guidelines"):

1. **One PR = one problem.** No mixed refactor + feature, no drive-by edits to unrelated files. Open a new Issue for anything discovered along the way.
2. **PR description is mandatory** and must answer the four questions:
   - What problem does this solve? (link Issue + spec)
   - What changed and why? (design choices)
   - How was it tested? (paste pytest output, list test names)
   - What should the reviewer focus on?
3. **Size matters.** <200 ideal, 200–300 acceptable, 300–500 must justify, >500 split. No exceptions for the >500 line.
4. **Tests travel with code.** New component → unit test. Modified controller/adapter → integration test. Budget/cost logic → budget-exhaustion edge case test (required). CLI change → smoke test via `make run-*`. Use fakes from `tests/conftest.py`; never real API calls in unit tests.
5. **Reviewer comments use prefixes:** `[block]`, `[nit]`, `[question]`, `[idea]`. Only `[block]` gates approval.
6. **Branch names signal scope:** `feat/...`, `fix/...`, `docs/...`, `bench/...`, `test/...`, `refactor/...`.
7. **Commit format:** `<type>(<scope>): <imperative description>` — same prefix vocabulary as branches. One commit per logical change.
8. **Spec precedes code.** Any new component, integration, or significant behavior change requires a spec merged into `specs/` *before* the implementation PR. Exception: `fix/` branches on existing well-understood components.

### Copilot Code Review

GitHub auto-attaches `copilot-pull-request-reviewer[bot]` to every PR in this repo. The bot reviews **once on PR open** and does **not** re-review when you push follow-up commits — its inline comments stay frozen even after they're addressed. Two consequences:

- **After you push fixes for Copilot's comments**, open the PR in the web UI → in the "Reviewers" sidebar, click the 🔄 **Re-request review** icon next to `copilot-pull-request-reviewer`. This triggers a fresh pass. Without it, Copilot's stale comments give a misleading impression that issues remain.
- **Copilot comments are advisory only.** Only `[block]` from a human reviewer gates approval (Rule 5). A Copilot comment left unaddressed isn't a blocker — but if you choose not to address one, say so in a reply so the human reviewer doesn't have to second-guess.

There is no project CI, so no automated re-review trigger exists; the manual button is the only mechanism.

## See Also

See `AGENTS.md` for coding style, naming conventions, commit guidelines, and PR process.
