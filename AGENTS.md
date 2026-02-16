# Repository Guidelines

## Project Structure & Module Organization
- Core logic lives in `src/ragtune` (`core/controller.py`, `core/budget.py`, `registry.py`, and adapters under `adapters/`).
- Shared utilities are in `src/ragtune/utils`, and CLI entrypoint is `ragtune.cli.main` (exposed via `ragtune` script).
- Tests are split into `tests/unit` and `tests/integration`; benchmarks and fixtures sit under `tests/benchmarks` and `data/`.
- Examples for demos are under `examples/`, while automation and experiments live in `scripts/`; long-form docs are in `docs/` and specs in `specs/`.

## Data & Benchmarks Setup
- Benchmark inputs live in `data/`; keep raw downloads out of version control and document any external dataset URLs in PRs.
- Benchmarks under `tests/benchmarks` and `scripts/benchmark_suite.py` expect dataset paths—add synthetic samples to keep CI light.
- When adding new corpora, prefer a lightweight loader helper in `scripts/` or `data/README.md` and gate expensive paths behind flags/env vars (e.g., `DATA_DIR`, `BENCH_FAST=1`).

## Build, Test, and Development Commands
- Install: `pip install -e .` (run inside an activated `venv`).
- Quick demos: `make run-terrier`, `make run-langchain`, or `make run-active-learning` to run packaged example pipelines; use `make run SCRIPT=examples/<file>.py` for any script.
- Benchmarks: `make run-benchmarks` executes `scripts/benchmark_suite.py`.
- Tests: `pytest` runs the full suite; scope to `pytest tests/unit` or `pytest tests/integration` when iterating.
- Cleanup: `make clean` removes local log artifacts.

## Coding Style & Naming Conventions
- Python 3.9+ with 4-space indents and type hints everywhere; favor clear dataclasses/Pydantic models for configs and responses.
- Follow Black formatting (`python -m black .`); keep imports tidy and avoid unused symbols.
- Module/file names are `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`; functions/variables `lower_snake`.
- Keep CLI help and docstrings current when touching `ragtune.cli` or public APIs.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` for async controllers; place fixtures in `tests/conftest.py`.
- Name tests `test_<feature>.py` with functions `test_<behavior>`; mirror module paths between `src/` and `tests/`.
- Add focused unit tests alongside new logic; add integration tests when touching controllers, adapters, or registry flows.
- Capture cost/latency edge cases and error handling (budget exhaustion, missing adapters) before merging.

## Commit & Pull Request Guidelines
- Use concise, scoped commits with prefixes seen in history (`feat`, `docs`, etc.), optionally with a scope (e.g., `feat(cli): ...`).
- PRs should describe intent, key changes, and verification (`pytest` output and relevant `make run-*` demos); link issues or specs when applicable.
- Include new configs/scripts in examples or docs when introducing user-facing behavior, and note any breaking changes explicitly.
