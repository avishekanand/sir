# -*- coding: utf-8 -*-
"""test_benchmarks.ipynb

End-to-end smoke tests for all RAGtune benchmarks.
Run on Colab (GPU runtime recommended) or any machine with internet access.

Benchmarks covered:
  1. FreshStack  — technical IR on GitHub / StackOverflow corpora
  2. CoIR        — code-domain retrieval (Stack Overflow, CoSQA, …)
  3. BRIGHT      — reasoning-intensive retrieval

Each benchmark runs with a small slice of data (1 domain / dataset, few queries)
to confirm the full pipeline — data loading → indexing → RAGtune → evaluation — works.

Original file location:
  https://colab.research.google.com/  (create a new notebook and paste this)
"""

# ── 0. Clone & configure ────────────────────────────────────────────────────

!git clone https://github.com/avishekanand/sir
# %cd sir

!git fetch -v -a
!git checkout bench/coir-integration
!git pull origin bench/coir-integration

# ── 1. Install dependencies ─────────────────────────────────────────────────

# Core + benchmark extras
!pip install -e '.[benchmarks]' -q

# Evaluation backend (needed by CoIR RetrievalEvaluator)
!pip install pytrec_eval-terrier -q

# FreshStack-specific evaluator (α-NDCG, Coverage metrics)
!pip install freshstack -q

# GPU-accelerated FAISS — replaces faiss-cpu from pyproject.toml
# Comment out if running on a CPU-only runtime.
!pip install faiss-gpu -q

# HuggingFace ecosystem (embeddings + dataset downloads)
!pip install datasets sentence-transformers -q

# ── 2. Sanity-check: unit tests ─────────────────────────────────────────────

!pip install pytest -q
!pytest tests/unit/ -q --tb=short 2>&1 | tail -20

# ── 3. FreshStack benchmark (smoke: 1 domain, 5 queries) ────────────────────
#
# Full run: remove env var overrides.
# Env vars:
#   FRESHSTACK_DOMAINS  — comma-separated topic names (langchain yolo angular laravel godot)

import subprocess, os

print("\n" + "=" * 60)
print("BENCHMARK 1 / 3 — FreshStack")
print("=" * 60)

env_freshstack = {
    **os.environ,
    "FRESHSTACK_DOMAINS": "langchain",   # 1 domain
}

result = subprocess.run(
    ["python", "scripts/benchmark_freshstack.py"],
    env=env_freshstack,
    capture_output=False,   # stream output live
)

if result.returncode != 0:
    print(f"\n[FAIL] benchmark_freshstack.py exited with code {result.returncode}")
else:
    print("\n[PASS] FreshStack benchmark complete")

# ── 4. CoIR benchmark (smoke: 1 dataset, 5 queries) ─────────────────────────
#
# Full run: remove env var overrides.
# Env vars:
#   COIR_DATASETS  — comma-separated dataset names
#   COIR_QUERIES   — queries per dataset

print("\n" + "=" * 60)
print("BENCHMARK 2 / 3 — CoIR")
print("=" * 60)

env_coir = {
    **os.environ,
    "COIR_DATASETS": "stackoverflow-qa",   # 1 dataset
    "COIR_QUERIES": "5",
}

result = subprocess.run(
    ["python", "scripts/benchmark_coir.py"],
    env=env_coir,
    capture_output=False,
)

if result.returncode != 0:
    print(f"\n[FAIL] benchmark_coir.py exited with code {result.returncode}")
else:
    print("\n[PASS] CoIR benchmark complete")

# ── 5. BRIGHT benchmark (smoke: default 3 domains, 5 queries each) ──────────
#
# The script hardcodes DOMAINS and QUERIES_PER_DOMAIN.
# To run a single domain, edit benchmark_bright.py or patch inline:

print("\n" + "=" * 60)
print("BENCHMARK 3 / 3 — BRIGHT")
print("=" * 60)

# Patch the domain list to a single domain for a quick smoke test.
# Remove this block to run all three domains.
!python - << 'PATCH'
import re, pathlib
p = pathlib.Path("scripts/benchmark_bright.py")
src = p.read_text()
src = re.sub(
    r'DOMAINS\s*=\s*\[.*?\]',
    'DOMAINS = ["biology"]',
    src,
)
src = re.sub(
    r'QUERIES_PER_DOMAIN\s*=\s*\d+',
    'QUERIES_PER_DOMAIN = 3',
    src,
)
p.write_text(src)
print("Patched benchmark_bright.py: DOMAINS=['biology'], QUERIES_PER_DOMAIN=3")
PATCH

result = subprocess.run(
    ["python", "scripts/benchmark_bright.py"],
    capture_output=False,
)

if result.returncode != 0:
    print(f"\n[FAIL] benchmark_bright.py exited with code {result.returncode}")
else:
    print("\n[PASS] BRIGHT benchmark complete")

# ── 6. Summary ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("All benchmark smoke tests finished.")
print("Check [PASS] / [FAIL] lines above for status.")
print("=" * 60)
