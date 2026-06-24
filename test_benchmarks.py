# -*- coding: utf-8 -*-
"""test_benchmarks.ipynb

End-to-end smoke tests for all RAGtune benchmarks.
Run on Colab (GPU runtime recommended) or any machine with internet access.

Benchmarks covered:
  1. FreshStack  — technical IR on GitHub / StackOverflow corpora
  2. CoIR        — code-domain retrieval (Stack Overflow, CoSQA, …)
  3. BRIGHT      — reasoning-intensive retrieval
  4. OBLIQ       — oblique-query retrieval (tip-of-tongue, analogue, descriptive)
  5. CRUMB       — passage retrieval across 8 domains
  6. SKILLRET    — skill / tool retrieval for agentic tasks

Each benchmark runs with a small slice of data to confirm the full pipeline
— data loading → indexing → RAGtune → evaluation — works end to end.

Original file location:
  https://colab.research.google.com/  (create a new notebook and paste this)
"""

# ── 0. Clone & setup ─────────────────────────────────────────────────────────

import os, sys

# Navigate to the repo root regardless of where Colab started.
# If we're already inside the repo (pyproject.toml present), stay here.
# If we're one level above (sir/ exists), cd into it.
# Otherwise clone from scratch.
if os.path.exists('pyproject.toml'):
    pass  # already inside sir/
elif os.path.exists('sir/pyproject.toml'):
    os.chdir('sir')
else:
    !git clone https://github.com/avishekanand/sir
    os.chdir('sir')

print(f"Working directory: {os.getcwd()}")

!git fetch -v -a

# ── 1. Install shared dependencies (done once, covers all benchmarks) ─────────

!git checkout bench/coir-integration
!git pull origin bench/coir-integration

!pip install -e '.[benchmarks]' -q

# Evaluation backend (RetrievalEvaluator)
!pip install pytrec_eval-terrier -q

# FreshStack-specific α-NDCG / Coverage metrics
!pip install freshstack -q

# GPU-accelerated FAISS — comment out if on a CPU-only runtime
!pip install faiss-gpu -q

# ── 2. Unit tests ─────────────────────────────────────────────────────────────

!pip install pytest -q
!pytest tests/unit/ -q --tb=short 2>&1 | tail -20

# ── 3–5. Benchmarks on bench/coir-integration ────────────────────────────────

import subprocess, os

def run(label, script, env=None):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    r = subprocess.run([sys.executable, script], env={**os.environ, **(env or {})},
                       capture_output=True, text=True)
    if r.stdout:
        print(r.stdout)
    if r.returncode != 0:
        print(f"STDERR:\n{r.stderr}")
        print(f"\n[FAIL] exit {r.returncode} — {label}")
    else:
        print(f"\n[PASS] {label}")
    return r.returncode == 0

# FreshStack — 1 domain, 5 queries
run("BENCHMARK 1/6 — FreshStack",
    "scripts/benchmark_freshstack.py",
    {"FRESHSTACK_DOMAINS": "langchain"})

# CoIR — 1 dataset, 5 queries
run("BENCHMARK 2/6 — CoIR",
    "scripts/benchmark_coir.py",
    {"COIR_DATASETS": "stackoverflow-qa", "COIR_QUERIES": "5"})

# BRIGHT — 1 domain, 3 queries
run("BENCHMARK 3/6 — BRIGHT",
    "scripts/benchmark_bright.py",
    {"BRIGHT_DOMAINS": "biology", "BRIGHT_QUERIES": "3"})

# ── 6. OBLIQ (rseetharaman/oblique-integration) ───────────────────────────────
#
# Env vars:
#   OBLIQ_TASKS    — comma-separated task names (congress math writing twitter wildchat)
#   OBLIQ_QUERIES  — queries per task

!git stash
!git checkout rseetharaman/oblique-integration
!git pull origin rseetharaman/oblique-integration

run("BENCHMARK 4/6 — OBLIQ",
    "scripts/benchmark_obliq.py",
    {"OBLIQ_TASKS": "congress", "OBLIQ_QUERIES": "5"})

# ── 7. CRUMB (rseetharaman/crumb-integration) ─────────────────────────────────
#
# Env vars:
#   CRUMB_TASKS    — comma-separated task names
#   CRUMB_QUERIES  — queries per task

!git stash
!git checkout rseetharaman/crumb-integration
!git pull origin rseetharaman/crumb-integration

run("BENCHMARK 5/6 — CRUMB",
    "scripts/benchmark_crumb.py",
    {"CRUMB_TASKS": "paper_retrieval", "CRUMB_QUERIES": "5"})

# ── 8. SKILLRET (rseetharaman/skillret-integration) ──────────────────────────
#
# Env vars:
#   SKILLRET_QUERIES — number of queries to evaluate

!git stash
!git checkout rseetharaman/skillret-integration
!git pull origin rseetharaman/skillret-integration

run("BENCHMARK 6/6 — SKILLRET",
    "scripts/benchmark_skillret.py",
    {"SKILLRET_QUERIES": "10"})

# ── 9. Summary ────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("All benchmark smoke tests finished.")
print("Check [PASS] / [FAIL] lines above for individual results.")
print(f"{'='*60}")
