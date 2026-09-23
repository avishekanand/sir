#!/usr/bin/env python
"""
End-to-end BEIR benchmark for RAGtune — script form of examples/ragtune_benchmark.ipynb.

Sweeps a grid of (BEIR dataset x retriever x seed) cells.  Each cell runs up to
three stages:

  sanity    Raw retriever NDCG@10 (no RAGtune loop) vs published BEIR numbers.
            Catches broken indexes / query parsing before any expensive tuning.
  baseline  One fixed RAGtune pipeline per (reranker, checkpoint) — the untuned
            reference that tuning has to beat.
  tune      Bayesian TPE (Optuna), the LLM/GEPA agent, and/or random search
            optimizing the pipeline around the fixed retriever, multi-objective
            on (NDCG@10 up, mean rerank docs down).

Coverage
--------
  datasets   26 BEIR tasks (14 standard + 12 CQADupStack subforums), every ID
             verified against the ir_datasets registry
  retrievers 5 lexical (BM25/DPH/PL2/DirichletLM/TF-IDF), 8 dense bi-encoders,
             1 RRF hybrid
  rerankers  noop, CrossEncoder (4 checkpoints), MonoT5 (2), pointwise LLM
  optimizers bayes (TPE), llm (GEPA agent), random (control arm)

Everything is written incrementally to --out-dir, so a killed run resumes with
--resume, and --report-only rebuilds the tables from state.json alone.

Quick start
-----------
    # size the run before launching it — prints the grid and a cost estimate
    python examples/beir_full_benchmark.py --preset exhaustive --plan

    # ~20 min CPU smoke test: nfcorpus+scifact, BM25+DPH, noop+cross-encoder
    python examples/beir_full_benchmark.py --preset smoke

    # the full GPU sweep (multi-day — check --plan first)
    python examples/beir_full_benchmark.py --preset exhaustive --device cuda

    # everything tunable, printed from the live search space
    python examples/beir_full_benchmark.py --list-knobs

Requirements
------------
    pip install -e ".[tuning]"
    pip install python-terrier ir-datasets optuna sentence-transformers matplotlib
    pip install pyterrier-t5        # only for --rerankers monot5
    pip install pyterrier-dr        # only for --retrievers dense-* / hybrid-rrf
    export OPENAI_API_KEY=...       # only for --optimizers llm / --space full
    A JDK (not just a JRE) on PATH or in the conda env — pyjnius resolves javac.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Run straight from a checkout, without `pip install -e .` (repo convention).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# ══════════════════════════════════════════════════════════════════════════════
# Catalogues
# ══════════════════════════════════════════════════════════════════════════════

# BEIR subsets that are small enough to index and tune on a single machine.
# ndcg10_bm25 = published Anserini BM25 (k1=0.9, b=0.4) from the BEIR paper,
# used only as a sanity reference — Terrier's tokenizer differs, so +/-0.05 is
# normal.
DATASETS: Dict[str, Dict[str, Any]] = {
    # Every BEIR task ir_datasets registers with both queries and qrels, verified
    # against the local ir_datasets registry (0.5.5).  doc/query counts are exact.
    # ndcg10_bm25 is the BEIR paper's Anserini BM25 (k1=0.9, b=0.4) figure, kept
    # only as an order-of-magnitude sanity reference — Terrier's tokenizer and
    # stemmer differ, so a gap of +/-0.05 is unremarkable.  None = no published
    # figure to compare against.
    # heavy=True marks corpora where indexing alone is hours and dense encoding
    # is a GPU-day; they are excluded from --preset exhaustive unless --allow-heavy.
    "nfcorpus":     {"irds_id": "irds:beir/nfcorpus/test",       "n_docs": 3_633,     "n_queries": 323,   "ndcg10_bm25": 0.325, "heavy": False},
    "scifact":      {"irds_id": "irds:beir/scifact/test",        "n_docs": 5_183,     "n_queries": 300,   "ndcg10_bm25": 0.665, "heavy": False},
    "arguana":      {"irds_id": "irds:beir/arguana",             "n_docs": 8_674,     "n_queries": 1_406, "ndcg10_bm25": 0.315, "heavy": False},
    "scidocs":      {"irds_id": "irds:beir/scidocs",             "n_docs": 25_657,    "n_queries": 1_000, "ndcg10_bm25": 0.158, "heavy": False},
    "fiqa":         {"irds_id": "irds:beir/fiqa/test",           "n_docs": 57_638,    "n_queries": 648,   "ndcg10_bm25": 0.236, "heavy": False},
    "trec-covid":   {"irds_id": "irds:beir/trec-covid",          "n_docs": 171_332,   "n_queries": 50,    "ndcg10_bm25": 0.656, "heavy": False},
    "touche2020":   {"irds_id": "irds:beir/webis-touche2020/v2", "n_docs": 382_545,   "n_queries": 49,    "ndcg10_bm25": 0.367, "heavy": False},
    "quora":        {"irds_id": "irds:beir/quora/test",          "n_docs": 522_931,   "n_queries": 10_000,"ndcg10_bm25": 0.789, "heavy": False},
    "nq":           {"irds_id": "irds:beir/nq",                  "n_docs": 2_681_468, "n_queries": 3_452, "ndcg10_bm25": 0.329, "heavy": True},
    "dbpedia":      {"irds_id": "irds:beir/dbpedia-entity/test", "n_docs": 4_635_922, "n_queries": 400,   "ndcg10_bm25": 0.313, "heavy": True},
    "hotpotqa":     {"irds_id": "irds:beir/hotpotqa/test",       "n_docs": 5_233_329, "n_queries": 7_405, "ndcg10_bm25": 0.603, "heavy": True},
    "fever":        {"irds_id": "irds:beir/fever/test",          "n_docs": 5_416_568, "n_queries": 6_666, "ndcg10_bm25": 0.753, "heavy": True},
    "climate-fever":{"irds_id": "irds:beir/climate-fever",       "n_docs": 5_416_593, "n_queries": 1_535, "ndcg10_bm25": 0.213, "heavy": True},
    "msmarco":      {"irds_id": "irds:beir/msmarco/dev",         "n_docs": 8_841_823, "n_queries": 6_980, "ndcg10_bm25": 0.228, "heavy": True},
}

# CQADupStack is 12 independent StackExchange subforums; BEIR reports the mean
# over all of them (BM25 0.299).  Each is small, so the whole family is tractable.
_CQA = {
    "android": 22_998, "english": 40_221, "gaming": 45_301, "gis": 37_637,
    "mathematica": 16_705, "physics": 38_316, "programmers": 32_176, "stats": 42_269,
    "tex": 68_184, "unix": 47_382, "webmasters": 17_405, "wordpress": 48_605,
}
for _sub, _n in _CQA.items():
    DATASETS[f"cqadupstack-{_sub}"] = {
        "irds_id": f"irds:beir/cqadupstack/{_sub}",
        "n_docs": _n, "n_queries": None, "ndcg10_bm25": 0.299, "heavy": False,
    }

# BEIR's signal1m / trec-news / robust04 / bioasq need licensed corpora and are
# NOT in the public ir_datasets registry — verified absent, so they are omitted
# rather than listed and left to fail at run time.

DATASET_GROUPS: Dict[str, List[str]] = {
    "tiny":     ["nfcorpus", "scifact"],
    "small":    ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"],
    "standard": ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa", "trec-covid", "touche2020"],
    "cqadupstack": [k for k in DATASETS if k.startswith("cqadupstack-")],
    "heavy":    [k for k, v in DATASETS.items() if v["heavy"]],
    "all":      list(DATASETS),
}

# Retrievers.  Lexical models are CPU-only and parameter-free (bar BM25); dense
# models are small bi-encoders run through pyterrier-dr's FlexIndex.  encoder_kwargs
# are forwarded to GenericHFEncoder, which is what FlexIndexer falls back to for a
# raw HF id (see ragtune/indexing/encoders/__init__.py:resolve_encoder).
RETRIEVERS: Dict[str, Dict[str, Any]] = {
    # ── Lexical (Terrier weighting models) ────────────────────────────────────
    "bm25":      {"kind": "sparse", "wmodel": "BM25",        "desc": "Terrier BM25, BEIR-tuned k1=0.9 b=0.4"},
    "dph":       {"kind": "sparse", "wmodel": "DPH",         "desc": "Terrier DPH — parameter-free DFR model"},
    "pl2":       {"kind": "sparse", "wmodel": "PL2",         "desc": "Terrier PL2 — Poisson DFR model"},
    "dirichlet": {"kind": "sparse", "wmodel": "DirichletLM", "desc": "Terrier Dirichlet-smoothed language model"},
    "tfidf":     {"kind": "sparse", "wmodel": "TF_IDF",      "desc": "Terrier TF-IDF — weakest lexical baseline"},

    # ── Dense bi-encoders (shorthands resolve via pyterrier_dr built-ins) ─────
    "dense-minilm": {"kind": "dense", "model": "sentence-transformers/all-MiniLM-L6-v2",
                     "encoder_kwargs": {"pooling": "mean", "max_length": 256},
                     "desc": "all-MiniLM-L6-v2 (22M) — cheapest dense baseline"},
    "dense-mpnet":  {"kind": "dense", "model": "sentence-transformers/all-mpnet-base-v2",
                     "encoder_kwargs": {"pooling": "mean", "max_length": 384},
                     "desc": "all-mpnet-base-v2 (110M) — stronger general-purpose ST model"},
    "dense-bge":    {"kind": "dense", "model": "BAAI/bge-small-en-v1.5",
                     "encoder_kwargs": {"pooling": "cls", "max_length": 512,
                                        "query_prefix": "Represent this sentence for searching relevant passages: "},
                     "desc": "bge-small-en-v1.5 (33M) — CLS pooling + query instruction prefix"},
    "dense-gte":    {"kind": "dense", "model": "thenlper/gte-small",
                     "encoder_kwargs": {"pooling": "mean", "max_length": 512},
                     "desc": "gte-small (33M)"},
    "dense-e5":     {"kind": "dense", "model": "intfloat/e5-small-v2",
                     "encoder_kwargs": {"pooling": "mean", "max_length": 512,
                                        "query_prefix": "query: ", "doc_prefix": "passage: "},
                     "desc": "e5-small-v2 (33M) — requires query:/passage: prefixes"},
    "dense-tasb":   {"kind": "dense", "model": "tasb",  "desc": "TAS-B (pyterrier_dr built-in), MS MARCO-distilled"},
    "dense-ance":   {"kind": "dense", "model": "ance",  "desc": "ANCE (pyterrier_dr built-in)"},
    "dense-tct":    {"kind": "dense", "model": "tct",   "desc": "TCT-ColBERT (pyterrier_dr built-in)"},

    # ── Hybrid ────────────────────────────────────────────────────────────────
    # Reciprocal-rank fusion of a lexical and a dense run.  Components are set
    # with --hybrid-sparse / --hybrid-dense.
    "hybrid-rrf": {"kind": "hybrid", "desc": "RRF fusion of --hybrid-sparse and --hybrid-dense runs"},
}

RETRIEVER_GROUPS: Dict[str, List[str]] = {
    "lexical": ["bm25", "dph", "pl2", "dirichlet", "tfidf"],
    "dense":   ["dense-minilm", "dense-bge", "dense-gte", "dense-e5", "dense-tasb"],
    "small":   ["bm25", "dph"],
    "standard": ["bm25", "dph", "dense-minilm", "dense-bge"],
    "all":     [k for k in RETRIEVERS if k != "hybrid-rrf"] + ["hybrid-rrf"],
}

# Reranker registry types.  Which checkpoints each type may use is a separate
# axis (--ce-models / --monot5-models), because the search space treats model
# choice as its own conditional parameter.
RERANKERS: Dict[str, Dict[str, Any]] = {
    "noop":          {"desc": "No reranking — retriever order passes straight through"},
    "cross-encoder": {"desc": "SentenceTransformers CrossEncoder (see --ce-models)"},
    "monot5":        {"desc": "MonoT5 seq2seq reranker via pyterrier-t5 (see --monot5-models)"},
    "llm":           {"desc": "Pointwise LLM reranker via litellm — one API call per document"},
}

CE_MODEL_MENU = {
    "tiny":     ["cross-encoder/ms-marco-MiniLM-L-2-v2"],
    "standard": ["cross-encoder/ms-marco-MiniLM-L-6-v2"],
    "all":      ["cross-encoder/ms-marco-MiniLM-L-2-v2",
                 "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 "cross-encoder/ms-marco-MiniLM-L-12-v2",
                 "BAAI/bge-reranker-base"],
}
MONOT5_MODEL_MENU = {
    "standard": ["castorini/monot5-base-msmarco"],
    "all":      ["castorini/monot5-base-msmarco", "castorini/monot5-large-msmarco"],
}

# Knobs that are NOT swept by this script but are exposed by the codebase.
# Printed by --list-knobs.
EXTRA_KNOBS: List[tuple] = [  # (scope, knob, description)
    # ── Retriever level (fixed per cell — this script sweeps them as a grid) ──
    ("retriever", "wmodel", "BM25 / DPH / TF_IDF / PL2 / DirichletLM / BM25F — any Terrier weighting model (--retrievers)"),
    ("retriever", "bm25_k1 / bm25_b", "BM25 term saturation + length normalisation (--bm25-k1 / --bm25-b). BEIR default 0.9 / 0.4; Terrier default 1.2 / 0.75"),
    ("retriever", "num_results", "Candidate depth returned by the first-stage retriever (--retrieval-depth). Hard ceiling on what reranking can rescue"),
    ("retriever", "dense encoder", "Any HF bi-encoder via ragtune.indexing.FlexIndexer (qwen3 / bge-m3 / tasb / ance / tct / raw HF id)"),
    ("retriever", "dense backend", "FlexIndex search backend: np (exact) / torch / faiss_flat / faiss_hnsw (approx, faster)"),
    ("retriever", "query field", "BEIR topics often carry title + narrative (trec-covid); EvalDataset.from_pyterrier_irds prefers the short 'query' field"),
    # ── Search space: discrete menus (RAGtuneSearchSpace) ────────────────────
    ("pipeline", "reformulator_types", "identity / llm_rewrite / reformir — query rewriting. Off by default here (needs an LLM key and costs budget)"),
    ("pipeline", "estimator_types", "baseline / utility / similarity / reformir / composite — the value model driving which docs get reranked next"),
    ("pipeline", "scheduler_types", "active-learning / graceful-degradation — batch selection policy"),
    ("pipeline", "feedback_types", "none / budget-stop / reformir-convergence — early-stop controller"),
    ("pipeline", "assembler max_docs", "GreedyAssembler cutoff (3–20). NOTE: values < 10 mathematically cap NDCG@10"),
    # ── Search space: model menus ────────────────────────────────────────────
    ("models", "ce_models", "cross-encoder/ms-marco-MiniLM-L-6-v2 (default) or -L-12-v2 (2x slower, ~+0.01 NDCG)"),
    ("models", "monot5_models", "castorini/monot5-base-msmarco / monot5-large-msmarco"),
    ("models", "monot5_batch_size", "4 / 8 / 16 / 32 — throughput vs GPU memory"),
    ("models", "similarity_models", "all-MiniLM-L6-v2 / all-mpnet-base-v2 — only used by the 'similarity' estimator"),
    ("models", "reformulator_models", "gpt-4o-mini / gpt-4o — only used by llm_rewrite / reformir"),
    # ── Budget / cost ────────────────────────────────────────────────────────
    ("budget", "rerank_docs", "Docs the reranker may score per query (5–200). This is the cost axis of the Pareto front"),
    ("budget", "reformulations", "Number of query rewrites allowed (0–5)"),
    ("budget", "tokens / latency_ms / retrieval_calls", "Hard-coded in RAGtuneSearchSpace.build_controller (1M / 120s / 20) — edit there to make them tunable"),
    # ── Optimizer level ──────────────────────────────────────────────────────
    ("optimizer", "n_trials / n_iterations", "Search budget per cell (--budget)"),
    ("optimizer", "n_startup_trials", "Random exploration before TPE engages (default budget//8, min 5)"),
    ("optimizer", "pruner thresholds", "max_mean_rerank_docs / max_trial_seconds / pareto_warmup_trials (--max-cost / --max-trial-seconds)"),
    ("optimizer", "storage_url", "sqlite:///study.db makes an Optuna study resumable and inspectable with optuna-dashboard"),
    ("optimizer", "llm_model / temperature", "LLM agent's proposer model and sampling temperature (--llm-model)"),
    ("optimizer", "n_minibatch / merge_every / n_startup", "GEPA screening size, crossover period, random seeding (--gepa-*)"),
    # ── Evaluation ───────────────────────────────────────────────────────────
    ("eval", "n_eval_queries", "Queries per trial (--n-queries). Under ~30 the NDCG noise (+/-0.02) exceeds most config differences"),
    ("eval", "seed", "Query subsample + optimizer seed (--seed). Re-run with 2–3 seeds before believing a ranking"),
    ("eval", "k in NDCG@k", "Fixed at 10 in ragtune.tuning.evaluator.ndcg_at_k — change there for @5 / @100 or to add MRR/Recall"),
    # ── Global config singleton (src/ragtune/config/defaults.yaml) ───────────
    ("config", "retrieval.pool_multiplier", "Over-retrieval factor feeding the candidate pool"),
    ("config", "retrieval.num_reformulations", "Default rewrite count when the budget allows it"),
    ("config", "retrieval.near_duplicate_threshold", "Dedup similarity cutoff (0.5–0.95) — swept by the tuner"),
    ("config", "reranking.batch_size", "Default scheduler batch size when not set per component"),
    ("config", "assembly.token_buffer", "Headroom left in the token budget during assembly"),
]

# One fixed pipeline configuration, used for the `baseline` stage and as the
# template that per-cell overrides are applied to.  Every key the search space
# builders read must be present.
BASELINE_PARAMS: Dict[str, Any] = {
    "reranker_type": "noop",
    "reformulator_type": "identity",
    "estimator_type": "baseline",
    "scheduler_type": "active-learning",
    "feedback_type": "none",
    # retrieval / pool
    "original_query_depth": 100,
    "depth_per_reformulation": 5,
    "max_pool_size": 100,
    "near_duplicate_threshold": 0.8,
    # assembler — 20 >= 10 so NDCG@10 is never truncated by the assembler
    "assembler_max_docs": 20,
    # budget
    "budget_rerank_docs": 50,
    "budget_reformulations": 0,
    # scheduler
    "scheduler_batch_size": 10,
    "gd_llm_limit": 3,
    "gd_ce_limit": 10,
    # reranker sub-params
    "ce_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "monot5_model": "castorini/monot5-base-msmarco",
    "monot5_batch_size": 16,
    # reformulator / estimator / feedback sub-params (inactive by default)
    "reformulator_model": "gpt-4o-mini",
    "reformulator_n_variants": 3,
    "similarity_model": "all-MiniLM-L6-v2",
    "min_reranked_for_regression": 3,
    "budget_stop_token_threshold": 0.9,
}


# ══════════════════════════════════════════════════════════════════════════════
# PyTerrier helpers
# ══════════════════════════════════════════════════════════════════════════════

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _ensure_java_home() -> None:
    """
    Point JAVA_HOME at a JDK before pyjnius looks for one.

    pyjnius needs a full JDK (it resolves javac), so a bare JRE — what many
    images ship — fails with "Unable to find javac". conda-forge's openjdk
    installs into <prefix>/lib/jvm, which pyjnius won't find on its own.
    """
    current = os.environ.get("JAVA_HOME")
    if current and Path(current, "bin", "javac").exists():
        return

    candidates = [Path(sys.prefix) / "lib" / "jvm"]
    candidates += sorted(Path("/usr/lib/jvm").glob("*"), reverse=True)
    for cand in candidates:
        if (cand / "bin" / "javac").exists():
            os.environ["JAVA_HOME"] = str(cand)
            libjvm = cand / "lib" / "server" / "libjvm.so"
            if libjvm.exists():
                os.environ.setdefault("JVM_PATH", str(libjvm))
            print(f"  [java] JAVA_HOME={cand}")
            return

    print(
        "  [java] WARNING: no JDK found (a JRE alone is not enough — pyjnius needs javac).\n"
        "         Fix with:  conda install -y -c conda-forge openjdk=21",
        file=sys.stderr,
    )


def init_pyterrier():
    """
    Import PyTerrier and make sure the JVM is up.

    The JVM bootstrap API moved around: <=0.10 has pt.init()/pt.started(),
    0.11+ has pt.java.init()/pt.java.started() and auto-starts on first use.
    Resolve whatever this install offers rather than assuming one of them.
    """
    _ensure_java_home()
    import pyterrier as pt

    if not hasattr(pt, "get_dataset"):
        raise RuntimeError(
            f"'pyterrier' imported from {getattr(pt, '__file__', '?')} is not the "
            "Terrier IR platform (no pt.get_dataset). The unrelated PyPI package "
            "'pyterrier' shadows it — run:\n"
            "  pip uninstall -y pyterrier python-terrier && pip install python-terrier"
        )

    java = getattr(pt, "java", None)
    started_fn = getattr(java, "started", None) or getattr(pt, "started", None)
    init_fn = getattr(java, "init", None) or getattr(pt, "init", None)
    if started_fn is not None and init_fn is not None and not started_fn():
        init_fn()
    return pt


def pt_attr(pt, *names: str):
    """
    Resolve a Terrier class that moved between PyTerrier releases: it may live
    at pt.<name> (<=0.10) or pt.terrier.<name> (0.11+), and may have been
    renamed (BatchRetrieve -> Retriever). First hit wins.
    """
    for name in names:
        for holder in (getattr(pt, "terrier", None), pt):
            obj = getattr(holder, name, None)
            if obj is not None:
                return obj
    raise AttributeError(f"PyTerrier exposes none of: {', '.join(names)}")


class SanitizingTransformer:
    """
    Strips punctuation from queries before handing them to Terrier.

    BEIR topics contain characters ('?', '/', ':', apostrophes) that Terrier's
    query parser rejects, which would surface as per-query exceptions and be
    silently scored 0.0 by the evaluator. Only applied to lexical retrievers —
    dense encoders handle raw text fine.
    """

    def __init__(self, inner: Any):
        self.inner = inner

    def transform(self, queries_df):
        df = queries_df.copy()
        df["query"] = df["query"].map(
            lambda q: _PUNCT_RE.sub(" ", str(q)).strip() or "empty"
        )
        return self.inner.transform(df)

    def search(self, query: str):
        import pandas as pd

        return self.transform(pd.DataFrame([{"qid": "q1", "query": query}]))


class RRFTransformer:
    """
    Reciprocal-rank fusion of two PyTerrier runs.

    score(d) = sum_r 1 / (k + rank_r(d)) over the runs that returned d, with the
    standard k=60. Fusion is on rank, not score, so the two runs need no
    calibration — which is the whole point of using RRF for lexical+dense.
    """

    def __init__(self, left: Any, right: Any, k: int = 60, num_results: int = 200):
        self.left = left
        self.right = right
        self.k = k
        self.num_results = num_results

    def transform(self, queries_df):
        import pandas as pd

        fused: Dict[str, Dict[str, Any]] = {}
        for run in (self.left, self.right):
            res = run.transform(queries_df)
            if res is None or len(res) == 0:
                continue
            res = res.sort_values("score", ascending=False)
            for rank, (_, row) in enumerate(res.iterrows()):
                docno = str(row["docno"])
                entry = fused.setdefault(
                    docno,
                    {"docno": docno, "score": 0.0, "text": row.get("text", ""),
                     "qid": row.get("qid", "q1"), "query": row.get("query", "")},
                )
                entry["score"] += 1.0 / (self.k + rank + 1)
                # Prefer whichever run actually carried document text.
                if not entry["text"] and row.get("text"):
                    entry["text"] = row["text"]

        out = pd.DataFrame(sorted(fused.values(), key=lambda r: -r["score"])[: self.num_results])
        if len(out):
            out["rank"] = range(len(out))
        return out


def sparse_index_path(index_dir: str, dataset: str) -> str:
    return os.path.join(index_dir, f"terrier_{dataset}")


def dense_index_path(index_dir: str, dataset: str, retriever: str) -> str:
    return os.path.join(index_dir, f"{retriever}_{dataset}")


def build_sparse_index(pt, dataset: str, index_dir: str) -> str:
    """
    Build (or reuse) a Terrier index with a stored 'body' field.

    PyTerrier 1.1+ routes queries through Terrier MatchOps, whose
    checkForFields() call blows up on indexes built without field statistics —
    hence fields=["body"]. Text is kept in the meta index so rerankers can read
    document content.
    """
    path = sparse_index_path(index_dir, dataset)
    props = Path(path) / "data.properties"
    if props.exists():
        idx = pt_attr(pt, "IndexFactory").of(path)
        stats = idx.getCollectionStatistics()
        if stats.getNumberOfFields() > 0:
            print(f"    [index] {path} ready — {stats.getNumberOfDocuments():,} docs")
            return path
        print(f"    [index] {path} has no field stats — rebuilding")

    print(f"    [index] building {path} …")
    ds = pt.get_dataset(DATASETS[dataset]["irds_id"])

    def _with_body(it):
        for doc in it:
            doc["body"] = doc.get("text", "")
            yield doc

    indexer = pt_attr(pt, "IterDictIndexer")(
        os.path.abspath(path),
        overwrite=True,
        meta={"docno": 26, "text": 131072},
        fields=["body"],
    )
    indexer.index(_with_body(ds.get_corpus_iter()))
    stats = pt_attr(pt, "IndexFactory").of(path).getCollectionStatistics()
    print(f"    [index] done — {stats.getNumberOfDocuments():,} docs")
    return path


def build_dense_index(pt, dataset: str, retriever: str, index_dir: str,
                      batch_size: int, device: str) -> str:
    """Build (or reuse) a pyterrier-dr FlexIndex for a dense retriever."""
    from ragtune.indexing.flex_indexer import FlexIndexer

    spec = RETRIEVERS[retriever]
    path = dense_index_path(index_dir, dataset, retriever)
    indexer = FlexIndexer(
        model_name=spec["model"],
        batch_size=batch_size,
        device=device,
        **spec.get("encoder_kwargs", {}),
    )
    if indexer.exists(path):
        print(f"    [index] {path} ready")
        return path

    n_docs = DATASETS[dataset]["n_docs"]
    if device == "cpu" and n_docs > 100_000:
        print(f"    [index] WARNING: encoding {n_docs:,} docs on CPU will take many "
              f"hours — pass --device cuda", file=sys.stderr)
    print(f"    [index] encoding {n_docs:,} docs into {path} (device={device}) …")
    ds = pt.get_dataset(DATASETS[dataset]["irds_id"])
    # build_from_corpus wants the whole corpus in memory; fine up to a few
    # million short passages, which is why heavy=True sets are opt-in.
    corpus = {
        str(doc["docno"]): {"text": doc.get("text", "")}
        for doc in ds.get_corpus_iter()
    }
    indexer.build_from_corpus(corpus, path)
    return path


def make_retriever(pt, retriever: str, dataset: str, index_dir: str, args) -> Any:
    """Return a RAGtune BaseRetriever for (retriever, dataset)."""
    from ragtune.adapters.pyterrier import PyTerrierRetriever

    spec = RETRIEVERS[retriever]
    sparse_path = sparse_index_path(index_dir, dataset)

    if spec["kind"] == "sparse":
        properties = {}
        if spec["wmodel"] == "BM25":
            # Terrier reads BM25 params from ApplicationSetup properties:
            # 'c' is b (length normalisation), 'k1' is term saturation.
            properties = {"c": str(args.bm25_b), "k1": str(args.bm25_k1)}
        retriever_cls = pt_attr(pt, "Retriever", "BatchRetrieve")
        br = retriever_cls(
            os.path.abspath(sparse_path),
            wmodel=spec["wmodel"],
            controls={"matchopql": "off"},  # skip MatchOps → no checkForFields()
            properties=properties,
            metadata=["docno", "text"],
            num_results=args.retrieval_depth,
        )
        return PyTerrierRetriever(pt_transformer=SanitizingTransformer(br))

    if spec["kind"] == "hybrid":
        # RRF over one lexical and one dense run. Text comes from whichever
        # component supplied the document, so no extra meta lookup is needed.
        sparse_rt = make_retriever(pt, args.hybrid_sparse, dataset, index_dir, args)
        dense_rt = make_retriever(pt, args.hybrid_dense, dataset, index_dir, args)
        fused = RRFTransformer(
            sparse_rt.pt_transformer, dense_rt.pt_transformer,
            k=args.rrf_k, num_results=args.retrieval_depth,
        )
        return PyTerrierRetriever(pt_transformer=fused)

    # Dense: encoder >> FlexIndex retrieval, then pull document text out of the
    # Terrier meta index so downstream rerankers have content to score.
    from ragtune.indexing.flex_indexer import FlexIndexer

    indexer = FlexIndexer(
        model_name=spec["model"], batch_size=args.encode_batch_size,
        device=args.device, **spec.get("encoder_kwargs", {}),
    )
    pipeline = indexer.get_retriever(
        dense_index_path(index_dir, dataset, retriever), backend=args.dense_backend
    )
    pipeline = pipeline >> pt.text.get_text(pt_attr(pt, "IndexFactory").of(sparse_path), "text")
    return PyTerrierRetriever(pt_transformer=pipeline)


# ══════════════════════════════════════════════════════════════════════════════
# Result bookkeeping
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Bench:
    """Accumulates every row the run produces and persists them incrementally."""

    out_dir: Path
    sanity: List[Dict[str, Any]] = field(default_factory=list)
    baseline: List[Dict[str, Any]] = field(default_factory=list)
    trials: List[Dict[str, Any]] = field(default_factory=list)
    done: Dict[str, bool] = field(default_factory=dict)

    def state_path(self) -> Path:
        return self.out_dir / "state.json"

    def load(self) -> None:
        p = self.state_path()
        if not p.exists():
            return
        data = json.loads(p.read_text())
        self.sanity = data.get("sanity", [])
        self.baseline = data.get("baseline", [])
        self.trials = data.get("trials", [])
        self.done = data.get("done", {})
        print(f"  Resumed: {len(self.done)} completed units from {p}")

    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.state_path().write_text(
            json.dumps(
                {
                    "sanity": self.sanity,
                    "baseline": self.baseline,
                    "trials": self.trials,
                    "done": self.done,
                },
                indent=2,
                default=str,
            )
        )

    def completed(self, key: str) -> bool:
        return bool(self.done.get(key))

    def mark(self, key: str) -> None:
        self.done[key] = True
        self.save()


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {path}  ({len(rows)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — retriever sanity check
# ══════════════════════════════════════════════════════════════════════════════

def stage_sanity(pt, dataset: str, retriever: str, eval_ds, args, bench: Bench) -> None:
    """Raw first-stage NDCG@10, no RAGtune loop, compared to published BEIR BM25."""
    from ragtune.tuning.evaluator import ndcg_at_k
    import numpy as np
    import pandas as pd

    key = f"sanity::{dataset}::{retriever}::s{args.seed}"
    if bench.completed(key):
        print(f"  [sanity] {retriever} — cached")
        return

    rt = make_retriever(pt, retriever, dataset, args.index_dir, args)
    transformer = rt.pt_transformer

    scores, failures, t0 = [], 0, time.time()
    for eq in eval_ds.iter_queries(limit=args.n_queries):
        try:
            res = transformer.transform(pd.DataFrame([{"qid": "q1", "query": eq.query}]))
            ranked = list(res.sort_values("score", ascending=False)["docno"].astype(str))
        except Exception as exc:  # noqa: BLE001 — a bad query shouldn't kill the sweep
            failures += 1
            if failures <= 3:
                print(f"    query {eq.query_id!r} failed: {type(exc).__name__}: {exc}")
            ranked = []
        scores.append(ndcg_at_k(ranked, eq.qrels, k=10))

    ndcg = float(np.mean(scores)) if scores else 0.0
    ref = DATASETS[dataset]["ndcg10_bm25"]
    row = {
        "dataset": dataset,
        "retriever": retriever,
        "seed": args.seed,
        "ndcg_at_10": round(ndcg, 4),
        "beir_bm25_ref": ref,
        "gap_vs_ref": round(ndcg - ref, 4),
        "n_queries": len(scores),
        "failed_queries": failures,
        "seconds": round(time.time() - t0, 1),
    }
    bench.sanity.append(row)
    bench.mark(key)
    flag = "" if abs(row["gap_vs_ref"]) < 0.08 or retriever != "bm25" else "  <-- CHECK"
    print(
        f"  [sanity] {retriever:<13} NDCG@10={ndcg:.4f}  "
        f"(BEIR BM25 ref {ref:.3f}, gap {row['gap_vs_ref']:+.3f}){flag}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — fixed-config baselines
# ══════════════════════════════════════════════════════════════════════════════

def reranker_variants(args) -> List[tuple]:
    """
    Expand the reranker axis into concrete (registry_type, checkpoint) pairs.

    noop and llm have no checkpoint axis in the search space; cross-encoder and
    monot5 each carry their own model menu, and the baseline stage evaluates
    every one of them so the tuned numbers have a per-checkpoint reference.
    """
    out: List[tuple] = []
    for reranker in args.rerankers:
        if reranker == "cross-encoder":
            out.extend(("cross-encoder", m) for m in args.ce_models)
        elif reranker == "monot5":
            out.extend(("monot5", m) for m in args.monot5_models)
        else:
            out.append((reranker, None))
    return out


def stage_baseline(pt, dataset: str, retriever: str, eval_ds, args, bench: Bench) -> None:
    """Evaluate one fixed RAGtune pipeline per reranker — the untuned reference."""
    from ragtune.tuning.llm_optimizer import evaluate_controller_full
    from ragtune.tuning.search_space import RAGtuneSearchSpace

    space = RAGtuneSearchSpace()
    rt = None

    for reranker, model in reranker_variants(args):
        label = reranker if model is None else f"{reranker}:{model.split('/')[-1]}"
        key = f"baseline::{dataset}::{retriever}::{label}::s{args.seed}"
        if bench.completed(key):
            print(f"  [baseline] {label} — cached")
            continue

        if rt is None:
            rt = make_retriever(pt, retriever, dataset, args.index_dir, args)

        params = dict(BASELINE_PARAMS)
        params.update(
            reranker_type=reranker,
            original_query_depth=args.baseline_depth,
            max_pool_size=args.baseline_depth,
            budget_rerank_docs=args.baseline_rerank_budget,
        )
        if reranker == "cross-encoder":
            params["ce_model"] = model
        elif reranker == "monot5":
            params["monot5_model"] = model
        elif reranker == "llm":
            params["llm_reranker_model"] = args.llm_reranker_model

        t0 = time.time()
        try:
            controller = space.build_controller(params, rt)
            result = evaluate_controller_full(
                controller, eval_ds, args.n_queries, space.to_retrieval_overrides(params)
            )
            row = {
                "dataset": dataset,
                "retriever": retriever,
                "reranker": label,
                "seed": args.seed,
                "ndcg_at_10": round(result.objectives.ndcg_at_10, 4),
                "mean_rerank_docs": round(result.objectives.rerank_docs, 2),
                "mean_latency_ms": round(result.objectives.latency_ms, 1),
                "n_queries": result.objectives.queries_evaluated,
                "seconds": round(time.time() - t0, 1),
                "error": "",
            }
            print(
                f"  [baseline] {label:<26} NDCG@10={row['ndcg_at_10']:.4f}  "
                f"cost={row['mean_rerank_docs']:6.1f} docs  "
                f"{row['mean_latency_ms']:7.0f} ms/query"
            )
        except Exception as exc:  # noqa: BLE001 — e.g. pyterrier-t5 not installed
            row = {
                "dataset": dataset, "retriever": retriever, "reranker": label,
                "seed": args.seed,
                "ndcg_at_10": 0.0, "mean_rerank_docs": 0.0, "mean_latency_ms": 0.0,
                "n_queries": 0, "seconds": round(time.time() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  [baseline] {label:<26} FAILED — {row['error'][:90]}")

        bench.baseline.append(row)
        bench.mark(key)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — optimizers
# ══════════════════════════════════════════════════════════════════════════════

def search_space_overrides(args) -> Dict[str, Any]:
    """Restrict the tuner's menus to the components selected on the CLI."""
    depth_hi = max(20, args.retrieval_depth)  # keep low < high for suggest_int
    overrides: Dict[str, Any] = {
        "reranker_types": list(args.rerankers),
        "scheduler_types": ["active-learning", "graceful-degradation"],
        "ce_models": list(args.ce_models),
        "monot5_models": list(args.monot5_models),
        # Depth can't exceed what the first stage actually returns.
        "original_query_depth_range": (10, depth_hi),
        "max_pool_size_range": (10, depth_hi),
        # 10..20: never truncate NDCG@10 via the assembler.
        "assembler_max_docs_range": (10, 20),
    }

    if args.space == "full":
        # Everything the registry offers, including the components that cost API
        # calls. 'reformir' estimator/reformulator and llm_rewrite all need a key.
        overrides.update({
            "reformulator_types": ["identity", "llm_rewrite", "reformir"],
            "estimator_types": ["baseline", "utility", "similarity", "reformir"],
            "feedback_types": ["none", "budget-stop", "reformir-convergence"],
            "reformulator_models": [args.llm_reranker_model],
            "budget_reformulations_range": (0, 5),
        })
    else:
        # Default: no LLM in the inner loop, so a sweep needs no API key and its
        # cost is bounded by local model throughput.
        overrides.update({
            "reformulator_types": ["identity"],
            "estimator_types": ["baseline", "utility", "similarity"],
            "feedback_types": ["none", "budget-stop"],
            "budget_reformulations_range": (0, 0),
        })

    if args.search_space_json:
        overrides.update(json.loads(Path(args.search_space_json).read_text()))
    return overrides


def run_bayes(dataset: str, retriever: str, rt, eval_ds, args, out_dir: Path) -> List[Dict[str, Any]]:
    import optuna

    from ragtune.tuning.optimizer import extract_pareto_configs, run_study
    from ragtune.tuning.search_space import RAGtuneSearchSpace
    from ragtune.tuning.study_config import DatasetConfig, TuningStudyConfig

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    overrides = search_space_overrides(args)
    cfg_dir = out_dir / "pareto_configs" / f"{dataset}_{retriever}_bayes_s{args.seed}"

    cfg = TuningStudyConfig(
        name=f"bayes-{dataset}-{retriever}-s{args.seed}",
        dataset=DatasetConfig(name=dataset, irds_id=DATASETS[dataset]["irds_id"]),
        n_trials=args.budget,
        n_startup_trials=max(5, args.budget // 8),
        n_eval_queries=args.n_queries,
        seed=args.seed,
        n_parallel_workers=1,  # the global config singleton is not thread-safe
        max_mean_rerank_docs=args.max_cost,
        max_trial_seconds=args.max_trial_seconds,
        pareto_warmup_trials=max(10, args.budget // 3),
        output_dir=str(cfg_dir),
        search_space_overrides=overrides,
    )

    t0 = time.time()
    study = run_study(cfg, rt, eval_ds)
    elapsed = time.time() - t0

    # run_study() does not persist configs; do it here.
    paths = extract_pareto_configs(study, RAGtuneSearchSpace(**overrides), str(cfg_dir))

    pareto_numbers = {t.number for t in study.best_trials}
    rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        if trial.values is None:
            continue
        row = {
            "dataset": dataset, "retriever": retriever, "optimizer": "bayes",
            "seed": args.seed, "iteration": trial.number,
            "ndcg_at_10": round(trial.values[0], 4),
            "mean_rerank_docs": round(trial.values[1], 2),
            "on_pareto": trial.number in pareto_numbers,
            "wall_seconds": round(elapsed, 1),
        }
        row.update({f"p_{k}": v for k, v in trial.params.items()})
        rows.append(row)

    best = max((r["ndcg_at_10"] for r in rows), default=0.0)
    print(
        f"  [bayes] {len(rows)}/{args.budget} trials scored in {elapsed:.0f}s — "
        f"best NDCG@10={best:.4f}, Pareto size {len(study.best_trials)}, "
        f"{len(paths)} configs → {cfg_dir}"
    )
    return rows


def run_llm(dataset: str, retriever: str, rt, eval_ds, args, out_dir: Path) -> List[Dict[str, Any]]:
    from ragtune.tuning.llm_optimizer import (
        LLMAgentOptimizer,
        LLMOptimizerConfig,
        compute_pareto_front,
    )

    overrides = search_space_overrides(args)
    cfg_dir = out_dir / "pareto_configs" / f"{dataset}_{retriever}_llm_s{args.seed}"

    cfg = LLMOptimizerConfig(
        name=f"llm-{dataset}-{retriever}-s{args.seed}",
        llm_model=args.llm_model,
        temperature=args.llm_temperature,
        n_iterations=args.budget,
        n_startup=args.gepa_startup,
        n_minibatch=min(args.gepa_minibatch, args.n_queries),
        merge_every=args.gepa_merge_every,
        n_eval_queries=args.n_queries,
        seed=args.seed,
        output_dir=str(cfg_dir),
        search_space_overrides=overrides,
    )

    t0 = time.time()
    history = LLMAgentOptimizer(config=cfg).run(rt, eval_ds)
    elapsed = time.time() - t0

    pareto_ids = {id(c) for c in compute_pareto_front(history)}
    rows: List[Dict[str, Any]] = []
    for cand in history:
        if cand.error:
            continue
        row = {
            "dataset": dataset, "retriever": retriever, "optimizer": "llm",
            "seed": args.seed, "iteration": cand.iteration,
            "ndcg_at_10": round(cand.ndcg_at_10, 4),
            "mean_rerank_docs": round(cand.mean_rerank_docs, 2),
            "on_pareto": id(cand) in pareto_ids,
            "wall_seconds": round(elapsed, 1),
            "mutated_module": cand.mutated_module or "",
            "rationale": (cand.rationale or "")[:300],
        }
        row.update({f"p_{k}": v for k, v in cand.params.items()})
        rows.append(row)

    screened = sum(1 for c in history if c.error and "screen" in (c.error or ""))
    best = max((r["ndcg_at_10"] for r in rows), default=0.0)
    print(
        f"  [llm]   {len(rows)} full evals (+{screened} minibatch-screened) in "
        f"{elapsed:.0f}s — best NDCG@10={best:.4f}, Pareto size {len(pareto_ids)} "
        f"→ {cfg_dir}"
    )
    return rows


def run_random(dataset: str, retriever: str, rt, eval_ds, args, out_dir: Path) -> List[Dict[str, Any]]:
    """
    Random-search control arm.

    Same search space, same evaluator, same budget as the other two — only the
    sampler differs. Without this it is impossible to tell whether TPE or the LLM
    agent is actually beating chance at these small budgets.
    """
    import optuna
    from optuna.samplers import RandomSampler

    from ragtune.tuning.evaluator import TrialEvaluator
    from ragtune.tuning.pruners import CostPruner, ParetoPruner, RuntimePruner
    from ragtune.tuning.search_space import RAGtuneSearchSpace

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    overrides = search_space_overrides(args)
    space = RAGtuneSearchSpace(**overrides)

    study = optuna.create_study(
        study_name=f"random-{dataset}-{retriever}-s{args.seed}",
        directions=["maximize", "minimize"],
        sampler=RandomSampler(seed=args.seed),
    )
    evaluator = TrialEvaluator(
        dataset=eval_ds,
        n_eval_queries=args.n_queries,
        pruners=[
            CostPruner(max_mean_rerank_docs=args.max_cost, warmup_steps=3),
            RuntimePruner(max_trial_seconds=args.max_trial_seconds, warmup_steps=3),
            ParetoPruner(
                study=study,
                warmup_trials=max(10, args.budget // 3),
                zscore=1.645,
            ),
        ],
    )

    def objective(trial):
        params = space.sample(trial)
        try:
            controller = space.build_controller(params, rt)
        except Exception as exc:  # noqa: BLE001 — model unavailable etc.
            trial.set_user_attr("build_error", str(exc))
            return 0.0, float("inf")
        obj = evaluator.evaluate(controller, trial, space.to_retrieval_overrides(params))
        return obj.ndcg_at_10, obj.rerank_docs

    t0 = time.time()
    study.optimize(objective, n_trials=args.budget, catch=(Exception,))
    elapsed = time.time() - t0

    pareto_numbers = {t.number for t in study.best_trials}
    rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        if trial.values is None:
            continue
        row = {
            "dataset": dataset, "retriever": retriever, "optimizer": "random",
            "seed": args.seed, "iteration": trial.number,
            "ndcg_at_10": round(trial.values[0], 4),
            "mean_rerank_docs": round(trial.values[1], 2),
            "on_pareto": trial.number in pareto_numbers,
            "wall_seconds": round(elapsed, 1),
        }
        row.update({f"p_{k}": v for k, v in trial.params.items()})
        rows.append(row)

    best = max((r["ndcg_at_10"] for r in rows), default=0.0)
    print(f"  [random] {len(rows)}/{args.budget} trials in {elapsed:.0f}s — "
          f"best NDCG@10={best:.4f}, Pareto size {len(pareto_numbers)}")
    return rows


OPTIMIZER_RUNNERS = {"bayes": run_bayes, "llm": run_llm, "random": run_random}


def stage_tune(pt, dataset: str, retriever: str, eval_ds, args, bench: Bench) -> None:
    rt = None
    for optimizer in args.optimizers:
        key = f"tune::{dataset}::{retriever}::{optimizer}::s{args.seed}"
        if bench.completed(key):
            print(f"  [{optimizer}] cached")
            continue
        if rt is None:
            rt = make_retriever(pt, retriever, dataset, args.index_dir, args)

        runner = OPTIMIZER_RUNNERS[optimizer]
        try:
            rows = runner(dataset, retriever, rt, eval_ds, args, Path(args.out_dir))
        except Exception as exc:  # noqa: BLE001 — keep the sweep alive
            print(f"  [{optimizer}] FAILED — {type(exc).__name__}: {exc}")
            continue
        bench.trials.extend(rows)
        bench.mark(key)


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _stdev(values) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = sum(vals) / len(vals)
    return (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _ordered_keys(rows: List[Dict[str, Any]], fields: tuple) -> List[tuple]:
    """Distinct field tuples, in first-seen order (keeps report order stable)."""
    seen: List[tuple] = []
    for r in rows:
        key = tuple(r[f] for f in fields)
        if key not in seen:
            seen.append(key)
    return seen


def hypervolume_2d(pairs: List[tuple], ref_cost: float) -> float:
    """Dominated area for (maximize NDCG, minimize cost)."""
    pts = sorted([(n, c) for n, c in pairs if c < ref_cost and n > 0], key=lambda p: p[1])
    hv, prev = 0.0, 0.0
    for ndcg, cost in pts:
        hv += ndcg * (cost - prev)
        prev = cost
    return hv


def report(bench: Bench, args) -> None:
    out = Path(args.out_dir)
    write_csv(bench.sanity, out / "sanity.csv")
    write_csv(bench.baseline, out / "baseline.csv")
    write_csv(bench.trials, out / "trials.csv")

    lines: List[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit()
    emit("=" * 96)
    emit(f"  RAGtune BEIR benchmark  |  budget={args.budget}  queries={args.n_queries}  "
         f"seeds={','.join(map(str, args.seed_list))}  space={args.space}")
    emit("=" * 96)

    if bench.sanity:
        emit()
        emit("First-stage retrieval (no RAGtune loop)")
        emit(f"  {'dataset':<14} {'retriever':<14} {'seed':>5} {'NDCG@10':>8} {'BEIR ref':>9} "
             f"{'gap':>7} {'failed q':>9}")
        emit("  " + "-" * 72)
        for r in bench.sanity:
            ref = r["beir_bm25_ref"]
            emit(
                f"  {r['dataset']:<14} {r['retriever']:<14} {r.get('seed', ''):>5} "
                f"{r['ndcg_at_10']:>8.4f} {ref:>9.3f} {r['gap_vs_ref']:>+7.3f} {r['failed_queries']:>9}"
            )

    if bench.baseline:
        emit()
        emit("Fixed-config baselines (depth="
             f"{args.baseline_depth}, rerank budget={args.baseline_rerank_budget}; "
             "mean over seeds)")
        emit(f"  {'dataset':<14} {'retriever':<14} {'reranker':<38} {'NDCG@10':>8} "
             f"{'cost':>7} {'ms/query':>9} {'n':>3}")
        emit("  " + "-" * 100)
        for key in _ordered_keys(bench.baseline, ("dataset", "retriever", "reranker")):
            rows = [r for r in bench.baseline
                    if (r["dataset"], r["retriever"], r["reranker"]) == key]
            ok = [r for r in rows if not r.get("error")]
            if not ok:
                emit(f"  {key[0]:<14} {key[1]:<14} {key[2]:<38} {'FAILED':>8} "
                     f"{'':>7} {'':>9} {len(rows):>3}")
                continue
            emit(
                f"  {key[0]:<14} {key[1]:<14} {key[2]:<38} "
                f"{_mean(r['ndcg_at_10'] for r in ok):>8.4f} "
                f"{_mean(r['mean_rerank_docs'] for r in ok):>7.1f} "
                f"{_mean(r['mean_latency_ms'] for r in ok):>9.0f} {len(ok):>3}"
            )

    if bench.trials:
        emit()
        emit("Tuned pipelines (per-seed best NDCG@10, averaged; Pareto hypervolume)")
        emit(f"  {'dataset':<14} {'retriever':<14} {'optimizer':<9} {'best NDCG':>10} {'+/-':>7} "
             f"{'vs base':>9} {'HV':>8} {'|front|':>8} {'seeds':>6} {'wall s':>8}")
        emit("  " + "-" * 104)

        for ds, rtv, opt in _ordered_keys(bench.trials, ("dataset", "retriever", "optimizer")):
            rows = [r for r in bench.trials
                    if (r["dataset"], r["retriever"], r["optimizer"]) == (ds, rtv, opt)]
            seeds = sorted({r.get("seed", args.seed) for r in rows})
            # Best-per-seed, then average: averaging raw trials would just
            # measure how much of the space each sampler wasted.
            per_seed_best = [
                max(r["ndcg_at_10"] for r in rows if r.get("seed", args.seed) == sd)
                for sd in seeds
            ]
            best = _mean(per_seed_best)
            spread = _stdev(per_seed_best)
            base_rows = [r["ndcg_at_10"] for r in bench.baseline
                         if r["dataset"] == ds and r["retriever"] == rtv and not r.get("error")]
            base = max(base_rows, default=0.0)
            front = [(r["ndcg_at_10"], r["mean_rerank_docs"]) for r in rows if r["on_pareto"]]
            hv = hypervolume_2d(front, args.max_cost)
            wall = max(r["wall_seconds"] for r in rows)
            emit(
                f"  {ds:<14} {rtv:<14} {opt:<9} {best:>10.4f} {spread:>7.4f} "
                f"{best - base:>+9.4f} {hv:>8.2f} {len(front):>8} {len(seeds):>6} {wall:>8.0f}"
            )
        emit()
        emit("  best NDCG = mean over seeds of each seed's best trial; +/- is the stdev over seeds.")
        emit("  'vs base' = that mean minus the best fixed-config baseline for the same cell.")
        emit(f"  HV = dominated hypervolume of the Pareto front (ref cost {args.max_cost:.0f} docs); higher is better.")
        if len(args.seed_list) < 2:
            emit("  NOTE: single seed — differences under ~0.02 NDCG are not distinguishable "
                 "from noise. Re-run with --seeds 42,43,44.")

    emit()
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"  Wrote {out / 'summary.md'}")

    if not args.no_plots:
        make_plots(bench, args)


def make_plots(bench: Bench, args) -> None:
    if not bench.trials:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plots")
        return

    out = Path(args.out_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    colors = {"bayes": "#B71C1C", "llm": "#1565C0"}

    cells = sorted({(r["dataset"], r["retriever"]) for r in bench.trials})
    for ds, rtv in cells:
        fig, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(13, 4.5))
        for opt in args.optimizers:
            rows = [r for r in bench.trials
                    if (r["dataset"], r["retriever"], r["optimizer"]) == (ds, rtv, opt)]
            if not rows:
                continue
            c = colors.get(opt, "#555555")
            ax_p.scatter([r["mean_rerank_docs"] for r in rows],
                         [r["ndcg_at_10"] for r in rows],
                         s=22, alpha=0.25, color=c, label=f"{opt} — all evals")
            front = sorted(
                [(r["mean_rerank_docs"], r["ndcg_at_10"]) for r in rows if r["on_pareto"]]
            )
            if front:
                xs, ys = zip(*front)
                ax_p.step(xs, ys, where="pre", color=c, linewidth=2.2)
                ax_p.scatter(xs, ys, s=80, color=c, zorder=5,
                             edgecolors="white", linewidths=0.8, label=f"{opt} — Pareto")

            best, curve = 0.0, []
            for r in sorted(rows, key=lambda x: x["iteration"]):
                best = max(best, r["ndcg_at_10"])
                curve.append(best)
            ax_c.plot(range(1, len(curve) + 1), curve, color=c, linewidth=2.2,
                      marker="o", markersize=3, label=opt)

        base = [r for r in bench.baseline if r["dataset"] == ds and r["retriever"] == rtv]
        for r in base:
            if r["ndcg_at_10"] > 0:
                ax_p.axhline(r["ndcg_at_10"], color="#777777", linestyle=":", linewidth=1)
                ax_p.annotate(f"baseline/{r['reranker']}", (0, r["ndcg_at_10"]),
                              fontsize=7, color="#555555", va="bottom")

        ax_p.set_xlabel("Mean rerank docs per query (cost, lower is better)")
        ax_p.set_ylabel("NDCG@10")
        ax_p.set_title(f"{ds} / {rtv} — Pareto front", fontweight="bold")
        ax_p.legend(fontsize=8)
        ax_p.grid(alpha=0.3)

        ax_c.set_xlabel("Iteration / trial")
        ax_c.set_ylabel("Best NDCG@10 so far")
        ax_c.set_title(f"{ds} / {rtv} — convergence", fontweight="bold")
        ax_c.legend(fontsize=8)
        ax_c.grid(alpha=0.3)
        ax_c.set_ylim(bottom=0)

        fig.tight_layout()
        path = out / f"{ds}_{rtv}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {path}")


def list_knobs() -> None:
    from ragtune.tuning.search_space import RAGtuneSearchSpace

    space = RAGtuneSearchSpace()
    print("\n" + "=" * 88)
    print("  Swept by this script (RAGtuneSearchSpace fields, override with --search-space-json)")
    print("=" * 88)
    # model_fields on pydantic v2, __fields__ on v1
    fields = getattr(RAGtuneSearchSpace, "model_fields", None) or RAGtuneSearchSpace.__fields__
    for name in fields:
        value = getattr(space, name)
        kind = "range" if isinstance(value, tuple) else "menu "
        print(f"  {kind}  {name:<34} {value}")

    print("\n" + "=" * 88)
    print("  Other knobs (fixed per run here, but tunable)")
    print("=" * 88)
    scope = None
    for s, name, desc in EXTRA_KNOBS:
        if s != scope:
            print(f"\n  [{s}]")
            scope = s
        print(f"    {name:<38} {desc}")

    print("\n" + "=" * 88)
    print("  Grid dimensions (CLI)")
    print("=" * 88)
    for label, catalogue in (("datasets", DATASETS), ("retrievers", RETRIEVERS), ("rerankers", RERANKERS)):
        print(f"\n  [{label}]")
        for k, v in catalogue.items():
            desc = v.get("desc") or f"{v.get('n_docs', 0):,} docs, BEIR BM25 ref {v.get('ndcg10_bm25')}"
            print(f"    {k:<16} {desc}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def validate_datasets(args) -> int:
    """Check every selected dataset resolves before a multi-day run starts."""
    try:
        import ir_datasets
    except ImportError:
        print("ir-datasets is not installed", file=sys.stderr)
        return 2

    bad = 0
    for name in args.datasets:
        irds_id = DATASETS[name]["irds_id"].replace("irds:", "")
        try:
            ds = ir_datasets.load(irds_id)
            ok = ds.has_queries() and ds.has_qrels() and ds.has_docs()
            print(f"  {'OK  ' if ok else 'PART'}  {name:22s} {irds_id}")
            bad += 0 if ok else 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {name:22s} {irds_id} — {type(exc).__name__}: {exc}")
            bad += 1
    return 1 if bad else 0


def print_plan(args) -> None:
    """
    Print the grid and a cost estimate.

    The per-query timings are the script's own measurements on one CPU box, so
    the estimate is an order of magnitude, not a promise — run the baseline
    stage on the target hardware and rescale.
    """
    n_cells = len(args.datasets) * len(args.retrievers) * len(args.seed_list)
    variants = reranker_variants(args)
    n_baseline = n_cells * len(variants)
    n_tune = n_cells * len(args.optimizers)
    query_evals = n_tune * args.budget * args.n_queries

    print("=" * 96)
    print("  RAGtune BEIR benchmark — plan")
    print("=" * 96)
    print(f"  datasets   : {len(args.datasets):>3}  {', '.join(args.datasets)}")
    print(f"  retrievers : {len(args.retrievers):>3}  {', '.join(args.retrievers)}")
    print(f"  rerankers  : {len(variants):>3}  "
          + ", ".join(r if m is None else f"{r}:{m.split('/')[-1]}" for r, m in variants))
    print(f"  optimizers : {len(args.optimizers):>3}  {', '.join(args.optimizers)}"
          f"      search space: {args.space}")
    print(f"  seeds      : {len(args.seed_list):>3}  {', '.join(map(str, args.seed_list))}")
    print(f"  stages     : {', '.join(args.stages)}")
    print("-" * 96)
    print(f"  cells (dataset x retriever x seed) : {n_cells}")
    print(f"  baseline evaluations               : {n_baseline}")
    print(f"  tuning runs                        : {n_tune}"
          f"  ({args.budget} evals x {args.n_queries} queries each)")
    print(f"  total query evaluations            : {query_evals:,}")

    corpus = sum(DATASETS[d]["n_docs"] for d in args.datasets)
    n_dense = sum(1 for r in args.retrievers if RETRIEVERS[r]["kind"] == "dense")
    print(f"  documents to index                 : {corpus:,}"
          f"  (x{n_dense} dense encoder pass{'es' if n_dense != 1 else ''})")

    # Measured on CPU: ~0.06 s/query with no reranking, ~0.4 s per reranked doc
    # with MiniLM-L6. A modern GPU is roughly 30x on the reranking term.
    per_doc = 0.4 if args.device == "cpu" else 0.013
    est_s = query_evals * (0.06 + 0.5 * args.baseline_rerank_budget * per_doc)
    print(f"  rough tuning wall time             : {est_s / 3600:.1f} h"
          f"  (assumes {args.device}; measure with --stages baseline first)")
    print("=" * 96)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end RAGtune benchmark over BEIR datasets x retrievers x rerankers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("grid")
    g.add_argument("--preset", default=None,
                   choices=["smoke", "standard", "exhaustive", "gpu-full"],
                   help="named grid preset; individual flags given alongside still win")
    g.add_argument("--datasets", default="nfcorpus,scifact",
                   help="comma-separated dataset names, or a group: "
                        + ", ".join(DATASET_GROUPS))
    g.add_argument("--retrievers", default="bm25,dph",
                   help="comma-separated retriever names, or a group: "
                        + ", ".join(RETRIEVER_GROUPS))
    g.add_argument("--rerankers", default="noop,cross-encoder",
                   help=f"comma-separated; choices: {','.join(RERANKERS)}")
    g.add_argument("--stages", default="sanity,baseline,tune",
                   help="comma-separated subset of sanity,baseline,tune")
    g.add_argument("--optimizers", default="bayes",
                   help="comma-separated subset of bayes,llm,random "
                        "(llm needs OPENAI_API_KEY; random is the control arm)")
    g.add_argument("--space", default="restricted", choices=["restricted", "full"],
                   help="'full' adds llm_rewrite/reformir reformulators, the reformir "
                        "estimator and reformir-convergence feedback — all need an API key")
    g.add_argument("--allow-heavy", action="store_true",
                   help="permit corpora >1M docs (nq, hotpotqa, fever, msmarco, ...)")

    e = p.add_argument_group("evaluation")
    e.add_argument("--n-queries", type=int, default=50, help="queries evaluated per config")
    e.add_argument("--budget", type=int, default=30, help="trials (bayes) / iterations (llm) per cell")
    e.add_argument("--seed", type=int, default=42, help="single seed (ignored if --seeds given)")
    e.add_argument("--seeds", default=None,
                   help="comma-separated seeds to repeat the whole grid over, e.g. 42,43,44. "
                        "Repetition is the only way to tell a real gap from NDCG noise")
    e.add_argument("--max-cost", type=float, default=200.0,
                   help="cost pruner threshold and hypervolume reference (mean rerank docs)")
    e.add_argument("--max-trial-seconds", type=float, default=600.0,
                   help="runtime pruner: abort a trial projected to exceed this")

    r = p.add_argument_group("retrieval")
    r.add_argument("--retrieval-depth", type=int, default=200,
                   help="first-stage candidate depth; also caps the tuner's depth range")
    r.add_argument("--bm25-k1", type=float, default=0.9)
    r.add_argument("--bm25-b", type=float, default=0.4)
    r.add_argument("--dense-backend", default="np", choices=["np", "torch", "faiss_flat", "faiss_hnsw"])
    r.add_argument("--encode-batch-size", type=int, default=32, help="dense indexing batch size")
    r.add_argument("--device", default="cpu", help="cpu or cuda — dense encoders only")
    r.add_argument("--hybrid-sparse", default="bm25", help="lexical side of hybrid-rrf")
    r.add_argument("--hybrid-dense", default="dense-minilm", help="dense side of hybrid-rrf")
    r.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant")

    b = p.add_argument_group("baseline pipeline")
    b.add_argument("--baseline-depth", type=int, default=100)
    b.add_argument("--baseline-rerank-budget", type=int, default=50)
    b.add_argument("--ce-models", default=",".join(CE_MODEL_MENU["standard"]),
                   help="comma-separated CrossEncoder checkpoints, or a menu: "
                        + ", ".join(CE_MODEL_MENU))
    b.add_argument("--monot5-models", default=",".join(MONOT5_MODEL_MENU["standard"]),
                   help="comma-separated MonoT5 checkpoints, or a menu: "
                        + ", ".join(MONOT5_MODEL_MENU))
    b.add_argument("--llm-reranker-model", default="gpt-4o-mini",
                   help="model behind --rerankers llm and the 'full' search space")

    lo = p.add_argument_group("llm optimizer")
    lo.add_argument("--llm-model", default="gpt-4o-mini")
    lo.add_argument("--llm-temperature", type=float, default=0.7)
    lo.add_argument("--gepa-startup", type=int, default=3)
    lo.add_argument("--gepa-minibatch", type=int, default=10)
    lo.add_argument("--gepa-merge-every", type=int, default=10)

    o = p.add_argument_group("io")
    o.add_argument("--index-dir", default="./indexes")
    o.add_argument("--out-dir", default="./benchmark_results")
    o.add_argument("--search-space-json", default=None,
                   help="JSON file of RAGtuneSearchSpace overrides, merged last")
    o.add_argument("--resume", action="store_true", help="skip units already recorded in state.json")
    o.add_argument("--report-only", action="store_true",
                   help="rebuild CSVs/summary/plots from an existing state.json and exit "
                        "(use after an interrupted run — no JVM, no re-evaluation)")
    o.add_argument("--no-plots", action="store_true")
    o.add_argument("--list-knobs", action="store_true", help="print every tunable knob and exit")
    o.add_argument("--smoke", action="store_true", help="alias for --preset smoke")
    o.add_argument("--plan", action="store_true",
                   help="print the full grid, unit count and cost estimate, then exit "
                        "without evaluating anything")
    o.add_argument("--validate-datasets", action="store_true",
                   help="check every selected dataset resolves in ir_datasets, then exit")

    args = p.parse_args(argv)

    preset = "smoke" if args.smoke else args.preset
    if preset:
        given = {a.split("=")[0] for a in (argv if argv is not None else sys.argv[1:])}

        def maybe(flag: str, attr: str, value: Any) -> None:
            """Apply a preset value only where the user didn't say otherwise."""
            if flag not in given:
                setattr(args, attr, value)

        if preset == "smoke":
            maybe("--datasets", "datasets", "tiny")
            maybe("--retrievers", "retrievers", "bm25,dph")
            maybe("--rerankers", "rerankers", "noop,cross-encoder")
            maybe("--optimizers", "optimizers", "bayes")
            maybe("--n-queries", "n_queries", 5)
            maybe("--budget", "budget", 6)
            maybe("--retrieval-depth", "retrieval_depth", 50)
            maybe("--baseline-depth", "baseline_depth", 50)
            maybe("--max-trial-seconds", "max_trial_seconds", 3600.0)
        elif preset == "standard":
            maybe("--datasets", "datasets", "standard")
            maybe("--retrievers", "retrievers", "standard")
            maybe("--rerankers", "rerankers", "noop,cross-encoder,monot5")
            maybe("--optimizers", "optimizers", "bayes,random")
            maybe("--n-queries", "n_queries", 50)
            maybe("--budget", "budget", 50)
        elif preset == "gpu-full":
            # The practical "as much coverage as fits in ~a day or two on one
            # modern GPU" setting. Run --plan to confirm against your hardware.
            maybe("--datasets", "datasets", "standard")
            maybe("--retrievers", "retrievers", "bm25,dph,dense-minilm,dense-bge,dense-tasb,hybrid-rrf")
            maybe("--rerankers", "rerankers", "noop,cross-encoder,monot5")
            maybe("--ce-models", "ce_models", "standard")
            maybe("--monot5-models", "monot5_models", "standard")
            maybe("--optimizers", "optimizers",
                  "bayes,random,llm" if os.environ.get("OPENAI_API_KEY") else "bayes,random")
            maybe("--seeds", "seeds", "42,43")
            maybe("--n-queries", "n_queries", 50)
            maybe("--budget", "budget", 50)
            maybe("--retrieval-depth", "retrieval_depth", 200)
            maybe("--device", "device", "cuda")
            maybe("--max-trial-seconds", "max_trial_seconds", 1800.0)
            maybe("--encode-batch-size", "encode_batch_size", 128)
        elif preset == "exhaustive":
            # Every retriever x every reranker checkpoint x every non-licensed
            # dataset x 3 seeds. This is a multi-WEEK sweep on one GPU; it exists
            # so the ceiling is expressible, not because you should run it whole.
            # --plan prints the bill. Split it across machines by dataset group.
            maybe("--datasets", "datasets", "all" if args.allow_heavy else "standard")
            maybe("--retrievers", "retrievers", "all")
            maybe("--rerankers", "rerankers", "noop,cross-encoder,monot5")
            maybe("--ce-models", "ce_models", "all")
            maybe("--monot5-models", "monot5_models", "all")
            maybe("--optimizers", "optimizers",
                  "bayes,random,llm" if os.environ.get("OPENAI_API_KEY") else "bayes,random")
            maybe("--seeds", "seeds", "42,43,44")
            maybe("--n-queries", "n_queries", 100)
            maybe("--budget", "budget", 100)
            maybe("--retrieval-depth", "retrieval_depth", 200)
            maybe("--device", "device", "cuda")
            maybe("--max-trial-seconds", "max_trial_seconds", 1800.0)
            maybe("--encode-batch-size", "encode_batch_size", 128)

    def split(value: str, valid, label: str, groups: Optional[Dict[str, List[str]]] = None) -> List[str]:
        if groups and value in groups:
            return list(groups[value])
        items = [x.strip() for x in str(value).split(",") if x.strip()]
        bad = [x for x in items if x not in valid]
        if bad:
            p.error(f"unknown {label}: {', '.join(bad)} (choices: {', '.join(valid)})")
        return items

    args.datasets = split(args.datasets, DATASETS, "dataset", DATASET_GROUPS)
    args.retrievers = split(args.retrievers, RETRIEVERS, "retriever", RETRIEVER_GROUPS)
    args.rerankers = split(args.rerankers, RERANKERS, "reranker")
    args.stages = split(args.stages, ["sanity", "baseline", "tune"], "stage")
    args.optimizers = split(args.optimizers, ["bayes", "llm", "random"], "optimizer")

    # Model menus accept either a menu name or an explicit comma list.
    args.ce_models = CE_MODEL_MENU.get(args.ce_models, None) or [
        m.strip() for m in str(args.ce_models).split(",") if m.strip()
    ]
    args.monot5_models = MONOT5_MODEL_MENU.get(args.monot5_models, None) or [
        m.strip() for m in str(args.monot5_models).split(",") if m.strip()
    ]

    args.seed_list = (
        [int(x) for x in str(args.seeds).split(",") if x.strip()] if args.seeds else [args.seed]
    )

    if not args.allow_heavy:
        heavy = [d for d in args.datasets if DATASETS[d]["heavy"]]
        if heavy:
            p.error(
                f"{', '.join(heavy)} exceed 1M documents; indexing alone is hours and "
                "dense encoding is a GPU-day each. Pass --allow-heavy to include them."
            )

    if "hybrid-rrf" in args.retrievers:
        for flag, val in (("--hybrid-sparse", args.hybrid_sparse), ("--hybrid-dense", args.hybrid_dense)):
            if val not in RETRIEVERS:
                p.error(f"{flag}={val!r} is not a known retriever")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_knobs:
        list_knobs()
        return 0

    if args.report_only:
        bench = Bench(out_dir=Path(args.out_dir))
        bench.load()
        if not (bench.sanity or bench.baseline or bench.trials):
            print(f"No results in {Path(args.out_dir) / 'state.json'}", file=sys.stderr)
            return 1
        report(bench, args)
        return 0

    if args.validate_datasets:
        return validate_datasets(args)

    if args.plan:
        print_plan(args)
        return 0

    needs_key = ("llm" in args.optimizers or args.space == "full" or "llm" in args.rerankers)
    if "tune" in args.stages and needs_key and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: --optimizers llm / --space full / --rerankers llm all require "
              "OPENAI_API_KEY in the environment.", file=sys.stderr)
        return 2

    print_plan(args)

    pt = init_pyterrier()
    import ragtune.adapters  # noqa: F401 — registry side effects
    import ragtune.components  # noqa: F401
    from ragtune.tuning.evaluator import EvalDataset

    bench = Bench(out_dir=Path(args.out_dir))
    if args.resume:
        bench.load()

    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for dataset in args.datasets:
        print(f"\n{'=' * 96}\n  {dataset.upper()}  ({DATASETS[dataset]['n_docs']:,} docs)\n{'=' * 96}")
        build_sparse_index(pt, dataset, args.index_dir)
        for retriever in args.retrievers:
            needed = ([args.hybrid_dense] if retriever == "hybrid-rrf" else [retriever])
            for dep in needed:
                if RETRIEVERS[dep]["kind"] == "dense":
                    build_dense_index(pt, dataset, dep, args.index_dir,
                                      args.encode_batch_size, args.device)

        for seed in args.seed_list:
            args.seed = seed  # stage keys and samplers read args.seed
            eval_ds = EvalDataset.from_pyterrier_irds(
                irds_id=DATASETS[dataset]["irds_id"], n_queries=args.n_queries, seed=seed
            )
            print(f"\n    [seed {seed}] {len(eval_ds.queries)} topics loaded")

            for retriever in args.retrievers:
                print(f"\n  --- {dataset} / {retriever} / seed {seed} ---")
                if "sanity" in args.stages:
                    stage_sanity(pt, dataset, retriever, eval_ds, args, bench)
                if "baseline" in args.stages:
                    stage_baseline(pt, dataset, retriever, eval_ds, args, bench)
                if "tune" in args.stages:
                    stage_tune(pt, dataset, retriever, eval_ds, args, bench)

    bench.save()
    report(bench, args)
    print(f"\nTotal wall time: {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
