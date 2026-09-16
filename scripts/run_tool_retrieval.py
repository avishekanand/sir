"""
RAGtune Tool & Skill Retrieval Benchmark Runner
================================================
Evaluates RAGtune against baseline retrieval on ToolRet, SkillRet, and
SRA-Bench datasets. Uses the YAML config system (ConfigLoader) for
scenario configuration.

Usage:
    BENCHMARK=sra python scripts/run_tool_retrieval.py
    BENCHMARK=toolret SUBSET=apibank python scripts/run_tool_retrieval.py
    SCENARIOS='[{"name":"custom","pipeline":{"components":{"reranker":{"type":"cross-encoder"}}}}]' BENCHMARK=sra python scripts/run_tool_retrieval.py
"""

import os
import tempfile
import time
from typing import Dict, List, Tuple

import pandas as pd
from rich.console import Console

from ragtune.data.loaders import DataLoaderFactory
from ragtune.data.constants import Benchmark, TOOLRET_SUBSETS, SRA_BENCH_SUBSETS
from ragtune.evaluation.RetrievalEvaluator import RetrievalEvaluator
from ragtune.indexing import IndexFactory
from ragtune.cli.config_loader import ConfigLoader
from ragtune.adapters.pyterrier import PyTerrierRetriever
from ragtune.utils.config import config

# Trigger registry registration for all components
import ragtune.components  # noqa: F401
import ragtune.adapters  # noqa: F401

_console = Console()


def ph(msg):
    _console.print(f"[bold blue]{msg}[/bold blue]")


def ps(msg):
    _console.print(f"[dim]{msg}[/dim]")


# ── Query sanitization (strip LaTeX/math/special chars that crash TerrierQL) ──

import re as _re


def sanitize_query(text):
    """Strip LaTeX, math symbols, unicode, and special chars for PyTerrier."""
    if not text:
        return ""
    text = _re.sub(r"\$[^$]*\$", "", text)
    text = _re.sub(r"\$\$[^$]*\$\$", "", text)
    text = _re.sub(r"\\[a-zA-Z]+", "", text)
    text = _re.sub(r"\\(.)", r"\1", text)
    text = _re.sub(r"\^+", " ** ", text)
    text = text.replace("{", "(").replace("}", ")")
    text = _re.sub(r"#\d+", "", text)
    text = text.replace("#", " ").replace("*", " × ")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = (
        text.replace("\u00b0", " deg ").replace("\u00b5", " u ").replace("\u00ae", "")
    )
    text = text.encode("ascii", "ignore").decode("ascii")
    text = _re.sub(r"\s+", " ", text).strip()
    return text[:2000]


# --- Configuration (via environment variables) ---

BENCHMARK: str = os.environ.get("BENCHMARK", "toolret")
SUBSET: str = os.environ.get("SUBSET", "")
QUERIES: int = int(os.environ.get("QUERIES", "0"))
TOP_K: int = int(os.environ.get("TOP_K", "100"))
EVAL_K: int = int(os.environ.get("EVAL_K", "10"))

_evaluator = RetrievalEvaluator(k_values=[EVAL_K])

# --- Data Loading ---


def load_task(benchmark: str, subset: str) -> Tuple:
    ps(f"Loading [{benchmark}] {subset}...")
    bm = {
        "toolret": Benchmark.TOOLRET,
        "skillret": Benchmark.SKILLRET,
        "sra": Benchmark.SRA_BENCH,
    }.get(benchmark)
    if not bm:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    factory = DataLoaderFactory()
    loader = factory.create_dataloader(
        dataset_name=subset, benchmark_name=bm, n_queries=QUERIES
    )
    return loader.get_corpus(), loader.get_queries(), loader.get_qrels()


# --- Index Building ---


def build_retriever(corpus):
    """Build retriever via Mandeep's IndexFactory."""
    ps(f"Indexing {len(corpus)} documents...")
    import pyterrier as pt

    if not pt.java.started():
        pt.java.init()
    indexer = IndexFactory.create("pyterrier")
    index_path = os.path.join(tempfile.mkdtemp(), "idx")
    indexer.build_from_corpus(corpus, index_path=index_path)
    idx_ref = pt.IndexFactory.of(index_path)
    bm25 = pt.terrier.Retriever(
        idx_ref, wmodel="BM25", metadata=["docno", "text"], num_results=TOP_K
    )
    return PyTerrierRetriever(bm25)


# --- Main ---


def main():
    config.set("retrieval.original_query_depth", TOP_K)
    ph("RAGtune Tool & Skill Retrieval Benchmarks")
    ps(
        f"Benchmark: {BENCHMARK}  |  Subset: {SUBSET or '(all)'}  |  Queries: {QUERIES}  |  Top-K: {TOP_K}"
    )

    subsets = {
        "toolret": TOOLRET_SUBSETS,
        "skillret": ["test"],
        "sra": SRA_BENCH_SUBSETS,
    }.get(BENCHMARK, [])
    if SUBSET:
        subsets = [SUBSET]

    all_rows: List[Dict] = []
    for subset in subsets:
        ph(f"\n--- {BENCHMARK}/{subset} ---")
        corpus, queries, qrels = load_task(BENCHMARK, subset)
        n_qrels = sum(len(v) for v in qrels.values())
        ps(f"Loaded {len(corpus)} docs, {len(queries)} queries, {n_qrels} qrel pairs")

        retriever = build_retriever(corpus)

        # Build scenarios via ConfigLoader (registry-backed, env-configurable)
        scenarios = ConfigLoader.create_controllers_from_env(retriever)

        for name, controller in scenarios:
            ps(f"  Running [{name}]...")
            t0 = time.time()
            results = {}
            for qid, qtext in queries.items():
                try:
                    out = controller.run(sanitize_query(qtext))
                    results[qid] = {
                        d.id: 1.0 / (i + 1) for i, d in enumerate(out.documents)
                    }
                except Exception as e:
                    _console.print(f"  [yellow]ERR {qid}: {e}[/yellow]")
            elapsed = time.time() - t0
            metrics = _evaluator.evaluate(qrels, results, k_values=[EVAL_K])
            ndcg = metrics.get("ndcg", {}).get(f"NDCG@{EVAL_K}", 0)
            ps(f"  {name:28s} NDCG@{EVAL_K}={ndcg:.4f}  ({elapsed:.1f}s)")
            all_rows.append(
                {
                    "benchmark": BENCHMARK,
                    "subset": subset,
                    "scenario": name,
                    f"NDCG@{EVAL_K}": ndcg,
                    "queries": len(results),
                    "time_s": round(elapsed, 1),
                }
            )

    ph("\nFINAL RESULTS")
    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
