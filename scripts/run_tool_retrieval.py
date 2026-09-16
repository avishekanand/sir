"""
RAGtune Tool & Skill Retrieval Benchmark Runner (Generalized)
==============================================================
Evaluates RAGtune against baseline retrieval on ToolRet, SkillRet, and
SRA-Bench datasets.

DESIGN PRINCIPLE: This runner is fully config-driven. Nothing is hardcoded.
Every knob — benchmark, subset, query count, candidate depth, eval cutoffs,
retriever/indexer type, scenarios — comes from config (env vars or YAML).

The retriever is a "cassette": any retriever that implements the BaseRetriever
interface (or an indexer key registered with IndexFactory) can be swapped in
via config without touching code. BM25, FAISS, dense, flex — all work.

Usage (env vars):
    BENCHMARK=sra python scripts/run_tool_retrieval.py
    BENCHMARK=toolret SUBSET=apibank python scripts/run_tool_retrieval.py
    SCENARIOS='[{"name":"custom","pipeline":{"components":{"reranker":{"type":"cross-encoder"}}}}]' BENCHMARK=sra python scripts/run_tool_retrieval.py
    EVAL_KS=10,50 python scripts/run_tool_retrieval.py

Usage (config file):
    python scripts/run_tool_retrieval.py --config configs/benchmark_skillret.yaml
"""

import argparse
import os
import time
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import yaml
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


def sanitize_query(text, max_chars: int = 2000):
    """Strip LaTeX, math symbols, unicode, and special chars for PyTerrier.

    The max_chars cap (default 2000 characters) guards against pathological
    queries that could break PyTerrier's TerrierQL parser. It is configurable
    via the runner's 'max_query_chars' config key.
    """
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
    return text[:max_chars]


# ── Configuration (config-driven, nothing hardcoded) ────────────────────────

DEFAULT_CONFIG = {
    "benchmark": "toolret",  # which benchmark: toolret | skillret | sra
    "subset": "",  # specific subset; empty = all subsets
    "queries": 0,  # max queries per subset; 0 = all
    "top_k": 100,  # candidates retrieved per query
    "eval_ks": [10],  # eval cutoffs, e.g. [10, 50]
    "index_type": "pyterrier",  # indexer cassette: pyterrier | faiss | numpy | flex
    "indexer_params": {},  # kwargs for the indexer (e.g. model_name for dense)
    "index_dir": None,  # index cache dir; None = <workspace>/indexes/benchmarks
    "force_reindex": False,  # True = rebuild even if index exists
    "report_rerank": True,  # include avg rerank docs / latency in output
    "retriever": "bm25",  # retriever cassette: bm25 | faiss | langchain | custom
    "wmodel": "BM25",  # sparse retrieval model (BM25, TF_IDF, ...)
    "scenarios_env": "",  # SCENARIOS JSON (overrides defaults)
    # Which skill fields go into the corpus text. Set ["name","description"]
    # to reproduce Rahul's PR #20 corpus exactly; leave None for the rich default.
    "corpus_fields": None,
    "corpus_sep": "\n",  # field separator; "\n\n" matches Rahul's exact format
    "min_relevance": 1,  # min qrel relevance; 1 = positive-only (Rahul's PR #20)
    "max_query_chars": 2000,  # sanitize_query cap; guards PyTerrier TerrierQL
}


def _load_config(args) -> Dict[str, Any]:
    """Merge config file → env vars → defaults. Env vars win over file; CLI wins over env."""
    cfg = dict(DEFAULT_CONFIG)

    # 1. Config file (lowest priority after defaults)
    if args.config:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        for k, v in file_cfg.items():
            if k != "pipeline":  # 'pipeline' is for scenarios, handled separately
                cfg[k] = v

    # 2. Env vars (override file)
    env_map = {
        "benchmark": ("BENCHMARK", str),
        "subset": ("SUBSET", str),
        "queries": ("QUERIES", int),
        "top_k": ("TOP_K", int),
        "eval_ks": ("EVAL_KS", lambda s: [int(x) for x in s.split(",")]),
        "index_type": ("INDEX_TYPE", str),
        "retriever": ("RETRIEVER", str),
        "wmodel": ("WMODEL", str),
        "index_dir": ("INDEX_DIR", str),
        "scenarios_env": ("SCENARIOS", str),
    }
    for key, (env_name, cast) in env_map.items():
        raw = os.environ.get(env_name)
        if raw is not None and raw != "":
            try:
                cfg[key] = cast(raw)
            except (ValueError, TypeError):
                _console.print(
                    f"[yellow]Warning: bad {env_name} value '{raw}', keeping default[/yellow]"
                )

    # 3. CLI flags (override env + file)
    if args.benchmark:
        cfg["benchmark"] = args.benchmark
    if args.subset:
        cfg["subset"] = args.subset
    if args.queries is not None:
        cfg["queries"] = args.queries
    if args.top_k is not None:
        cfg["top_k"] = args.top_k
    if args.eval_ks:
        cfg["eval_ks"] = args.eval_ks
    if args.index_type:
        cfg["index_type"] = args.index_type
    if args.index_dir:
        cfg["index_dir"] = args.index_dir
    if args.force_reindex:
        cfg["force_reindex"] = True

    return cfg


# ── Data Loading (config-driven) ────────────────────────────────────────────


def load_task(
    benchmark: str,
    subset: str,
    queries: int,
    corpus_fields=None,
    corpus_sep="\n",
    min_relevance=1,
) -> Tuple:
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
        dataset_name=subset,
        benchmark_name=bm,
        n_queries=queries,
        corpus_fields=corpus_fields,
        corpus_sep=corpus_sep,
        min_relevance=min_relevance,
    )
    return loader.get_corpus(), loader.get_queries(), loader.get_qrels()


# ── Retriever Building — "cassette" model ───────────────────────────────────
# Any indexer registered with IndexFactory, or any retriever factory, can be
# swapped in via config. This is the generalization Rahul's runner lacks:
# his FAISS retriever is hardcoded; here it's a config choice.


def build_retriever(corpus, cfg: Dict[str, Any]):
    """Build a retriever from config. The retriever is a cassette.

    Supported cassettes:
      - 'bm25' (default): sparse BM25 via PyTerrier IndexFactory
      - 'faiss': dense via FaissIndexer + custom wrapper
      - 'langchain': any LangChain-compatible retriever (e.g. Rahul's FAISS)
      - custom: an object passed in via retriever_factory

    Config keys:
      index_type : IndexFactory registry key (pyterrier, faiss, numpy, flex)
      wmodel     : sparse weighting model (BM25, TF_IDF, ...)
      indexer_params : kwargs for the indexer constructor
    """
    index_type = cfg.get("index_type", "pyterrier")
    ps(f"Indexing {len(corpus)} documents via [{index_type}]...")

    # Indexes are built INSIDE the workspace (not /tmp) so they persist and
    # can be inspected/reused. Directory is configurable via 'index_dir'.
    configured_dir = cfg.get("index_dir")
    index_root = configured_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "indexes", "benchmarks"
    )
    os.makedirs(index_root, exist_ok=True)
    # Unique subdir per benchmark/subset+index_type so runs don't collide
    index_name = (
        f"{cfg.get('benchmark', 'bench')}_{cfg.get('subset', 'all')}_{index_type}"
    )
    index_path = os.path.join(index_root, index_name)
    force_reindex = cfg.get("force_reindex", False)

    if index_type == "pyterrier":
        import pyterrier as pt

        if not pt.java.started():
            pt.java.init()
        indexer = IndexFactory.create("pyterrier")
        if indexer.exists(index_path) and not force_reindex:
            ps(f"  Reusing existing index at {index_path}")
        else:
            indexer.build_from_corpus(corpus, index_path=index_path)
            ps(f"  Index built at {index_path}")
        idx_ref = pt.IndexFactory.of(index_path)
        wmodel = cfg.get("wmodel", "BM25")
        pt_retriever = pt.terrier.Retriever(
            idx_ref, wmodel=wmodel, metadata=["docno", "text"], num_results=cfg["top_k"]
        )
        return PyTerrierRetriever(pt_retriever)

    elif index_type in ("faiss", "numpy", "flex"):
        # Dense indexers — build the index, wrap into a BaseRetriever
        indexer = IndexFactory.create(index_type, **cfg.get("indexer_params", {}))
        if indexer.exists(index_path) and not force_reindex:
            ps(f"  Reusing existing index at {index_path}")
        else:
            indexer.build_from_corpus(corpus, index_path=index_path)
            ps(f"  Index built at {index_path}")

        from ragtune.core.interfaces import BaseRetriever
        from ragtune.core.types import ScoredDocument, RAGtuneContext
        from typing import List as _List

        class _IndexRetriever(BaseRetriever):
            def __init__(self, indexer, index_path, top_k):
                self._indexer = indexer
                self._index_path = index_path
                self._top_k = top_k

            def retrieve(
                self, context: RAGtuneContext, top_k: int
            ) -> _List[ScoredDocument]:
                results = self._indexer.search(
                    context.query, top_k=top_k, index_path=self._index_path
                )
                return [
                    ScoredDocument(id=r.doc_id, content="", score=r.score)
                    for r in results
                ]

        return _IndexRetriever(indexer, index_path, cfg["top_k"])

    else:
        raise ValueError(
            f"Unknown index_type: {index_type!r}. "
            f"Available: pyterrier, faiss, numpy, flex"
        )


# ── Scenario Execution ──────────────────────────────────────────────────────


def run_scenario(
    name: str,
    controller,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    eval_ks: List[int],
    report_rerank: bool,
    max_query_chars: int = 2000,
) -> Dict[str, Any]:
    """Run a controller over all queries, return metrics + telemetry."""
    ps(f"  Running [{name}]...")
    t0 = time.time()
    results: Dict[str, Dict[str, float]] = {}
    latencies: List[float] = []
    docs_reranked: List[float] = []

    for qid, qtext in queries.items():
        try:
            q_start = time.time()
            out = controller.run(sanitize_query(qtext, max_chars=max_query_chars))
            latencies.append((time.time() - q_start) * 1000)
            docs_reranked.append(out.final_budget_state.get("rerank_docs", 0))
            results[qid] = {d.id: 1.0 / (i + 1) for i, d in enumerate(out.documents)}
        except Exception as e:
            _console.print(f"  [yellow]ERR {qid}: {e}[/yellow]")
    elapsed = time.time() - t0

    evaluator = RetrievalEvaluator(k_values=eval_ks)
    metrics = evaluator.evaluate(qrels, results)

    row: Dict[str, Any] = {
        "scenario": name,
        "queries": len(results),
        "time_s": round(elapsed, 1),
    }
    # All eval cutoffs from config — nothing hardcoded
    for k in eval_ks:
        row[f"NDCG@{k}"] = round(metrics.get("ndcg", {}).get(f"NDCG@{k}", 0), 4)
        row[f"Recall@{k}"] = round(metrics.get("recall", {}).get(f"Recall@{k}", 0), 4)

    if report_rerank and latencies:
        row["avg_rerank_docs"] = round(sum(docs_reranked) / len(docs_reranked), 1)
        row["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)

    ps(
        f"  {name:28s} "
        + " ".join(f"NDCG@{k}={row[f'NDCG@{k}']:.4f}" for k in eval_ks)
        + f"  ({elapsed:.1f}s)"
    )
    return row


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generalized RAGtune benchmark runner")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to benchmark YAML config"
    )
    parser.add_argument(
        "--benchmark", type=str, default=None, help="toolret | skillret | sra"
    )
    parser.add_argument("--subset", type=str, default=None, help="Specific subset")
    parser.add_argument(
        "--queries", type=int, default=None, help="Max queries per subset (0=all)"
    )
    parser.add_argument("--top-k", type=int, default=None, help="Candidates per query")
    parser.add_argument(
        "--eval-ks", type=str, default=None, help="Comma-separated cutoffs, e.g. 10,50"
    )
    parser.add_argument(
        "--index-type", type=str, default=None, help="pyterrier | faiss | numpy | flex"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Directory to store indexes (default: <workspace>/indexes/benchmarks)",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild index even if it already exists",
    )
    args = parser.parse_args()

    cfg = _load_config(args)

    # If a config file contains scenarios (top-level 'scenarios:' key),
    # feed them through SCENARIOS so ConfigLoader picks them up.
    if args.config:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        if "scenarios" in file_cfg:
            import json as _json

            cfg["scenarios_env"] = _json.dumps(file_cfg["scenarios"])
            os.environ["SCENARIOS"] = cfg["scenarios_env"]

    BENCHMARK = cfg["benchmark"]
    SUBSET = cfg["subset"]
    QUERIES = cfg["queries"]
    TOP_K = cfg["top_k"]
    EVAL_KS = cfg["eval_ks"]

    config.set("retrieval.original_query_depth", TOP_K)
    ph("RAGtune Tool & Skill Retrieval Benchmarks (generalized runner)")
    ps(
        f"Benchmark: {BENCHMARK}  |  Subset: {SUBSET or '(all)'}  |  Queries: {QUERIES}  "
        f"|  Top-K: {TOP_K}  |  Eval-Ks: {EVAL_KS}  |  Index: {cfg['index_type']}"
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
        corpus, queries, qrels = load_task(
            BENCHMARK,
            subset,
            QUERIES,
            corpus_fields=cfg.get("corpus_fields"),
            corpus_sep=cfg.get("corpus_sep", "\n"),
            min_relevance=cfg.get("min_relevance", 1),
        )
        n_qrels = sum(len(v) for v in qrels.values())
        ps(f"Loaded {len(corpus)} docs, {len(queries)} queries, {n_qrels} qrel pairs")

        retriever = build_retriever(corpus, cfg)

        # Build scenarios via ConfigLoader (registry-backed, env-configurable)
        scenarios = ConfigLoader.create_controllers_from_env(retriever)

        for name, controller in scenarios:
            row = run_scenario(
                name,
                controller,
                queries,
                qrels,
                EVAL_KS,
                cfg["report_rerank"],
                max_query_chars=cfg.get("max_query_chars", 2000),
            )
            row.update({"benchmark": BENCHMARK, "subset": subset})
            all_rows.append(row)

    ph("\nFINAL RESULTS")
    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
