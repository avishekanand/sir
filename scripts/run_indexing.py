"""
Build (or reuse) an index from a bright_indexing.yaml, optionally run a query.

Usage
-----
    python scripts/run_indexing.py src/ragtune/indexing/configs/bright_indexing.yaml
    python scripts/run_indexing.py src/ragtune/indexing/configs/bright_indexing.yaml --dry-run
    python scripts/run_indexing.py src/ragtune/indexing/configs/bright_indexing.yaml \\
        --query "what is the role of mitochondria in apoptosis?" --top-k 5

If the index already exists at index_path, the corpus is not re-downloaded
and the build step is skipped entirely — --query alone is enough to test
retrieval against a previously built index.
"""

import argparse
import logging
import os
import sys
import time
import yaml

# Ensure both `ragtune` (installed) and `src.ragtune` (data-loader imports) resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_corpus(data_cfg: dict):
    from src.ragtune.data.loaders.RetrieverDataset import RetrieverDataset

    benchmark = data_cfg["benchmark"]
    dataset   = data_cfg["dataset"]
    split     = data_cfg.get("split", "test")
    long_ctx  = data_cfg.get("long_context", False)

    log.info(f"Loading corpus — benchmark={benchmark!r}  dataset={dataset!r}  split={split!r}")
    rd = RetrieverDataset(dataset=dataset, benchmark=benchmark, split=split, long_context=long_ctx)
    _, _, corpus = rd.qrels()
    log.info(f"Corpus loaded: {len(corpus):,} documents")
    return corpus


def _get_indexer(index_cfg: dict):
    from ragtune.indexing import IndexFactory
    from ragtune.config.models import IndexConfig, ModelConfig

    idx_type   = index_cfg["type"]
    index_path = index_cfg["index_path"]
    model_cfg  = index_cfg.get("model") or {}
    params_cfg = index_cfg.get("params") or {}

    cfg = IndexConfig(
        type       = idx_type,
        index_path = index_path,
        backend    = index_cfg.get("backend"),
        model      = ModelConfig(name=model_cfg.get("name")) if model_cfg else None,
        params     = params_cfg,
    )
    indexer = IndexFactory.from_config(cfg)

    log.info(f"Index type   : {idx_type}")
    log.info(f"Indexer      : {indexer.__class__.__name__}")
    if idx_type == "dense":
        log.info(f"Backend      : {index_cfg.get('backend')}")
        log.info(f"Model        : {model_cfg.get('name')}")
        log.info(f"Params       : {params_cfg}")
    log.info(f"Index path   : {index_path}")

    return indexer, index_path


def _build_index(indexer, index_path: str, corpus: dict):
    log.info("Building index …")
    t0 = time.time()
    indexer.build_from_corpus(corpus, index_path)
    elapsed = time.time() - t0
    log.info(f"Index built in {elapsed:.1f}s  →  {os.path.abspath(index_path)}")


def _run_search(indexer, index_path: str, query: str, top_k: int):
    log.info(f"Searching top_k={top_k} for query={query!r}")
    t0 = time.time()
    results = indexer.search(query, top_k=top_k, index_path=index_path)
    elapsed = time.time() - t0
    log.info(f"Search took {elapsed * 1000:.1f}ms")
    print(f"\nTop {len(results)} results for: {query!r}\n")
    for rank, hit in enumerate(results, start=1):
        print(f"  {rank:>2}. {hit.doc_id}   score={hit.score:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Build (or reuse) an index from an indexing YAML config.")
    parser.add_argument("config", help="Path to indexing YAML (e.g. src/ragtune/indexing/configs/bright_indexing.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config but skip index build/search.")
    parser.add_argument("--query", type=str, default=None, help="Run a search query against the (built or existing) index.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return for --query.")
    args = parser.parse_args()

    cfg       = _load_yaml(args.config)
    index_cfg = cfg["index"]
    data_cfg  = index_cfg["data"]

    indexer, index_path = _get_indexer(index_cfg)

    if args.dry_run:
        log.info("[dry-run] Skipping build/search — exiting.")
        return

    if indexer.exists(index_path):
        log.info("Index already exists at target path — skipping corpus load and build.")
    else:
        corpus = _load_corpus(data_cfg)
        _build_index(indexer, index_path, corpus)

    if args.query:
        _run_search(indexer, index_path, args.query, args.top_k)


if __name__ == "__main__":
    main()
