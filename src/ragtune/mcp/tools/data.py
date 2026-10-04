"""Dataset and index tools: load benchmarks, inspect them, build and search indexes."""

import csv
import json
import shutil
import time
from typing import Any, Dict, List, Optional

import yaml

from mcp.server.mcpserver import MCPServer

from ragtune.data.constants import (
    BRIGHT_TASKS, FRESHSTACK_TOPICS, SRA_BENCH_SUBSETS, TOOLRET_SUBSETS, HFDatasets,
)
from ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from ragtune.indexing.base import BaseIndexer
from ragtune.mcp._common import DESTRUCTIVE, READ_ONLY, WRITES, add_tools, to_jsonable
from ragtune.mcp.state import ServerState

# benchmark -> how load_dataset reaches it. "factory" entries go through DataLoaderFactory.
BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "bright": {"route": "factory", "datasets": BRIGHT_TASKS, "options": ["long_context", "reasoning_subset"]},
    "beir": {"route": "factory", "datasets": "any mteb/<name> on HuggingFace, e.g. 'scifact'",
             "options": ["hf_dataset_name", "subset"]},
    "freshstack": {"route": "factory", "datasets": FRESHSTACK_TOPICS},
    "toolret": {"route": "factory", "datasets": TOOLRET_SUBSETS},
    "skillret": {"route": "factory", "datasets": ["test"], "options": ["corpus_fields", "corpus_sep", "min_relevance"]},
    "sra_bench": {"route": "factory", "datasets": SRA_BENCH_SUBSETS},
    "crumb": {"route": "CRUMBLoader", "datasets": None, "options": []},
    "obliq": {"route": "OBLIQLoader", "datasets": None, "options": []},
    "irds": {"route": "IRDatasetsLoader", "datasets": "any ir_datasets id, e.g. 'beir/scifact/test'"},
    "hf": {"route": "HuggingFaceLoader", "datasets": "any HuggingFace dataset in BEIR layout",
           "options": ["subset", "corpus_split", "queries_split", "qrels_split", "id_col", "text_col"]},
    "local": {"route": "files", "datasets": "corpus_path (+ queries_path, qrels_path) in the workspace"},
}


# Factory benchmarks whose loaders accept n_queries.
N_QUERIES_BENCHMARKS = {"toolret", "skillret", "sra_bench"}


class LocalDataLoader(BaseDataLoader):
    """Corpus JSON/JSONL, queries JSONL ({id, text}), qrels TSV or JSONL from the workspace."""

    def __init__(self, corpus_path, queries_path=None, qrels_path=None,
                 id_field="doc_id", text_field="text", title_field="title"):
        super().__init__(dataset=str(corpus_path), split="local")
        self.paths = (corpus_path, queries_path, qrels_path)
        self.fields = {"id_field": id_field, "text_field": text_field, "title_field": title_field}

    def _load_data(self) -> None:
        corpus_path, queries_path, qrels_path = self.paths
        fmt = "jsonl" if str(corpus_path).endswith(".jsonl") else "json"
        for row in BaseIndexer._iter_file(str(corpus_path), fmt):
            doc_id = str(row.get(self.fields["id_field"], row.get("_id", row.get("id", ""))))
            self._corpus[doc_id] = {"text": str(row.get(self.fields["text_field"], "")),
                                    "title": str(row.get(self.fields["title_field"], ""))}
        if queries_path:
            for row in BaseIndexer._iter_file(str(queries_path), "jsonl"):
                qid = str(row.get("id", row.get("_id", row.get("query_id", ""))))
                self._queries[qid] = str(row.get("text", row.get("query", "")))
        if qrels_path:
            if str(qrels_path).endswith(".jsonl"):
                rows = list(BaseIndexer._iter_file(str(qrels_path), "jsonl"))
            else:  # BEIR-style TSV with a header: query-id, corpus-id, score
                with open(qrels_path, newline="") as f:
                    rows = list(csv.DictReader(f, delimiter="\t"))
            for row in rows:
                qid = str(row.get("query-id", row.get("query_id")))
                did = str(row.get("corpus-id", row.get("doc_id", row.get("corpus_id"))))
                self._qrels.setdefault(qid, {})[did] = int(row.get("score", row.get("relevance", 1)))


def _make_loader(state: ServerState, benchmark: str, dataset: str, split: str,
                 max_queries: Optional[int], max_corpus_docs: Optional[int], options: Dict[str, Any]):
    key = benchmark.lower()
    if key == "crumb":
        from ragtune.data.loaders import CRUMBLoader
        return CRUMBLoader(task=dataset, split=split, max_queries=max_queries, max_corpus_docs=max_corpus_docs, **options)
    if key == "obliq":
        from ragtune.data.loaders import OBLIQLoader
        return OBLIQLoader(task=dataset, split=split, max_queries=max_queries, max_corpus_docs=max_corpus_docs, **options)
    if key == "irds":
        from ragtune.data.loaders import IRDatasetsLoader
        return IRDatasetsLoader(dataset_id=dataset, split=split, **options)
    if key == "hf":
        from ragtune.data.loaders import HuggingFaceLoader
        return HuggingFaceLoader(hf_dataset_name=dataset, split=split, **options)
    if key == "local":
        resolved = {k: state.resolve(v, must_exist=True) for k, v in options.items() if k.endswith("_path") and v}
        rest = {k: v for k, v in options.items() if not k.endswith("_path")}
        if "corpus_path" not in resolved:
            raise ValueError("benchmark='local' needs options.corpus_path (plus optional queries_path, qrels_path)")
        return LocalDataLoader(**resolved, **rest)
    if key not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark!r}. Valid: {sorted(BENCHMARKS)}")
    from ragtune.data.loaders import DataLoaderFactory
    # The factory forwards extra kwargs only where the loader accepts them.
    if max_queries:
        if key not in N_QUERIES_BENCHMARKS:
            raise ValueError(f"max_queries is supported for {sorted(N_QUERIES_BENCHMARKS | {'crumb', 'obliq'})}; "
                             "use evaluate_pipeline(max_queries=...) to evaluate a subset instead")
        options = {"n_queries": max_queries, **options}
    if max_corpus_docs:
        raise ValueError("max_corpus_docs is supported for crumb and obliq only")
    return DataLoaderFactory().create_dataloader(dataset_name=dataset, benchmark_name=benchmark, split=split, **options)


def _stats(handle: Dict[str, Any]) -> Dict[str, Any]:
    loader = handle["loader"]
    corpus, queries, qrels = loader.get_corpus(), loader.get_queries(), loader.get_qrels()
    return {"dataset_id": handle["id"], "benchmark": handle["benchmark"], "dataset": handle["dataset"],
            "split": handle["split"], "documents": len(corpus), "queries": len(queries),
            "qrel_pairs": sum(len(v) for v in qrels.values()), "load_s": handle.get("load_s")}


def register(mcp: MCPServer, state: ServerState) -> None:
    def dataset(dataset_id: str) -> Dict[str, Any]:
        return state.lookup(state.datasets, "dataset_id", dataset_id)

    def list_benchmarks() -> Dict[str, Any]:
        """Benchmarks load_dataset understands, with their datasets and extra options.

        CRUMB/OBLIQ task lists and HuggingFace dataset ids are included.
        """
        from ragtune.data.loaders import CRUMB_TASKS, OBLIQ_TASKS

        catalog = {name: dict(info) for name, info in BENCHMARKS.items()}
        catalog["crumb"]["datasets"], catalog["obliq"]["datasets"] = CRUMB_TASKS, OBLIQ_TASKS
        hf_ids = {k: v for k, v in vars(HFDatasets).items() if not k.startswith("_")}
        return {"benchmarks": catalog, "huggingface_ids": hf_ids}

    def load_dataset(
        benchmark: str,
        dataset: str = "",
        split: str = "test",
        max_queries: Optional[int] = None,
        max_corpus_docs: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        background: bool = False,
    ) -> Dict[str, Any]:
        """Load a benchmark split and keep it in memory as a dataset_id.

        See list_benchmarks for names. Downloads (HuggingFace, ir_datasets) can
        take minutes: pass background=True and poll job_status. For your own
        files use benchmark="local", options={"corpus_path": ..., "queries_path":
        ..., "qrels_path": ...}.
        """
        loader = _make_loader(state, benchmark, dataset, split, max_queries, max_corpus_docs, dict(options or {}))
        handle = {"id": state.new_id("dataset"), "loader": loader, "benchmark": benchmark,
                  "dataset": dataset, "split": split}

        def load():
            start = time.time()
            loader.load()  # triggers the lazy download/parse
            handle["load_s"] = round(time.time() - start, 2)
            state.datasets[handle["id"]] = handle
            return _stats(handle)

        return state.run_or_background(background, f"load_dataset {benchmark}/{dataset}", load)

    def list_datasets() -> Dict[str, Any]:
        """Datasets loaded in this server, with sizes."""
        return {"datasets": [_stats(h) for h in state.datasets.values()]}

    def get_queries(dataset_id: str, offset: int = 0, limit: int = 20, with_qrels: bool = False) -> Dict[str, Any]:
        """Page through queries; with_qrels adds relevant doc ids. BRIGHT queries include 'reasoning'."""
        loader = dataset(dataset_id)["loader"]
        queries, qrels = loader.get_queries(), loader.get_qrels()
        reasoning = {str(q.id()): q.reasoning for q in loader.get_query_objects() if getattr(q, "reasoning", None)}
        page = []
        for qid in list(queries)[offset:offset + limit]:
            entry: Dict[str, Any] = {"query_id": qid, "text": queries[qid]}
            if qid in reasoning:
                entry["reasoning"] = reasoning[qid]
            if with_qrels:
                entry["qrels"] = qrels.get(qid, {})
            page.append(entry)
        return {"total": len(queries), "offset": offset, "queries": page}

    def get_documents(
        dataset_id: str, doc_ids: Optional[List[str]] = None, offset: int = 0, limit: int = 10, max_chars: int = 1000
    ) -> Dict[str, Any]:
        """Fetch documents by id, or page through the corpus. Text is cut to max_chars."""
        corpus = dataset(dataset_id)["loader"].get_corpus()
        ids = doc_ids if doc_ids is not None else list(corpus)[offset:offset + limit]
        missing = [d for d in ids if d not in corpus]
        docs = [{"doc_id": d, "title": corpus[d].get("title", ""), "text": corpus[d].get("text", "")[:max_chars]}
                for d in ids if d in corpus]
        return {"total": len(corpus), "documents": docs, "missing": missing}

    def get_qrels(dataset_id: str, query_id: Optional[str] = None, kind: str = "qrels") -> Dict[str, Any]:
        """Relevance judgments. kind: 'qrels' (default), 'excluded_ids' (BRIGHT, OBLIQ: docs
        never to retrieve), or 'nugget_qrels' (FreshStack nugget-level judgments)."""
        loader = dataset(dataset_id)["loader"]
        if kind == "qrels":
            data: Any = loader.get_qrels()
        elif kind == "excluded_ids" and hasattr(loader, "get_excluded_ids"):
            data = loader.get_excluded_ids()
        elif kind == "nugget_qrels" and hasattr(loader, "load_nugget_qrels"):
            nuggets, _, query_to_nuggets = loader.load_nugget_qrels()
            if query_id is not None:
                ids = query_to_nuggets.get(query_id, [])
                return {"query_id": query_id, "nuggets": {n: nuggets.get(n, {}) for n in ids}}
            return {"nugget_qrels": nuggets, "query_to_nuggets": query_to_nuggets}
        else:
            raise ValueError(f"kind={kind!r} is not available for {type(loader).__name__}")
        if query_id is not None:
            return {"query_id": query_id, kind: to_jsonable(data.get(query_id, {}))}
        return {kind: to_jsonable(data)}

    def export_corpus(dataset_id: str, output_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """Write the corpus as JSONL ({doc_id, text, title}) for `ragtune index` or a config's data section."""
        target = state.resolve(output_path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"{output_path!r} exists; pass overwrite=True")
        target.parent.mkdir(parents=True, exist_ok=True)
        corpus = dataset(dataset_id)["loader"].get_corpus()
        with open(target, "w") as f:
            for doc_id, doc in corpus.items():
                f.write(json.dumps({"doc_id": doc_id, "text": doc.get("text", ""), "title": doc.get("title", "")}) + "\n")
        return {"written_to": state.relative(target), "documents": len(corpus)}

    def drop_dataset(dataset_id: str) -> Dict[str, Any]:
        """Free a loaded dataset's memory."""
        dataset(dataset_id)
        del state.datasets[dataset_id]
        return {"dropped": dataset_id}

    add_tools(mcp, READ_ONLY, list_benchmarks, list_datasets, get_queries, get_documents, get_qrels)
    add_tools(mcp, WRITES, load_dataset, export_corpus, drop_dataset)

    # ── Indexing ─────────────────────────────────────────────────────────

    def make_indexer(index_type: str, indexer_params: Optional[Dict[str, Any]]):
        from ragtune.indexing import IndexFactory

        return IndexFactory.create(index_type, **(indexer_params or {}))

    def build(indexer, index_path: str, overwrite: bool, corpus_fn, description: str, background: bool):
        target = state.resolve(index_path)
        if indexer.exists(str(target)) and not overwrite:
            return {"index_path": state.relative(target), "status": "exists",
                    "next": "pass overwrite=True to rebuild"}

        def run():
            if overwrite and target.exists():
                shutil.rmtree(target)  # some backends (flex) refuse to write into an existing dir
            start = time.time()
            corpus = corpus_fn()
            indexer.build_from_corpus(corpus, index_path=str(target))
            return {"index_path": state.relative(target), "status": "built", "documents": len(corpus),
                    "build_s": round(time.time() - start, 2)}

        return state.run_or_background(background, description, run)

    def build_index(
        index_type: str,
        index_path: str,
        dataset_id: Optional[str] = None,
        collection_path: Optional[str] = None,
        collection_format: str = "jsonl",
        id_field: str = "doc_id",
        text_field: str = "text",
        indexer_params: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        background: bool = False,
    ) -> Dict[str, Any]:
        """Build an index from a loaded dataset or a JSON/JSONL file.

        index_type: pyterrier (BM25), faiss, numpy, flex (see list_components
        category='indexer'). indexer_params go to the constructor, e.g.
        {"model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2"} for
        faiss/numpy or {"model_name": "bge-m3"} for flex. An existing index is
        reused unless overwrite=True.
        """
        if (dataset_id is None) == (collection_path is None):
            raise ValueError("Pass exactly one of dataset_id or collection_path.")
        indexer = make_indexer(index_type, indexer_params)
        if dataset_id is not None:
            loader = dataset(dataset_id)["loader"]
            corpus_fn = loader.get_corpus
        else:
            source = str(state.resolve(collection_path, must_exist=True))
            fields = {"id_field": id_field, "text_field": text_field}
            corpus_fn = lambda: indexer._load_file_to_corpus(source, collection_format, fields)  # noqa: E731
        return build(indexer, index_path, overwrite, corpus_fn, f"build_index {index_type} {index_path}", background)

    def build_index_from_config(
        config_path: str, collection_path: Optional[str] = None, overwrite: bool = False, background: bool = False
    ) -> Dict[str, Any]:
        """Run `ragtune index`: build the index described by a config's data and index sections.

        index.type 'sparse' builds PyTerrier BM25; 'dense' uses index.backend
        and index.model.name. The target is index.index_path (or the legacy
        index.params.index_path written by `ragtune init`).
        """
        from ragtune.config.models import RAGtuneConfig
        from ragtune.indexing import IndexFactory

        pipeline = RAGtuneConfig(**yaml.safe_load(state.resolve(config_path, must_exist=True).read_text())).pipeline
        if not pipeline.data or not pipeline.index:
            raise ValueError("The config needs pipeline.data and pipeline.index sections to build an index.")
        params = dict(pipeline.index.params)
        index_path = params.pop("index_path", None) or pipeline.index.index_path
        indexer = IndexFactory.from_config(pipeline.index.model_copy(update={"params": params}))
        source = str(state.resolve(collection_path or pipeline.data.collection_path, must_exist=True))
        fields = {"id_field": pipeline.data.id_field, "text_field": pipeline.data.text_field}
        corpus_fn = lambda: indexer._load_file_to_corpus(source, pipeline.data.collection_format, fields)  # noqa: E731
        return build(indexer, index_path, overwrite, corpus_fn, f"build_index_from_config {config_path}", background)

    def index_status(index_type: str, index_path: str) -> Dict[str, Any]:
        """Whether an index exists, its files, and any metadata the indexer recorded."""
        target = state.resolve(index_path)
        exists = target.exists() and make_indexer(index_type, None).exists(str(target))
        status: Dict[str, Any] = {"index_path": state.relative(target), "exists": exists}
        if target.is_dir():
            status["files"] = {p.name: p.stat().st_size for p in sorted(target.iterdir()) if p.is_file()}
            for name in ("metadata.json", "flex_metadata.json"):
                if (target / name).exists():
                    status["metadata"] = json.loads((target / name).read_text())
        return status

    def search_index(
        index_type: str,
        index_path: str,
        query: str,
        top_k: int = 10,
        indexer_params: Optional[Dict[str, Any]] = None,
        backend: Optional[str] = None,
        dataset_id: Optional[str] = None,
        max_chars: int = 300,
    ) -> Dict[str, Any]:
        """Query a built index directly (no reranking). With dataset_id, hits include document text.

        backend applies to flex indexes: np (default), torch, faiss_flat, faiss_hnsw.
        Dense indexes need the same indexer_params (model) they were built with.
        """
        target = str(state.resolve(index_path, must_exist=True))
        extra = {"backend": backend} if backend else {}
        hits = make_indexer(index_type, indexer_params).search(query, top_k=top_k, index_path=target, **extra)
        corpus = dataset(dataset_id)["loader"].get_corpus() if dataset_id else {}
        results = []
        for rank, hit in enumerate(hits):
            entry: Dict[str, Any] = {"rank": rank, "doc_id": hit.doc_id, "score": hit.score}
            if hit.doc_id in corpus:
                entry["text"] = corpus[hit.doc_id].get("text", "")[:max_chars]
            results.append(entry)
        return {"query": query, "results": results}

    add_tools(mcp, READ_ONLY, index_status, search_index)
    add_tools(mcp, DESTRUCTIVE, build_index, build_index_from_config)
