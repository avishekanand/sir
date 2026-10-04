"""Pipeline and evaluation tools: build controllers once, run queries, score them."""

import time
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

from ragtune.cli.config_loader import ConfigLoader
from ragtune.core.budget import CostBudget
from ragtune.core.interfaces import BaseRetriever
from ragtune.core.types import RAGtuneContext, ScoredDocument
from ragtune.mcp._common import READ_ONLY, WRITES, ConfigInput, add_tools, merge_limits, parse_config_input, to_jsonable
from ragtune.mcp.state import ServerState


class IndexRetriever(BaseRetriever):
    """Retrieves through any built index (BaseIndexer.search); text comes from a loaded corpus.

    The generalized retriever "cassette" of scripts/run_tool_retrieval.py:
    BM25, FAISS, numpy and flex indexes all plug into the same pipeline.
    """

    def __init__(self, indexer: Any, index_path: str, corpus: Optional[Dict[str, Dict]] = None, **search_params):
        self.indexer, self.index_path = indexer, index_path
        self.corpus = corpus or {}
        self.search_params = search_params

    def retrieve(self, context: RAGtuneContext, top_k: int) -> List[ScoredDocument]:
        hits = self.indexer.search(context.query, top_k=top_k, index_path=self.index_path, **self.search_params)
        docs = []
        for hit in hits:
            text = self.corpus.get(hit.doc_id, {}).get("text", "")
            docs.append(ScoredDocument(id=hit.doc_id, content=text, score=hit.score, original_score=hit.score,
                                       token_count=int(len(text.split()) * 1.3)))
        return docs


def register(mcp: MCPServer, state: ServerState) -> None:
    def index_retriever_from(spec: Dict[str, Any]) -> IndexRetriever:
        from ragtune.indexing import IndexFactory

        unknown = set(spec) - {"index_type", "index_path", "indexer_params", "dataset_id", "backend"}
        if unknown or not {"index_type", "index_path"} <= set(spec):
            raise ValueError("index_retriever takes index_type, index_path and optionally "
                             f"indexer_params, dataset_id, backend (got {sorted(spec)})")
        corpus = (state.lookup(state.datasets, "dataset_id", spec["dataset_id"])["loader"].get_corpus()
                  if spec.get("dataset_id") else None)
        search = {"backend": spec["backend"]} if spec.get("backend") else {}
        return IndexRetriever(IndexFactory.create(spec["index_type"], **(spec.get("indexer_params") or {})),
                              str(state.resolve(spec["index_path"], must_exist=True)), corpus, **search)

    def build_pipeline(config_path, config, name, limit_overrides=None, documents=None, retriever=None,
                       cost_estimation=None, initial_top_k=None) -> Dict[str, Any]:
        data = parse_config_input(state.resolve, config_path, config)  # deep copy: never mutates input
        pipeline_conf = data.setdefault("pipeline", {})
        components = pipeline_conf.setdefault("components", {})
        if documents is not None and retriever is not None:
            raise ValueError("Pass documents or index_retriever, not both.")
        if documents is not None:
            components["retriever"] = {"type": "in-memory", "params": {"documents": documents}}
        if retriever is not None:
            components["retriever"] = retriever  # instances pass through ConfigLoader unchanged
        budget = pipeline_conf.setdefault("budget", {})
        budget["limits"] = merge_limits(budget.get("limits", {}), limit_overrides)

        controller = ConfigLoader.create_controller(data)
        if initial_top_k is not None:
            controller.initial_top_k = initial_top_k
        if cost_estimation:
            from ragtune.budget import BudgetLoaderFactory

            spec = dict(cost_estimation)
            path = spec.get("config_path")
            controller.cost_loader = BudgetLoaderFactory.create(
                spec.get("budget_type", "vllm"), config=spec.get("config"),
                config_path=str(state.resolve(path, must_exist=True)) if path else None)
            controller.cost_config = spec.get("cost_config") or {}

        handle = {"id": state.new_id("pipeline"), "controller": controller, "runs": 0,
                  "name": name or pipeline_conf.get("name") or config_path or "inline"}
        state.pipelines[handle["id"]] = handle
        return handle

    def summary(handle: Dict[str, Any]) -> Dict[str, Any]:
        c = handle["controller"]
        parts = {k: type(getattr(c, k)).__name__ for k in
                 ("retriever", "reformulator", "estimator", "scheduler", "reranker", "assembler")}
        if c.feedback is not None:
            parts["feedback"] = type(c.feedback).__name__
        return {"pipeline_id": handle["id"], "name": handle["name"], "components": parts,
                "limits": c.budget.limits, "initial_top_k": c.initial_top_k,
                "cost_estimation": type(c.cost_loader).__name__ if c.cost_loader else None, "runs": handle["runs"]}

    def controller_for(pipeline_id, config_path, config) -> Dict[str, Any]:
        if pipeline_id is not None:
            return state.lookup(state.pipelines, "pipeline_id", pipeline_id)
        return build_pipeline(config_path, config, None)

    def create_pipeline(
        config_path: Optional[str] = None,
        config: Optional[ConfigInput] = None,
        name: Optional[str] = None,
        limit_overrides: Optional[Dict[str, Optional[float]]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        index_retriever: Optional[Dict[str, Any]] = None,
        cost_estimation: Optional[Dict[str, Any]] = None,
        initial_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build a pipeline (controller) once and keep it as a pipeline_id; models load here.

        - limit_overrides: budget limits on top of the config's; null removes one.
        - documents: inline corpus [{"id", "content", "score"?}] via the in-memory retriever.
        - index_retriever: {"index_type", "index_path", "indexer_params"?, "dataset_id"?,
          "backend"?}: retrieve through a built index (see build_index); dataset_id
          supplies document text for reranking.
        - cost_estimation: {"budget_type", "config"?, "config_path"?, "cost_config"?}:
          attach a cost loader so each run reports total_cost_usd/energy/carbon.
        - initial_top_k: first-stage retrieval depth.
        """
        retriever = index_retriever_from(index_retriever) if index_retriever is not None else None
        return summary(build_pipeline(config_path, config, name, limit_overrides, documents,
                                      retriever, cost_estimation, initial_top_k))

    def run_pipeline(
        query: str,
        pipeline_id: Optional[str] = None,
        config_path: Optional[str] = None,
        config: Optional[ConfigInput] = None,
        limit_overrides: Optional[Dict[str, Optional[float]]] = None,
        include_trace: bool = False,
        max_chars: int = 300,
    ) -> Dict[str, Any]:
        """Run one query: `ragtune run`. Returns ranked documents and the final budget state.

        Use pipeline_id from create_pipeline, or pass a config (a pipeline is
        created and its pipeline_id returned for reuse). limit_overrides apply
        to this run only. include_trace returns the controller's decision log.
        """
        handle = controller_for(pipeline_id, config_path, config)
        controller = handle["controller"]
        override = (CostBudget(limits=merge_limits(controller.budget.limits, limit_overrides))
                    if limit_overrides else None)
        start = time.time()
        output = controller.run(query, override_budget=override)
        handle["runs"] += 1
        result: Dict[str, Any] = {
            "pipeline_id": handle["id"],
            "query": query,
            "elapsed_ms": round((time.time() - start) * 1000, 1),
            "documents": [
                {"rank": i, "id": d.id, "score": d.score, "reranker_score": d.reranker_score,
                 "original_score": d.original_score, "token_count": d.token_count,
                 "content": d.content[:max_chars], "metadata": to_jsonable(d.metadata)}
                for i, d in enumerate(output.documents)
            ],
            "budget_state": to_jsonable(output.final_budget_state),
        }
        if include_trace:
            result["trace"] = to_jsonable([e.model_dump(exclude={"timestamp"}) for e in output.trace.events])
        return result

    def list_pipelines() -> Dict[str, Any]:
        """Pipelines built in this server."""
        return {"pipelines": [summary(h) for h in state.pipelines.values()]}

    def close_pipeline(pipeline_id: str) -> Dict[str, Any]:
        """Drop a pipeline and the models it holds."""
        state.lookup(state.pipelines, "pipeline_id", pipeline_id)
        del state.pipelines[pipeline_id]
        return {"closed": pipeline_id}

    add_tools(mcp, READ_ONLY, list_pipelines)
    add_tools(mcp, WRITES, create_pipeline, run_pipeline, close_pipeline)

    # ── Evaluation ───────────────────────────────────────────────────────

    def evaluate_run(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: Optional[List[int]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Score a run against qrels: NDCG, MAP, Recall, Precision at k, and MRR.

        results maps query_id -> {doc_id: score}. metrics narrows the output,
        e.g. ["ndcg", "mrr"]. Default k_values: 1, 3, 5, 10, 50, 100.
        """
        from ragtune.evaluation import RetrievalEvaluator

        scores = RetrievalEvaluator(k_values=k_values).evaluate(qrels, results)
        if metrics:
            unknown = set(m.lower() for m in metrics) - set(scores)
            if unknown:
                raise ValueError(f"Unknown metrics {sorted(unknown)}. Valid: {sorted(scores)}")
            scores = {m: scores[m] for m in (m.lower() for m in metrics)}
        return {"metrics": scores}

    def score(controller, dataset_id, max_queries, query_ids, k_values, limit_overrides) -> Dict[str, Any]:
        from ragtune.evaluation import RetrievalEvaluator

        loader = state.lookup(state.datasets, "dataset_id", dataset_id)["loader"]
        queries, qrels = loader.get_queries(), loader.get_qrels()
        ids = [q for q in (query_ids or list(queries)) if q in queries][:max_queries or None]
        override = (CostBudget(limits=merge_limits(controller.budget.limits, limit_overrides))
                    if limit_overrides else None)
        results, errors, totals = {}, [], {}
        start = time.time()
        for qid in ids:
            try:
                output = controller.run(queries[qid], override_budget=override)
            except Exception as e:
                errors.append({"query_id": qid, "error": f"{type(e).__name__}: {e}"})
                continue
            # Rank-based scores, as in scripts/run_tool_retrieval.py
            results[qid] = {d.id: 1.0 / (i + 1) for i, d in enumerate(output.documents)}
            for key, value in output.final_budget_state.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[key] = totals.get(key, 0.0) + value
        metrics = RetrievalEvaluator(k_values=k_values or [10]).evaluate(
            {q: qrels.get(q, {}) for q in results}, results) if results else {}
        return {"queries_evaluated": len(results), "errors": errors[:10], "error_count": len(errors),
                "metrics": metrics, "time_s": round(time.time() - start, 2),
                "avg_budget": {k: round(v / len(results), 4) for k, v in sorted(totals.items())} if results else {}}

    def evaluate_pipeline(
        dataset_id: str,
        pipeline_id: Optional[str] = None,
        config_path: Optional[str] = None,
        config: Optional[ConfigInput] = None,
        max_queries: Optional[int] = None,
        query_ids: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
        limit_overrides: Optional[Dict[str, Optional[float]]] = None,
        background: bool = False,
    ) -> Dict[str, Any]:
        """Run a pipeline over a loaded dataset's queries and score it against its qrels.

        Reports metrics, mean budget usage per key (rerank_docs, tokens, any
        tracked key), and per-query errors. The pipeline's retriever must
        search this dataset's corpus, e.g. create_pipeline(index_retriever=...).
        Use background=True for more than a handful of queries.
        """
        handle = controller_for(pipeline_id, config_path, config)
        report = lambda: {"pipeline_id": handle["id"], "dataset_id": dataset_id,  # noqa: E731
                          **score(handle["controller"], dataset_id, max_queries, query_ids, k_values, limit_overrides)}
        return state.run_or_background(background, f"evaluate_pipeline {handle['id']} on {dataset_id}", report)

    def evaluate_scenarios(
        dataset_id: str,
        index_retriever: Dict[str, Any],
        scenarios: Optional[List[Dict[str, Any]]] = None,
        max_queries: Optional[int] = None,
        k_values: Optional[List[int]] = None,
        initial_top_k: Optional[int] = None,
        limit_overrides: Optional[Dict[str, Optional[float]]] = None,
        background: bool = False,
    ) -> Dict[str, Any]:
        """Benchmark several pipeline configs on one dataset with a shared retriever.

        The in-process equivalent of scripts/run_tool_retrieval.py. scenarios
        defaults to list_default_scenarios() (BM25 + 6 cross-encoder variants);
        each is {"name", "pipeline": {...}}. Returns one row per scenario.
        """
        configs = scenarios or ConfigLoader._default_scenarios()
        retriever = index_retriever_from(index_retriever)

        def run():
            rows = []
            for idx, scenario in enumerate(configs):
                pipeline_conf = {k: v for k, v in scenario.items() if k != "name"}
                handle = build_pipeline(None, pipeline_conf if "pipeline" in pipeline_conf else {"pipeline": pipeline_conf},
                                        scenario.get("name", f"scenario_{idx + 1}"), limit_overrides,
                                        retriever=retriever, initial_top_k=initial_top_k)
                try:
                    rows.append({"scenario": handle["name"], "pipeline_id": handle["id"],
                                 **score(handle["controller"], dataset_id, max_queries, None, k_values, None)})
                finally:
                    state.pipelines.pop(handle["id"], None)  # scenarios are throwaway
            return {"dataset_id": dataset_id, "scenarios": rows}

        return state.run_or_background(background, f"evaluate_scenarios on {dataset_id}", run)

    add_tools(mcp, READ_ONLY, evaluate_run)
    add_tools(mcp, WRITES, evaluate_pipeline, evaluate_scenarios)
