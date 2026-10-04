"""ragtune.mcp pipeline and evaluation tools, end to end and offline."""

import pytest

INLINE = {"pipeline": {"name": "inline", "components": {
    "reranker": {"type": "simulated"},
    "scheduler": {"type": "active-learning", "params": {"batch_size": 2}},
    "assembler": {"type": "greedy", "params": {"max_docs": 3}},
}, "budget": {"limits": {"rerank_docs": 4, "latency_ms": 60000}}}}
DOCUMENTS = [{"id": f"d{i}", "content": f"fox document {i}", "score": 1 - i / 10} for i in range(6)]


@pytest.fixture
def bm25(mcp_server, local_dataset):
    pytest.importorskip("pyterrier")
    mcp_server.call("build_index", index_type="pyterrier", index_path="indexes/bm25", dataset_id=local_dataset)
    return {"index_type": "pyterrier", "index_path": "indexes/bm25", "dataset_id": local_dataset}


def test_inline_pipeline_run_with_per_run_overrides_and_trace(mcp_server):
    created = mcp_server.call("create_pipeline", config=INLINE, documents=DOCUMENTS, limit_overrides={"tokens": None})
    pid = created["pipeline_id"]
    assert created["components"]["reranker"] == "SimulatedReranker" and "tokens" not in created["limits"]

    run = mcp_server.call("run_pipeline", pipeline_id=pid, query="fox", include_trace=True)
    assert run["budget_state"]["rerank_docs"] == 4 and len(run["documents"]) == 3
    assert run["documents"][0]["reranker_score"] == 0.95
    assert any(e["action"] == "rerank_batch" for e in run["trace"])

    tighter = mcp_server.call("run_pipeline", pipeline_id=pid, query="fox", limit_overrides={"rerank_docs": 2})
    assert tighter["budget_state"]["rerank_docs"] == 2
    assert mcp_server.call("list_pipelines")["pipelines"][0]["runs"] == 2

    mcp_server.call("close_pipeline", pipeline_id=pid)
    assert "Unknown pipeline_id" in mcp_server.error("run_pipeline", pipeline_id=pid, query="fox")


def test_run_pipeline_with_inline_config_creates_a_reusable_pipeline(mcp_server):
    config = {"pipeline": {**INLINE["pipeline"], "components": {
        **INLINE["pipeline"]["components"], "retriever": {"type": "in-memory", "params": {"documents": DOCUMENTS}}}}}
    first = mcp_server.call("run_pipeline", query="fox", config=config)
    second = mcp_server.call("run_pipeline", query="fox", pipeline_id=first["pipeline_id"])
    assert [d["id"] for d in first["documents"]] == [d["id"] for d in second["documents"]]


def test_cost_estimation_is_attached_to_the_controller(mcp_server):
    pid = mcp_server.call("create_pipeline", config=INLINE, documents=DOCUMENTS,
                          cost_estimation={"budget_type": "token"})["pipeline_id"]
    state = mcp_server.call("run_pipeline", pipeline_id=pid, query="fox")["budget_state"]
    assert state["total_cost_usd"] > 0 and len(state["iteration_costs"]) == 2


def test_bad_pipeline_inputs_explain_themselves(mcp_server, bm25):
    assert "not both" in mcp_server.error("create_pipeline", config=INLINE, documents=DOCUMENTS, index_retriever=bm25)
    assert "index_retriever takes" in mcp_server.error("create_pipeline", config=INLINE, index_retriever={"index": "x"})
    assert "not found for category 'reranker'" in mcp_server.error(
        "create_pipeline", config={"pipeline": {"components": {"reranker": {"type": "nope"}}}}, documents=[])


def test_index_backed_pipeline_runs_and_evaluates(mcp_server, local_dataset, bm25):
    pid = mcp_server.call("create_pipeline", config=INLINE, index_retriever=bm25)["pipeline_id"]
    run = mcp_server.call("run_pipeline", pipeline_id=pid, query="reranking budget")
    assert run["documents"][0]["id"] in {"d1", "d2"} and run["documents"][0]["content"]  # text joined from dataset

    report = mcp_server.call("evaluate_pipeline", dataset_id=local_dataset, pipeline_id=pid, k_values=[1, 10])
    assert report["queries_evaluated"] == 2 and report["error_count"] == 0
    assert report["metrics"]["ndcg"]["NDCG@10"] > 0.5
    assert report["avg_budget"]["rerank_docs"] > 0


def test_evaluate_scenarios_compares_configs_with_a_shared_retriever(mcp_server, local_dataset, bm25):
    scenarios = [
        {"name": "no_rerank", "pipeline": {"components": {"reranker": {"type": "noop"}},
                                           "budget": {"limits": {"rerank_docs": 0}}}},
        {"name": "simulated", "pipeline": {"components": {"reranker": {"type": "simulated"}},
                                           "budget": {"limits": {"rerank_docs": 4}}}},
    ]
    rows = mcp_server.call("evaluate_scenarios", dataset_id=local_dataset, index_retriever=bm25,
                           scenarios=scenarios)["scenarios"]
    assert [r["scenario"] for r in rows] == ["no_rerank", "simulated"]
    assert rows[0]["avg_budget"].get("rerank_docs", 0) == 0 and rows[1]["avg_budget"]["rerank_docs"] > 0
    assert mcp_server.call("list_pipelines")["pipelines"] == []  # throwaway pipelines are closed


def test_evaluate_run_filters_metrics(mcp_server):
    pytest.importorskip("pytrec_eval")
    scores = mcp_server.call("evaluate_run", qrels={"q": {"a": 1}}, results={"q": {"a": 1.0, "b": 0.5}},
                             k_values=[1], metrics=["mrr"])["metrics"]
    assert scores == {"mrr": {"MRR": 1.0}}
    assert "Unknown metrics" in mcp_server.error("evaluate_run", qrels={}, results={}, metrics=["f1"])
