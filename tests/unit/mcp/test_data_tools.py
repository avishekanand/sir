"""ragtune.mcp dataset and index tools, fully offline (local files, BM25, fake dense encoder)."""

import json
import time

import numpy as np
import pytest
import yaml


def test_list_benchmarks_includes_direct_loaders(mcp_server):
    catalog = mcp_server.call("list_benchmarks")
    assert "paper_retrieval" in catalog["benchmarks"]["crumb"]["datasets"]
    assert catalog["huggingface_ids"]["SKILLRET_REPO"] == "ThakiCloud/SKILLRET"


def test_local_dataset_round_trip(mcp_server, local_files):
    stats = mcp_server.call("load_dataset", benchmark="local", options=local_files)
    assert (stats["documents"], stats["queries"], stats["qrel_pairs"]) == (4, 2, 3)
    ds = stats["dataset_id"]

    queries = mcp_server.call("get_queries", dataset_id=ds, with_qrels=True)
    assert queries["queries"][0] == {"query_id": "q1", "text": "reranking budget", "qrels": {"d1": 1, "d2": 1}}
    docs = mcp_server.call("get_documents", dataset_id=ds, doc_ids=["d3", "zz"], max_chars=6)
    assert docs["documents"] == [{"doc_id": "d3", "title": "", "text": "a reci"}] and docs["missing"] == ["zz"]
    assert mcp_server.call("get_qrels", dataset_id=ds, query_id="q2") == {"query_id": "q2", "qrels": {"d3": 1}}
    assert "not available for LocalDataLoader" in mcp_server.error("get_qrels", dataset_id=ds, kind="excluded_ids")

    exported = mcp_server.call("export_corpus", dataset_id=ds, output_path="out/corpus.jsonl")
    assert exported["documents"] == 4
    assert json.loads((mcp_server.root / "out/corpus.jsonl").read_text().splitlines()[0])["doc_id"] == "d1"

    assert [d["dataset_id"] for d in mcp_server.call("list_datasets")["datasets"]] == [ds]
    mcp_server.call("drop_dataset", dataset_id=ds)
    assert "Unknown dataset_id" in mcp_server.error("get_queries", dataset_id=ds)


def test_load_dataset_rejects_unsupported_options_before_downloading(mcp_server):
    assert "Unknown benchmark" in mcp_server.error("load_dataset", benchmark="nope", dataset="x")
    assert "max_queries is supported for" in mcp_server.error("load_dataset", benchmark="bright", dataset="biology", max_queries=5)
    assert "needs options.corpus_path" in mcp_server.error("load_dataset", benchmark="local")


def test_background_load_reports_result_through_job_status(mcp_server, local_files):
    started = mcp_server.call("load_dataset", benchmark="local", options=local_files, background=True)
    deadline = time.time() + 10
    while (status := mcp_server.call("job_status", job_id=started["job_id"]))["status"] == "running":
        assert time.time() < deadline
        time.sleep(0.05)
    assert status["status"] == "succeeded" and status["result"]["documents"] == 4


def test_bm25_index_build_reuse_status_and_search(mcp_server, local_dataset):
    pytest.importorskip("pyterrier")
    built = mcp_server.call("build_index", index_type="pyterrier", index_path="indexes/bm25", dataset_id=local_dataset)
    assert built["status"] == "built" and built["documents"] == 4
    again = mcp_server.call("build_index", index_type="pyterrier", index_path="indexes/bm25", dataset_id=local_dataset)
    assert again["status"] == "exists"
    assert mcp_server.call("index_status", index_type="pyterrier", index_path="indexes/bm25")["exists"] is True

    hits = mcp_server.call("search_index", index_type="pyterrier", index_path="indexes/bm25",
                           query="reranking budget", top_k=2, dataset_id=local_dataset)["results"]
    assert hits[0]["doc_id"] == "d1" and hits[0]["text"].startswith("budget aware")


def fake_encode(self, texts):
    """Deterministic bag-of-words vectors so dense indexing needs no model download."""
    vectors = np.zeros((len(texts), 32), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.lower().split():
            vectors[row, sum(map(ord, word)) % 32] += 1.0
    return vectors


def test_dense_index_from_file(mcp_server, local_files, monkeypatch):
    monkeypatch.setattr("ragtune.indexing.dense_indexer.DenseIndexer.encode_corpus", fake_encode)
    built = mcp_server.call("build_index", index_type="numpy", index_path="indexes/dense",
                            collection_path=local_files["corpus_path"])
    assert built["documents"] == 4
    status = mcp_server.call("index_status", index_type="numpy", index_path="indexes/dense")
    assert status["metadata"]["num_docs"] == 4 and "vectors.npy" in status["files"]
    hits = mcp_server.call("search_index", index_type="numpy", index_path="indexes/dense", query="tomato soup", top_k=1)
    assert hits["results"][0]["doc_id"] == "d3"
    assert "exactly one of dataset_id or collection_path" in mcp_server.error(
        "build_index", index_type="numpy", index_path="x")


def test_build_index_from_legacy_init_config(mcp_server, local_files):
    pytest.importorskip("pyterrier")
    config = {"pipeline": {
        "data": {"collection_path": local_files["corpus_path"], "collection_format": "jsonl",
                 "id_field": "doc_id", "text_field": "text"},
        "index": {"type": "sparse", "params": {"index_path": "indexes/from_config"}},  # `ragtune init` layout
    }}
    (mcp_server.root / "pipeline.yaml").write_text(yaml.safe_dump(config))
    built = mcp_server.call("build_index_from_config", config_path="pipeline.yaml")
    assert built == {**built, "index_path": "indexes/from_config", "status": "built", "documents": 4}
