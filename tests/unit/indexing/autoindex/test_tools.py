import json

import pytest

from ragtune.indexing.autoindex.bm25 import ChunkBM25Index
from ragtune.indexing.autoindex.program import Chunk
from ragtune.indexing.autoindex.tools import TOOL_SCHEMAS, AnalysisToolbox

CORPUS = {
    "d1": {"text": "the quick brown fox", "title": "Fox"},
    "d2": {"text": "lazy dogs sleep all day " * 50, "title": ""},
}


@pytest.fixture
def box():
    idx = ChunkBM25Index([Chunk(f"{d}_0", d, e["text"]) for d, e in CORPUS.items()], backend="python")
    return AnalysisToolbox(idx, CORPUS, {"v1": "brown fox"}, {"v1": {"d1": 1, "d2": 0}})


def test_schemas_name_the_three_paper_tools():
    assert [s["function"]["name"] for s in TOOL_SCHEMAS] == ["bm25_retrieve", "read_file", "grep_search"]


def test_bm25_retrieve(box):
    out = json.loads(box.dispatch("bm25_retrieve", {"query": "fox"}))
    assert out[0]["doc_id"] == "d1" and out[0]["rank"] == 1


def test_read_file_filter_id_doc_and_query(box):
    doc = json.loads(box.read_file("documents.jsonl", filter_id="d1"))
    assert doc == {"doc_id": "d1", "text": "the quick brown fox", "title": "Fox"}
    q = json.loads(box.read_file("validation_queries.jsonl", filter_id="v1"))
    assert q["relevant_doc_ids"] == ["d1"]


def test_read_file_truncates(box):
    assert box.read_file("documents.jsonl", filter_id="d2", max_chars=50).endswith("[truncated]")


@pytest.mark.parametrize("path", ["evaluation_queries.jsonl", "queries.jsonl"])
def test_heldout_files_forbidden(box, path):
    assert "forbidden" in box.read_file(path)
    assert "forbidden" in box.grep_search("x", path)


def test_unknown_file(box):
    assert "unknown file" in box.read_file("../../etc/passwd")


def test_grep_search(box):
    assert box.grep_search("brown", "documents.jsonl").startswith("line 1:")
    assert box.grep_search("zebra", "documents.jsonl") == "No matches found."
    assert "invalid regex" in box.grep_search("(", "documents.jsonl")


def test_dispatch_handles_bad_tool_and_args(box):
    assert "unknown tool" in box.dispatch("rm_rf", {})
    assert "bad arguments" in box.dispatch("bm25_retrieve", {"nope": 1})
    assert box.calls == ["rm_rf", "bm25_retrieve"]
