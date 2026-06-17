"""
Unit tests for ragtune.indexing.base.BaseIndexer.

Uses a minimal FakeIndexer to exercise the concrete default methods
(build(), _load_file_to_corpus()) without any real indexing deps.
"""

import json
import os
import pytest

from ragtune.indexing.base import BaseIndexer


# ---------------------------------------------------------------------------
# Fake concrete implementation
# ---------------------------------------------------------------------------

class FakeIndexer(BaseIndexer):
    """Records calls to build_from_corpus for assertion."""

    def __init__(self):
        self.last_corpus = None
        self.last_index_path = None

    def build_from_corpus(self, corpus, index_path, **params):
        self.last_corpus = corpus
        self.last_index_path = index_path
        return True

    def exists(self, index_path):
        return False

    def load(self, index_path):
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadFileToCorporus:

    def test_jsonl_roundtrip(self, tmp_path):
        path = tmp_path / "docs.jsonl"
        docs = [
            {"id": "d1", "text": "hello world", "title": "Doc One"},
            {"id": "d2", "text": "foo bar",     "title": "Doc Two"},
        ]
        path.write_text("\n".join(json.dumps(d) for d in docs))

        indexer = FakeIndexer()
        corpus = indexer._load_file_to_corpus(
            str(path), "jsonl", {"id_field": "id", "text_field": "text", "title_field": "title"}
        )

        assert len(corpus) == 2
        assert corpus["d1"] == {"text": "hello world", "title": "Doc One"}
        assert corpus["d2"] == {"text": "foo bar",     "title": "Doc Two"}

    def test_json_list_roundtrip(self, tmp_path):
        path = tmp_path / "docs.json"
        docs = [
            {"doc_id": "a", "text": "alpha", "title": ""},
            {"doc_id": "b", "text": "beta",  "title": ""},
        ]
        path.write_text(json.dumps(docs))

        indexer = FakeIndexer()
        corpus = indexer._load_file_to_corpus(str(path), "json", {})  # default field names

        assert set(corpus.keys()) == {"a", "b"}
        assert corpus["a"]["text"] == "alpha"

    def test_missing_fields_default_to_empty_string(self, tmp_path):
        path = tmp_path / "docs.jsonl"
        path.write_text(json.dumps({"doc_id": "x"}) + "\n")  # no text, no title

        indexer = FakeIndexer()
        corpus = indexer._load_file_to_corpus(str(path), "jsonl", {})
        assert corpus["x"]["text"] == ""
        assert corpus["x"]["title"] == ""

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "docs.csv"
        path.write_text("id,text\n1,hello\n")

        indexer = FakeIndexer()
        with pytest.raises(NotImplementedError):
            indexer._load_file_to_corpus(str(path), "csv", {})


class TestBuildDelegatesToBuildFromCorpus:

    def test_build_parses_file_and_calls_build_from_corpus(self, tmp_path):
        path = tmp_path / "docs.jsonl"
        path.write_text(json.dumps({"doc_id": "d1", "text": "hi"}) + "\n")

        indexer = FakeIndexer()
        result = indexer.build(
            str(path), "jsonl", {}, index_path=str(tmp_path / "idx")
        )

        assert result is True
        assert indexer.last_corpus == {"d1": {"text": "hi", "title": ""}}
        assert indexer.last_index_path == str(tmp_path / "idx")
