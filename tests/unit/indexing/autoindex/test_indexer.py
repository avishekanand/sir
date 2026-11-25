import pytest

from ragtune.indexing.autoindex.indexer import AutoIndexBM25Indexer
from ragtune.indexing.factory import IndexFactory

from .conftest import TITLE_PROGRAM

CORPUS = {"d1": {"text": "apple pie", "title": "Dessert"}, "d2": {"text": "banana bread", "title": "Baking"}}


def test_registered_with_factory():
    assert isinstance(IndexFactory.create("autoindex"), AutoIndexBM25Indexer)


def test_build_persist_load_search_roundtrip(tmp_path):
    path = str(tmp_path / "idx")
    ix = AutoIndexBM25Indexer(program_code=TITLE_PROGRAM, backend="python")
    assert not ix.exists(path)
    assert ix.build_from_corpus(CORPUS, index_path=path)
    assert ix.exists(path)
    assert (tmp_path / "idx" / "program.py").read_text() == TITLE_PROGRAM
    assert ix.search("dessert", top_k=5, index_path=path)[0].doc_id == "d1"

    fresh = AutoIndexBM25Indexer(backend="python")  # loads from disk, not from cache
    hits = fresh.search("baking", top_k=5, index_path=path)
    assert [h.doc_id for h in hits] == ["d2"]


def test_identity_default_ignores_title(tmp_path):
    ix = AutoIndexBM25Indexer(backend="python")
    ix.build_from_corpus(CORPUS, index_path=str(tmp_path))
    assert ix.search("dessert", top_k=5, index_path=str(tmp_path)) == []


def test_program_path_and_conflicting_args(tmp_path):
    f = tmp_path / "p.py"
    f.write_text(TITLE_PROGRAM)
    assert AutoIndexBM25Indexer(program_path=str(f)).program_code == TITLE_PROGRAM
    with pytest.raises(ValueError):
        AutoIndexBM25Indexer(program_code="x", program_path=str(f))


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AutoIndexBM25Indexer().load(str(tmp_path / "nope"))
