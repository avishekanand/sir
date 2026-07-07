"""
Unit tests for CoIRLoader (src/ragtune/data/loaders/CoIRLoader.py).

`fetch_hf_split` is mocked throughout so these tests never hit the network /
HuggingFace Hub, per the project's unit-test conventions.
"""

import pytest
from unittest.mock import patch

from ragtune.data.loaders.CoIRLoader import CoIRLoader, COIR_DATASETS
from ragtune.data.datastructures.query import Query


QRELS_ROWS = [
    {"query-id": "q1", "corpus-id": "d1", "score": 1},
    {"query-id": "q1", "corpus-id": "d2", "score": 0},  # non-positive -> filtered
    {"query-id": "q2", "corpus-id": "d3", "score": 1},
    {"query-id": "q3", "corpus-id": "d4", "score": 1},
]

QUERIES_ROWS = [
    {"_id": "q1", "text": "query one"},
    {"_id": "q2", "text": "query two"},
    {"_id": "q3", "text": "query three"},
    {"_id": "q4", "text": "query four"},  # has no qrels -> dropped
]

CORPUS_ROWS = [
    {"_id": "d1", "text": "doc one", "title": "t1"},
    {"_id": "d2", "text": "doc two", "title": "t2"},
    {"_id": "d3", "text": "doc three", "title": "t3"},
    {"_id": "d4", "text": "doc four", "title": "t4"},
    {"_id": "d5", "text": "doc five", "title": "t5"},  # non-gold
    {"_id": "d6", "text": "doc six", "title": "t6"},  # non-gold
]


def make_fake_fetch(fail_splits=()):
    """
    Build a fetch_hf_split stand-in keyed on (config, split), raising
    RuntimeError for any (config, split) pair listed in `fail_splits`.
    """

    def _fake_fetch(dataset_id, config=None, split=None, cache_dir=None):
        if (config, split) in fail_splits:
            raise RuntimeError(f"simulated failure for config={config!r} split={split!r}")
        if config is None:
            return list(QRELS_ROWS)
        if config == "queries":
            return list(QUERIES_ROWS)
        if config == "corpus":
            return list(CORPUS_ROWS)
        raise AssertionError(f"unexpected config={config!r}")

    return _fake_fetch


def test_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        CoIRLoader(dataset="not-a-real-coir-dataset")


def test_dataset_id_and_defaults():
    loader = CoIRLoader(dataset="cosqa")
    assert loader.dataset == "CoIR-Retrieval/cosqa"
    assert loader.split == "test"
    assert loader.max_queries is None
    assert loader.max_corpus_docs is None


def test_load_filters_zero_score_qrels_and_orphan_queries():
    with patch("ragtune.data.loaders.CoIRLoader.fetch_hf_split", side_effect=make_fake_fetch()):
        loader = CoIRLoader(dataset="cosqa")
        corpus, queries, qrels = loader.load()

    # q4 has no qrels, so it must not appear in queries.
    assert set(queries) == {"q1", "q2", "q3"}
    assert queries["q1"] == "query one"

    # d2 was a score=0 qrel row, so it must be dropped from q1's qrels.
    assert qrels["q1"] == {"d1": 1}
    assert qrels["q2"] == {"d3": 1}
    assert qrels["q3"] == {"d4": 1}

    # All corpus docs are pulled in (no max_corpus_docs cap set).
    assert set(corpus) == {"d1", "d2", "d3", "d4", "d5", "d6"}
    assert corpus["d1"] == {"text": "doc one", "title": "t1"}


def test_get_query_objects_returns_query_instances():
    with patch("ragtune.data.loaders.CoIRLoader.fetch_hf_split", side_effect=make_fake_fetch()):
        loader = CoIRLoader(dataset="cosqa")
        query_objs = loader.get_query_objects()

    assert {q.id() for q in query_objs} == {"q1", "q2", "q3"}
    assert all(isinstance(q, Query) for q in query_objs)


def test_max_queries_caps_and_reconciles_qrels():
    with patch("ragtune.data.loaders.CoIRLoader.fetch_hf_split", side_effect=make_fake_fetch()):
        loader = CoIRLoader(dataset="cosqa", max_queries=2)
        corpus, queries, qrels = loader.load()

    assert set(queries) == {"q1", "q2"}
    assert set(qrels) == {"q1", "q2"}


def test_max_corpus_docs_always_keeps_gold_documents():
    with patch("ragtune.data.loaders.CoIRLoader.fetch_hf_split", side_effect=make_fake_fetch()):
        loader = CoIRLoader(dataset="cosqa", max_corpus_docs=1)
        corpus, queries, qrels = loader.load()

    # Gold docs (d1, d3, d4) must always be present, regardless of the cap.
    assert {"d1", "d3", "d4"}.issubset(corpus)
    # Only one non-gold doc (out of d2, d5, d6) should have been let through.
    non_gold_kept = {"d2", "d5", "d6"} & set(corpus)
    assert len(non_gold_kept) == 1


def test_qrels_split_falls_back_to_train():
    fake_fetch = make_fake_fetch(fail_splits={(None, "test")})
    with patch("ragtune.data.loaders.CoIRLoader.fetch_hf_split", side_effect=fake_fetch):
        loader = CoIRLoader(dataset="cosqa", split="test")
        _, queries, qrels = loader.load()

    assert set(queries) == {"q1", "q2", "q3"}


def test_qrels_missing_entirely_raises_runtime_error():
    fake_fetch = make_fake_fetch(fail_splits={(None, "test"), (None, "train")})
    with patch("ragtune.data.loaders.CoIRLoader.fetch_hf_split", side_effect=fake_fetch):
        loader = CoIRLoader(dataset="cosqa")
        with pytest.raises(RuntimeError):
            loader.load()


def test_queries_split_falls_back_to_test_split():
    fake_fetch = make_fake_fetch(fail_splits={("queries", "queries")})
    with patch("ragtune.data.loaders.CoIRLoader.fetch_hf_split", side_effect=fake_fetch):
        loader = CoIRLoader(dataset="cosqa")
        _, queries, _ = loader.load()

    assert set(queries) == {"q1", "q2", "q3"}


def test_lazy_loading_only_fetches_once():
    with patch(
        "ragtune.data.loaders.CoIRLoader.fetch_hf_split", side_effect=make_fake_fetch()
    ) as mock_fetch:
        loader = CoIRLoader(dataset="cosqa")
        assert mock_fetch.call_count == 0

        loader.get_corpus()
        call_count_after_first_access = mock_fetch.call_count
        assert call_count_after_first_access > 0

        loader.get_queries()
        loader.get_qrels()
        assert mock_fetch.call_count == call_count_after_first_access


@pytest.mark.parametrize("dataset_name", COIR_DATASETS)
def test_all_known_coir_datasets_are_accepted(dataset_name):
    with patch("ragtune.data.loaders.CoIRLoader.fetch_hf_split", side_effect=make_fake_fetch()):
        loader = CoIRLoader(dataset=dataset_name)
        assert loader.dataset == f"CoIR-Retrieval/{dataset_name}"
        assert len(loader) == 3
