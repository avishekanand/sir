"""
Unit tests for CRUMBLoader (src/ragtune/data/loaders/CRUMBLoader.py).

Queries go through the module-level `fetch_hf_split` helper (imported into
CRUMBLoader's namespace), which is mocked directly. Corpus streaming uses a
local `from datasets import load_dataset` inside `_load_data`, so that one is
patched on the real `datasets` module. No network call is made.
"""

import pytest
from unittest.mock import patch

pytest.importorskip("datasets")

from ragtune.data.loaders.CRUMBLoader import CRUMBLoader, CRUMB_TASKS, DATASET_ID
from ragtune.data.datastructures.query import Query


# Order matters here: q1, q2, q3 have positive qrels; q_zero (all-zero labels)
# and q_empty (no qrels at all) must be dropped entirely; q_missing_doc has a
# positive qrel pointing at a passage that never appears in the corpus.
QUERY_ROWS = [
    {"query_id": "q1", "query_content": "query one",
     "passage_qrels": [{"id": "d1", "label": 1}, {"id": "d2", "label": 0}]},
    {"query_id": "q_zero", "query_content": "query with only zero-label qrels",
     "passage_qrels": [{"id": "d2", "label": 0}]},
    {"query_id": "q2", "query_content": "query two",
     "passage_qrels": [{"id": "d3", "label": 1}]},
    {"query_id": "q_empty", "query_content": "query with no qrels",
     "passage_qrels": []},
    {"query_id": "q3", "query_content": "query three",
     "passage_qrels": [{"id": "d4", "label": 1}]},
    {"query_id": "q_missing_doc", "query_content": "gold doc absent from corpus",
     "passage_qrels": [{"id": "d99", "label": 1}]},
]

CORPUS_ROWS = [
    {"document_id": "d1", "document_content": "doc one"},
    {"document_id": "d2", "document_content": "doc two"},
    {"document_id": "d3", "document_content": "doc three"},
    {"document_id": "d4", "document_content": "doc four"},
    {"document_id": "d5", "document_content": "doc five"},  # non-gold
    {"document_id": "d6", "document_content": "doc six"},  # non-gold
    # Note: d99 (referenced by q_missing_doc) is deliberately absent.
]


def make_fake_fetch_hf_split(query_rows=QUERY_ROWS):
    def _fake_fetch(dataset_id, config, split, cache_dir=None):
        assert dataset_id == DATASET_ID
        assert config == "evaluation_queries"
        return list(query_rows)

    return _fake_fetch


def make_fake_load_dataset(corpus_rows=CORPUS_ROWS):
    def _fake_load_dataset(dataset_id, config, split=None, streaming=False, **kwargs):
        assert dataset_id == DATASET_ID
        assert config == "passage_corpus"
        return list(corpus_rows)

    return _fake_load_dataset


def _patched(query_rows=QUERY_ROWS, corpus_rows=CORPUS_ROWS):
    return (
        patch(
            "ragtune.data.loaders.CRUMBLoader.fetch_hf_split",
            side_effect=make_fake_fetch_hf_split(query_rows),
        ),
        patch("datasets.load_dataset", side_effect=make_fake_load_dataset(corpus_rows)),
    )


def test_rejects_unknown_task():
    with pytest.raises(ValueError):
        CRUMBLoader(task="not-a-real-task")


def test_dataset_id_and_defaults():
    loader = CRUMBLoader(task="paper_retrieval")
    assert loader.dataset == DATASET_ID
    assert loader.task == "paper_retrieval"
    assert loader.split == "test"
    assert loader.max_queries is None
    assert loader.max_corpus_docs is None


def test_load_drops_queries_with_no_positive_qrels():
    p_fetch, p_ld = _patched()
    with p_fetch, p_ld:
        loader = CRUMBLoader(task="paper_retrieval")
        corpus, queries, qrels = loader.load()

    # q_zero (all-zero labels) and q_empty (no qrels) must be dropped entirely.
    assert set(queries) == {"q1", "q2", "q3", "q_missing_doc"}
    assert "q_zero" not in queries and "q_zero" not in qrels
    assert "q_empty" not in queries and "q_empty" not in qrels

    # d2 was a label=0 entry -> dropped from q1's qrels.
    assert qrels["q1"] == {"d1": 1}
    assert qrels["q2"] == {"d3": 1}
    assert qrels["q3"] == {"d4": 1}
    assert qrels["q_missing_doc"] == {"d99": 1}


def test_get_query_objects_excludes_dropped_queries():
    p_fetch, p_ld = _patched()
    with p_fetch, p_ld:
        loader = CRUMBLoader(task="paper_retrieval")
        query_objs = loader.get_query_objects()

    assert {q.id() for q in query_objs} == {"q1", "q2", "q3", "q_missing_doc"}
    assert all(isinstance(q, Query) for q in query_objs)


def test_raw_data_sample_for_missing_gold_doc_has_no_evidence():
    p_fetch, p_ld = _patched()
    with p_fetch, p_ld:
        loader = CRUMBLoader(task="paper_retrieval")
        loader.load()

    samples_by_qid = {}
    for s in loader.raw_data:
        samples_by_qid.setdefault(s.query.id(), []).append(s)

    # d99 never appears in the corpus stream -> evidences=None fallback.
    assert len(samples_by_qid["q_missing_doc"]) == 1
    assert samples_by_qid["q_missing_doc"][0].evidences is None

    assert len(samples_by_qid["q1"]) == 1
    assert samples_by_qid["q1"][0].evidences.id() == "d1"
    assert samples_by_qid["q1"][0].evidences.text() == "doc one"


def test_max_queries_counts_only_queries_with_positive_qrels():
    p_fetch, p_ld = _patched()
    with p_fetch, p_ld:
        loader = CRUMBLoader(task="paper_retrieval", max_queries=2)
        corpus, queries, qrels = loader.load()

    # q_zero sits between q1 and q2 in QUERY_ROWS but has no positive qrels,
    # so it must not count toward the cap.
    assert set(queries) == {"q1", "q2"}
    assert set(qrels) == {"q1", "q2"}


def test_max_corpus_docs_always_keeps_gold_documents():
    p_fetch, p_ld = _patched()
    with p_fetch, p_ld:
        loader = CRUMBLoader(task="paper_retrieval", max_corpus_docs=1)
        corpus, queries, qrels = loader.load()

    assert {"d1", "d3", "d4"}.issubset(corpus)
    non_gold_kept = {"d2", "d5", "d6"} & set(corpus)
    assert len(non_gold_kept) == 1


def test_gold_doc_found_after_cap_is_still_included():
    # Only queries whose gold docs are all findable in the corpus, so the
    # early-break optimization can actually trigger once everything's found.
    query_rows = [
        {"query_id": "q1", "query_content": "query one",
         "passage_qrels": [{"id": "d1", "label": 1}]},
        {"query_id": "q2", "query_content": "query two",
         "passage_qrels": [{"id": "d3", "label": 1}]},
        {"query_id": "q3", "query_content": "query three",
         "passage_qrels": [{"id": "d4", "label": 1}]},
    ]
    # Non-gold docs (d5, d6) appear before the gold docs in the stream, so
    # the cap is hit before all gold ids are known -> loader must keep
    # scanning (not break early) until every gold id has been seen.
    reordered_corpus = [
        {"document_id": "d5", "document_content": "doc five"},
        {"document_id": "d6", "document_content": "doc six"},
        {"document_id": "d1", "document_content": "doc one"},
        {"document_id": "d2", "document_content": "doc two"},
        {"document_id": "d3", "document_content": "doc three"},
        {"document_id": "d4", "document_content": "doc four"},
    ]
    p_fetch, p_ld = _patched(query_rows=query_rows, corpus_rows=reordered_corpus)
    with p_fetch, p_ld:
        loader = CRUMBLoader(task="paper_retrieval", max_corpus_docs=1)
        corpus, _, qrels = loader.load()

    assert {"d1", "d3", "d4"}.issubset(corpus)


def test_lazy_loading_only_fetches_once():
    p_fetch, p_ld = _patched()
    with p_fetch as mock_fetch, p_ld as mock_ld:
        loader = CRUMBLoader(task="paper_retrieval")
        assert mock_fetch.call_count == 0
        assert mock_ld.call_count == 0

        loader.get_corpus()
        fetch_calls_after_first = mock_fetch.call_count
        ld_calls_after_first = mock_ld.call_count
        assert fetch_calls_after_first > 0
        assert ld_calls_after_first > 0

        loader.get_queries()
        loader.get_qrels()
        assert mock_fetch.call_count == fetch_calls_after_first
        assert mock_ld.call_count == ld_calls_after_first


@pytest.mark.parametrize("task", CRUMB_TASKS)
def test_all_known_tasks_are_accepted(task):
    p_fetch, p_ld = _patched()
    with p_fetch, p_ld:
        loader = CRUMBLoader(task=task)
        corpus, queries, qrels = loader.load()

    assert loader.dataset == DATASET_ID
    assert len(queries) == 4
    assert len(corpus) > 0
