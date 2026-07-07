"""
Unit tests for OBLIQLoader (src/ragtune/data/loaders/OBLIQLoader.py).

OBLIQLoader imports `datasets.load_dataset` and `huggingface_hub.hf_hub_download`
locally inside `_load_data`, so — unlike CoIRLoader, which goes through the
`fetch_hf_split` wrapper — these tests patch the real `datasets.load_dataset`
and `huggingface_hub.hf_hub_download` functions directly. No network call is
made; `hf_hub_download` is redirected to local fixture files on disk.
"""

import json
import csv

import pytest
from unittest.mock import patch

pytest.importorskip("datasets")
pytest.importorskip("huggingface_hub")

from ragtune.data.loaders.OBLIQLoader import OBLIQLoader, OBLIQ_TASKS, DATASET_ID
from ragtune.data.datastructures.query import Query


QUERY_ROWS = [
    {"_id": "q1", "text": "query one"},
    {"_id": "q2", "text": "query two"},
    {"_id": "q3", "text": "query three"},
    {"_id": "q4", "text": "query four (no qrels)"},
]

# (query-id, corpus-id, score) rows, in the on-disk TSV order (header first).
QRELS_TSV_ROWS = [
    ("query-id", "corpus-id", "score"),
    ("q1", "d1", "1"),
    ("q1", "d2", "0"),  # non-positive -> filtered
    ("q2", "d3", "1"),
    ("q3", "d4", "1"),
    # q4 intentionally has no qrels row.
]

CORPUS_ROWS = [
    {"_id": "d1", "text": "doc one", "title": "t1"},
    {"_id": "d2", "text": "doc two", "title": "t2"},
    {"_id": "d3", "text": "doc three", "title": "t3"},
    {"_id": "d4", "text": "doc four", "title": "t4"},
    {"_id": "d5", "text": "doc five", "title": "t5"},  # non-gold
    {"_id": "d6", "text": "doc six", "title": "t6"},  # non-gold
]

EXCLUDED_IDS = {"q1": ["d9"], "q2": ["d10", "d11"]}


def _write_qrels_tsv(path, rows=QRELS_TSV_ROWS):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
    return path


def _write_excluded_ids_json(path, data=EXCLUDED_IDS):
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def make_fake_load_dataset(query_rows=QUERY_ROWS, corpus_rows=CORPUS_ROWS):
    def _fake_load_dataset(dataset_id, task, split=None, streaming=False, **kwargs):
        assert dataset_id == DATASET_ID
        if split == "queries":
            return list(query_rows)
        if split == "corpus":
            return list(corpus_rows)
        raise AssertionError(f"unexpected split={split!r}")

    return _fake_load_dataset


def make_fake_hf_hub_download(qrels_path, excluded_ids_path=None):
    def _fake_hf_hub_download(repo_id, repo_type, filename):
        assert repo_id == DATASET_ID
        assert repo_type == "dataset"
        if filename.endswith("per_query_excluded_ids.json"):
            assert excluded_ids_path is not None, f"unexpected excluded-ids fetch for {filename!r}"
            return str(excluded_ids_path)
        if filename.endswith("qrels.tsv"):
            return str(qrels_path)
        raise AssertionError(f"unexpected filename={filename!r}")

    return _fake_hf_hub_download


@pytest.fixture
def qrels_file(tmp_path):
    return _write_qrels_tsv(tmp_path / "qrels.tsv")


@pytest.fixture
def excluded_ids_file(tmp_path):
    return _write_excluded_ids_json(tmp_path / "excluded.json")


def _patched(qrels_path, excluded_ids_path=None, query_rows=QUERY_ROWS, corpus_rows=CORPUS_ROWS):
    return (
        patch("datasets.load_dataset", side_effect=make_fake_load_dataset(query_rows, corpus_rows)),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=make_fake_hf_hub_download(qrels_path, excluded_ids_path),
        ),
    )


def test_rejects_unknown_task():
    with pytest.raises(ValueError):
        OBLIQLoader(task="not-a-real-task")


def test_dataset_id_and_defaults():
    loader = OBLIQLoader(task="congress")
    assert loader.dataset == DATASET_ID
    assert loader.task == "congress"
    assert loader.split == "test"
    assert loader.max_queries is None
    assert loader.max_corpus_docs is None


def test_load_filters_zero_score_qrels_and_keeps_orphan_queries(qrels_file):
    p_ld, p_hf = _patched(qrels_file)
    with p_ld, p_hf:
        loader = OBLIQLoader(task="congress")
        corpus, queries, qrels = loader.load()

    # Unlike CoIRLoader, OBLIQLoader keeps queries that have no qrels.
    assert set(queries) == {"q1", "q2", "q3", "q4"}
    assert "q4" not in qrels

    # d2 was a score=0 row -> dropped from q1's qrels.
    assert qrels["q1"] == {"d1": 1}
    assert qrels["q2"] == {"d3": 1}
    assert qrels["q3"] == {"d4": 1}

    # No max_corpus_docs cap set -> full corpus retained.
    assert set(corpus) == {"d1", "d2", "d3", "d4", "d5", "d6"}


def test_raw_data_orphan_query_has_no_evidence(qrels_file):
    p_ld, p_hf = _patched(qrels_file)
    with p_ld, p_hf:
        loader = OBLIQLoader(task="congress")
        loader.load()

    by_qid = {s.query.id(): s for s in loader.raw_data if s.query.id() == "q4"}
    assert by_qid["q4"].evidences is None

    gold_samples = [s for s in loader.raw_data if s.query.id() == "q1"]
    assert len(gold_samples) == 1
    assert gold_samples[0].evidences.id() == "d1"
    assert gold_samples[0].evidences.text() == "doc one"


def test_get_query_objects_includes_orphan_queries(qrels_file):
    p_ld, p_hf = _patched(qrels_file)
    with p_ld, p_hf:
        loader = OBLIQLoader(task="congress")
        query_objs = loader.get_query_objects()

    assert {q.id() for q in query_objs} == {"q1", "q2", "q3", "q4"}
    assert all(isinstance(q, Query) for q in query_objs)


def test_max_queries_truncates_before_qrel_filtering(qrels_file):
    p_ld, p_hf = _patched(qrels_file)
    with p_ld, p_hf:
        loader = OBLIQLoader(task="congress", max_queries=2)
        corpus, queries, qrels = loader.load()

    assert set(queries) == {"q1", "q2"}
    assert set(qrels) == {"q1", "q2"}


def test_max_corpus_docs_always_keeps_gold_documents(qrels_file):
    p_ld, p_hf = _patched(qrels_file)
    with p_ld, p_hf:
        loader = OBLIQLoader(task="congress", max_corpus_docs=1)
        corpus, queries, qrels = loader.load()

    assert {"d1", "d3", "d4"}.issubset(corpus)
    non_gold_kept = {"d2", "d5", "d6"} & set(corpus)
    assert len(non_gold_kept) == 1


def test_gold_doc_found_after_cap_is_still_included(qrels_file):
    # Non-gold docs (d5, d6) appear before the gold docs in the stream, so
    # the cap is hit before all gold ids are known -> loader must keep
    # scanning (not break early) until every gold id has been seen.
    reordered_corpus = [
        {"_id": "d5", "text": "doc five", "title": "t5"},
        {"_id": "d6", "text": "doc six", "title": "t6"},
        {"_id": "d1", "text": "doc one", "title": "t1"},
        {"_id": "d2", "text": "doc two", "title": "t2"},
        {"_id": "d3", "text": "doc three", "title": "t3"},
        {"_id": "d4", "text": "doc four", "title": "t4"},
    ]
    p_ld, p_hf = _patched(qrels_file, corpus_rows=reordered_corpus)
    with p_ld, p_hf:
        loader = OBLIQLoader(task="congress", max_corpus_docs=1)
        corpus, _, qrels = loader.load()

    assert {"d1", "d3", "d4"}.issubset(corpus)


@pytest.mark.parametrize(
    "task,expects_excluded_ids",
    [
        ("congress", False),
        ("math", True),
        ("writing", True),
        ("twitter", False),
        ("wildchat", False),
    ],
)
def test_excluded_ids_only_populated_for_math_and_writing(
    task, expects_excluded_ids, qrels_file, excluded_ids_file
):
    p_ld, p_hf = _patched(
        qrels_file,
        excluded_ids_path=excluded_ids_file if expects_excluded_ids else None,
    )
    with p_ld, p_hf:
        loader = OBLIQLoader(task=task)
        loader.load()
        excluded = loader.get_excluded_ids()

    if expects_excluded_ids:
        assert excluded == EXCLUDED_IDS
    else:
        assert excluded == {}


def test_lazy_loading_only_fetches_once(qrels_file):
    p_ld, p_hf = _patched(qrels_file)
    with p_ld as mock_ld, p_hf as mock_hf:
        loader = OBLIQLoader(task="congress")
        assert mock_ld.call_count == 0
        assert mock_hf.call_count == 0

        loader.get_corpus()
        ld_calls_after_first = mock_ld.call_count
        hf_calls_after_first = mock_hf.call_count
        assert ld_calls_after_first > 0
        assert hf_calls_after_first > 0

        loader.get_queries()
        loader.get_qrels()
        loader.get_excluded_ids()
        assert mock_ld.call_count == ld_calls_after_first
        assert mock_hf.call_count == hf_calls_after_first


@pytest.mark.parametrize("task", OBLIQ_TASKS)
def test_all_known_tasks_are_accepted(task, qrels_file, excluded_ids_file):
    p_ld, p_hf = _patched(qrels_file, excluded_ids_path=excluded_ids_file)
    with p_ld, p_hf:
        loader = OBLIQLoader(task=task)
        corpus, queries, qrels = loader.load()

    assert loader.dataset == DATASET_ID
    assert len(queries) == 4
    assert len(corpus) == 6
