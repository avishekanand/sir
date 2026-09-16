"""
Integration tests for OBLIQLoader against the real OBLIQ-Bench dataset on
HuggingFace Hub (dianetc/OBLIQ-Bench). Unlike
tests/unit/data/loaders/test_obliq_loader.py, these hit the network and
download real data — no mocking of load_dataset / hf_hub_download.

Every task in OBLIQ_TASKS (congress, math, writing, twitter, wildchat) is
exercised. Queries/corpus are capped via max_queries/max_corpus_docs to keep
each loaded object small.
"""

import pytest

pytest.importorskip("datasets")
pytest.importorskip("huggingface_hub")

from ragtune.data.loaders.OBLIQLoader import OBLIQLoader, OBLIQ_TASKS
from ragtune.data.datastructures.query import Query
from ragtune.data.datastructures.sample import Sample

MATH_WRITING_TASKS = {"math", "writing"}


@pytest.fixture(scope="module", params=OBLIQ_TASKS)
def obliq_loader(request):
    loader = OBLIQLoader(task=request.param, max_queries=10, max_corpus_docs=50)
    loader.load()
    return loader


def test_real_corpus_queries_qrels_are_populated(obliq_loader):
    corpus, queries, qrels = obliq_loader.load()

    assert len(queries) == 10
    assert len(corpus) > 0

    # Every qrel entry must be a real, positive-relevance judgment.
    assert qrels, "expected at least one query with qrels in the first 10 rows"
    for qid, rels in qrels.items():
        assert rels
        assert all(score > 0 for score in rels.values())

    for qid, text in queries.items():
        assert isinstance(qid, str) and qid
        assert isinstance(text, str) and text.strip()

    for doc_id, doc in corpus.items():
        assert isinstance(doc_id, str) and doc_id
        assert "text" in doc and "title" in doc


def test_real_qrel_gold_docs_are_present_in_corpus(obliq_loader):
    corpus, _, qrels = obliq_loader.load()

    gold_ids = {doc_id for rels in qrels.values() for doc_id in rels}
    # max_corpus_docs caps non-gold docs only; every gold doc must survive.
    assert gold_ids.issubset(corpus.keys())


def test_real_query_objects_match_queries_dict(obliq_loader):
    corpus, queries, _ = obliq_loader.load()
    query_objs = obliq_loader.get_query_objects()

    assert {q.id() for q in query_objs} == set(queries)
    assert all(isinstance(q, Query) for q in query_objs)
    for q in query_objs:
        assert q.text() == queries[q.id()]


def test_real_raw_data_samples_carry_gold_evidence_or_none(obliq_loader):
    corpus, _, qrels = obliq_loader.load()

    assert obliq_loader.raw_data, "raw_data should not be empty"
    assert all(isinstance(s, Sample) for s in obliq_loader.raw_data)

    samples_by_qid = {}
    for sample in obliq_loader.raw_data:
        samples_by_qid.setdefault(sample.query.id(), []).append(sample)

    # Every loaded query produces exactly one Sample group.
    assert set(samples_by_qid) == set(obliq_loader.get_queries())

    for qid, samples in samples_by_qid.items():
        gold_ids = set(qrels.get(qid, {}))
        if gold_ids:
            assert {s.evidences.id() for s in samples} == gold_ids
            for s in samples:
                assert s.evidences.text() == corpus[s.evidences.id()]["text"]
        else:
            # Orphan query (no qrels): a single placeholder Sample with no evidence.
            assert len(samples) == 1
            assert samples[0].evidences is None


def test_real_excluded_ids_only_for_math_and_writing(obliq_loader):
    excluded = obliq_loader.get_excluded_ids()
    if obliq_loader.task in MATH_WRITING_TASKS:
        assert isinstance(excluded, dict) and excluded
    else:
        assert excluded == {}
