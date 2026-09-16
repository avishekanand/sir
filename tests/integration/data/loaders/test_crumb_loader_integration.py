"""
Integration tests for CRUMBLoader against the real CRUMB benchmark on
HuggingFace Hub (jfkback/crumb). Unlike
tests/unit/data/loaders/test_crumb_loader.py, these hit the network and
download real data — no mocking of fetch_hf_split / load_dataset.

Every task in CRUMB_TASKS is exercised. Queries/corpus are capped via
max_queries/max_corpus_docs to keep each loaded object small; the corpus
split is streamed by the loader itself so the cap also bounds how much is
actually pulled over the network.
"""

import pytest

pytest.importorskip("datasets")

from ragtune.data.loaders.CRUMBLoader import CRUMBLoader, CRUMB_TASKS
from ragtune.data.datastructures.query import Query
from ragtune.data.datastructures.sample import Sample


@pytest.fixture(scope="module", params=CRUMB_TASKS)
def crumb_loader(request):
    loader = CRUMBLoader(task=request.param, max_queries=10, max_corpus_docs=50)
    loader.load()
    return loader


def test_real_corpus_queries_qrels_are_populated(crumb_loader):
    corpus, queries, qrels = crumb_loader.load()

    assert len(queries) > 0
    assert len(corpus) > 0

    # Every loaded query must have come with at least one positive qrel
    # (CRUMBLoader drops queries with none).
    assert set(qrels) == set(queries)
    for qid, rels in qrels.items():
        assert rels
        assert all(score > 0 for score in rels.values())

    for qid, text in queries.items():
        assert isinstance(qid, str) and qid
        assert isinstance(text, str) and text.strip()

    for doc_id, doc in corpus.items():
        assert isinstance(doc_id, str) and doc_id
        assert "text" in doc and doc["text"]


def test_real_qrel_gold_docs_are_present_in_corpus(crumb_loader):
    corpus, _, qrels = crumb_loader.load()

    gold_ids = {doc_id for rels in qrels.values() for doc_id in rels}
    # max_corpus_docs caps non-gold docs only; every gold doc must survive.
    assert gold_ids.issubset(corpus.keys())


def test_real_query_objects_match_queries_dict(crumb_loader):
    corpus, queries, _ = crumb_loader.load()
    query_objs = crumb_loader.get_query_objects()

    assert {q.id() for q in query_objs} == set(queries)
    assert all(isinstance(q, Query) for q in query_objs)
    for q in query_objs:
        assert q.text() == queries[q.id()]


def test_real_raw_data_samples_carry_gold_evidence(crumb_loader):
    corpus, _, qrels = crumb_loader.load()

    assert crumb_loader.raw_data, "raw_data should not be empty"
    assert all(isinstance(s, Sample) for s in crumb_loader.raw_data)

    samples_by_qid = {}
    for sample in crumb_loader.raw_data:
        samples_by_qid.setdefault(sample.query.id(), []).append(sample)

    assert set(samples_by_qid) == set(qrels)

    for qid, gold_ids in qrels.items():
        samples = samples_by_qid[qid]
        gold_ids_in_corpus = set(gold_ids) & set(corpus)
        if gold_ids_in_corpus:
            # One Sample per gold doc that actually resolved in the corpus.
            assert {s.evidences.id() for s in samples} == gold_ids_in_corpus
            for s in samples:
                assert s.evidences.text() == corpus[s.evidences.id()]["text"]
        else:
            # None of this query's gold docs turned up in the corpus stream
            # (can happen with real, imperfect benchmark data) -> placeholder.
            assert len(samples) == 1
            assert samples[0].evidences is None
