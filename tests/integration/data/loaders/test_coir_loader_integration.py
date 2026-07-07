"""
Integration tests for CoIRLoader against the real CoIR-Retrieval datasets on
HuggingFace Hub. Unlike tests/unit/data/loaders/test_coir_loader.py, these
hit the network and download real data — no mocking of fetch_hf_split.

Every dataset in COIR_DATASETS (stackoverflow-qa, codefeedback-st, apps,
cosqa, synthetic-text2sql) is exercised. Queries/corpus are capped via
max_queries/max_corpus_docs to keep each loaded object small — the
underlying HF split is still downloaded in full (and cached by `datasets`
after the first run), but that has proven fast (a few seconds) for every
CoIR-Retrieval corpus.
"""

import pytest

pytest.importorskip("datasets")

from ragtune.data.loaders.CoIRLoader import CoIRLoader, COIR_DATASETS
from ragtune.data.datastructures.query import Query
from ragtune.data.datastructures.sample import Sample


@pytest.fixture(scope="module", params=COIR_DATASETS)
def coir_loader(request):
    loader = CoIRLoader(dataset=request.param, max_queries=25, max_corpus_docs=50)
    loader.load()
    return loader


def test_real_corpus_queries_qrels_are_populated(coir_loader):
    corpus, queries, qrels = coir_loader.load()

    assert len(queries) == 25
    assert len(qrels) == 25
    assert len(corpus) > 0

    # Every query must have at least one positive-relevance qrel.
    for qid, rels in qrels.items():
        assert rels, f"query {qid!r} has no qrels"
        assert all(score > 0 for score in rels.values())

    # Query/doc ids and text must be real, non-empty strings.
    for qid, text in queries.items():
        assert isinstance(qid, str) and qid
        assert isinstance(text, str) and text.strip()

    for doc_id, doc in corpus.items():
        assert isinstance(doc_id, str) and doc_id
        assert "text" in doc and "title" in doc


def test_real_qrel_gold_docs_are_present_in_corpus(coir_loader):
    corpus, _, qrels = coir_loader.load()

    gold_ids = {doc_id for rels in qrels.values() for doc_id in rels}
    # max_corpus_docs caps non-gold docs only; every gold doc must survive.
    assert gold_ids.issubset(corpus.keys())


def test_real_query_objects_match_queries_dict(coir_loader):
    corpus, queries, _ = coir_loader.load()
    query_objs = coir_loader.get_query_objects()

    assert {q.id() for q in query_objs} == set(queries)
    assert all(isinstance(q, Query) for q in query_objs)
    for q in query_objs:
        assert q.text() == queries[q.id()]


def test_real_raw_data_samples_carry_gold_evidence(coir_loader):
    corpus, _, qrels = coir_loader.load()

    assert coir_loader.raw_data, "raw_data should not be empty"
    assert all(isinstance(s, Sample) for s in coir_loader.raw_data)

    samples_by_qid = {}
    for sample in coir_loader.raw_data:
        samples_by_qid.setdefault(sample.query.id(), []).append(sample)

    for qid, gold_ids in qrels.items():
        samples = samples_by_qid[qid]
        # One Sample per gold doc for that query.
        assert {s.evidences.id() for s in samples} == set(gold_ids)
        for s in samples:
            assert s.evidences.text() == corpus[s.evidences.id()]["text"]
