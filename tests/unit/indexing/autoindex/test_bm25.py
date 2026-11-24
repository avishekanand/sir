import math

import pytest

from ragtune.indexing.autoindex.bm25 import ChunkBM25Index, Tokenizer
from ragtune.indexing.autoindex.program import Chunk


def _chunks():
    return [
        Chunk("a_0", "a", "apple banana"),
        Chunk("a_1", "a", "apple apple apple cherry"),
        Chunk("b_0", "b", "banana cherry durian"),
        Chunk("c_0", "c", "eggplant"),
    ]


def test_tokenizer_defaults_match_reference():
    assert Tokenizer()("The Cat's 2 cats, X!") == ["the", "cat", "cats"]


def test_tokenizer_stopwords_and_stemmer():
    tok = Tokenizer(stopwords={"the"}, stemmer=lambda t: t.rstrip("s"))
    assert tok("the cats") == ["cat"]
    assert "stemmed" in tok.describe() and "stopwords removed" in tok.describe()


def test_lucene_score_matches_hand_computation():
    idx = ChunkBM25Index(_chunks(), backend="python")
    # 'eggplant' appears only in c_0 (len 1); N=4, avgdl=(2+4+3+1)/4=2.5
    idf = math.log(1 + (4 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 1 / (1 + 1.5 * (1 - 0.75 + 0.75 * 1 / 2.5))
    [(chunk_idx, score)] = idx.score_chunks("eggplant")
    assert chunk_idx == 3
    assert score == pytest.approx(expected)


def test_maxp_takes_best_chunk_per_doc():
    idx = ChunkBM25Index(_chunks(), backend="python")
    scores = dict(idx.score_chunks("apple"))
    hits = idx.search("apple")
    assert [h.doc_id for h in hits] == ["a"]
    assert hits[0].score == pytest.approx(max(scores[0], scores[1]))
    assert hits[0].rank == 1


def test_search_ranks_docs_and_respects_top_k():
    idx = ChunkBM25Index(_chunks(), backend="python")
    hits = idx.search("banana cherry", top_k=2)
    assert len(hits) == 2
    assert hits[0].doc_id == "b"
    assert [h.rank for h in hits] == [1, 2]


def test_candidate_k_limits_pool_before_maxp():
    idx = ChunkBM25Index(_chunks(), backend="python", candidate_k=1)
    assert len(idx.search("banana cherry")) == 1


def test_unknown_query_returns_empty():
    idx = ChunkBM25Index(_chunks(), backend="python")
    assert idx.search("zzz qqq") == []


def test_empty_chunks_rejected():
    with pytest.raises(ValueError):
        ChunkBM25Index([])


def test_bm25s_backend_matches_python():
    pytest.importorskip("bm25s")
    py = ChunkBM25Index(_chunks(), backend="python")
    bs = ChunkBM25Index(_chunks(), backend="bm25s")
    for q in ["apple", "banana cherry", "durian eggplant apple"]:
        assert [h.doc_id for h in py.search(q)] == [h.doc_id for h in bs.search(q)]
        for a, b in zip(py.search(q), bs.search(q)):
            assert a.score == pytest.approx(b.score, rel=1e-5)
