import pytest
from ragtune.core.pool import CandidatePool, PoolItem
from ragtune.core.types import ScoredDocument, ItemState

def test_pool_add_items_dedup_and_provenance():
    pool = CandidatePool()
    
    docs_orig = [
        ScoredDocument(id="doc1", content="c1", score=0.9, metadata={}),
        ScoredDocument(id="doc2", content="c2", score=0.8, metadata={}),
    ]
    pool.add_items(docs_orig, source="original")
    
    assert len(pool) == 2
    assert pool._items["doc1"].sources == {"original": 0.9}
    assert pool._items["doc1"].initial_rank == 0
    assert pool._items["doc1"].appearances_count == 1
    
    docs_rewrite = [
        ScoredDocument(id="doc2", content="c2", score=0.85, metadata={}),
        ScoredDocument(id="doc3", content="c3", score=0.7, metadata={}),
    ]
    pool.add_items(docs_rewrite, source="rewrite_0")
    
    assert len(pool) == 3
    # doc2 appeared in both
    assert pool._items["doc2"].sources == {"original": 0.8, "rewrite_0": 0.85}
    assert pool._items["doc2"].initial_rank == 0 # min(1 from original, 0 from rewrite)
    assert pool._items["doc2"].appearances_count == 2
    assert pool._items["doc2"].final_score() == 0.85 # max of sources

def test_pool_enforce_cap_deterministic():
    pool = CandidatePool()
    docs = [
        ScoredDocument(id="doc1", content="c1", score=0.1, metadata={}),
        ScoredDocument(id="doc2", content="c2", score=0.9, metadata={}),
        ScoredDocument(id="doc3", content="c3", score=0.5, metadata={}),
        ScoredDocument(id="doc4", content="c4", score=0.5, metadata={}), # Tie with doc3
    ]
    pool.add_items(docs, source="original")
    
    # Cap to 2. Should keep doc2 (0.9) and doc3 (0.5). 
    # Between doc3 and doc4 (tie), sort by doc_id ASC. 
    # doc3 < doc4, so doc3 is kept.
    pool.enforce_cap(2)
    
    assert len(pool) == 2
    doc_ids = sorted(pool._items.keys())
    assert doc_ids == ["doc2", "doc3"]

def test_pool_initial_rank_tracking():
    pool = CandidatePool()
    # doc1 is rank 1 in original
    pool.add_items([ScoredDocument(id="doc1", content="c", score=0.5, metadata={}) for _ in range(2)], source="orig")
    # Actually add different docs to see ranks
    docs1 = [ScoredDocument(id=f"doc{i}", content="c", score=0.5, metadata={}) for i in range(5)]
    pool.add_items(docs1, source="orig")
    assert pool._items["doc3"].initial_rank == 3
    
    # In rewrite, doc3 is rank 0
    docs2 = [ScoredDocument(id="doc3", content="c", score=0.6, metadata={})]
    pool.add_items(docs2, source="rewrite")
    assert pool._items["doc3"].initial_rank == 0
    assert pool._items["doc3"].appearances_count == 2
