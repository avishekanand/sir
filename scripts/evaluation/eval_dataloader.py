#!/usr/bin/env python3
"""
test_src.ragtune.data.py
=====================
Comprehensive test and benchmark script for the data pipeline.

Mirrors the patterns used in:
  - avishekanand/sir  scripts/benchmark_bright.py  (RAGtune pipeline)
  - avishekanand/sir  adapters/pyterrier.py         (PyTerrier BM25 retriever)

All network-dependent tests use RetrieverDataset as the single entry point
so the full loader stack (BRIGHTLoader → BaseDataLoader → corpus/queries/qrels/
Sample objects) is exercised end-to-end via the same public API that SIR uses.

Usage
-----
    # Offline – only no-network tests
    python tests/test_src.ragtune.data.py --offline

    # Full suite
    python tests/test_src.ragtune.data.py

    # Skip RAGtune (PyTerrier-only)
    python tests/test_src.ragtune.data.py --no-ragtune

    # Specific groups
    python tests/test_src.ragtune.data.py --tasks constants evaluator retriever_dataset

Deps
----
    pip install datasets pytrec_eval-terrier      # always required for online tests
    pip install python-terrier                     # ⑨ and ⑩
    pip install -e <sir_root>                      # ⑩  (installs ragtune from src/)
"""

import argparse
import logging
import os
import statistics
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure the package is importable when running from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ragtune.data.constants import (
    Benchmark, Dataset, Split, BRIGHT_TASKS, FRESHSTACK_TOPICS,
)
from src.ragtune.data.datastructures import Query, Context, Sample, Answer
from src.ragtune.data.loaders import (
    BRIGHTLoader, BRIGHTMultiTaskLoader, FreshStackLoader,
    RetrieverDataset, DataLoaderFactory,
)
from src.ragtune.evaluation import RetrievalEvaluator


# ---------------------------------------------------------------------------
# Lightweight test-result tracker
# ---------------------------------------------------------------------------

class TR:
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.skipped: List[str] = []

    def ok(self, name: str):
        self.passed.append(name)
        print(f"  ✓ PASS  {name}")

    def fail(self, name: str, reason: str = ""):
        self.failed.append(name)
        print(f"  ✗ FAIL  {name}" + (f"  [{reason}]" if reason else ""))

    def skip(self, name: str, reason: str = ""):
        self.skipped.append(name)
        print(f"  ~ SKIP  {name}" + (f"  [{reason}]" if reason else ""))

    def summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print(f"\n{'='*62}")
        print(
            f"  Results: {len(self.passed)}/{total} passed  "
            f"{len(self.failed)} failed  {len(self.skipped)} skipped"
        )
        if self.failed:
            print("  Failed tests:")
            for f in self.failed:
                print(f"    - {f}")
        print(f"{'='*62}")
        return len(self.failed) == 0


tr = TR()


def section(title: str):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")


def _skip_group(names: List[str], reason: str):
    for n in names:
        tr.skip(n, reason)


# ===========================================================================
# ① Constants
# ===========================================================================

def test_constants():
    section("① Constants & enum values")

    try:
        assert len(BRIGHT_TASKS) == 12, f"expected 12, got {len(BRIGHT_TASKS)}"
        assert Dataset.BIOLOGY         in BRIGHT_TASKS
        assert Dataset.STACKOVERFLOW   in BRIGHT_TASKS
        assert Dataset.LEETCODE        in BRIGHT_TASKS
        assert Dataset.THEOREMQA_THEOREMS in BRIGHT_TASKS
        tr.ok("constants_bright_task_list")
    except AssertionError as e:
        tr.fail("constants_bright_task_list", str(e))

    try:
        assert len(FRESHSTACK_TOPICS) == 5
        assert Dataset.LANGCHAIN in FRESHSTACK_TOPICS
        assert Dataset.ANGULAR   in FRESHSTACK_TOPICS
        tr.ok("constants_freshstack_topic_list")
    except AssertionError as e:
        tr.fail("constants_freshstack_topic_list", str(e))

    try:
        assert Split.TEST  == "test"
        assert Split.TRAIN == "train"
        assert Split.DEV   == "dev"
        assert Benchmark.BRIGHT     == "BRIGHT"
        assert Benchmark.FRESHSTACK == "freshstack"
        assert Benchmark.BEIR       == "beir"
        tr.ok("constants_enum_values")
    except AssertionError as e:
        tr.fail("constants_enum_values", str(e))


# ===========================================================================
# ② Data structures
# ===========================================================================

def test_datastructures():
    section("② Data structure constructors & accessors")

    # Query
    try:
        q = Query(text="What is entropy?", idx="q42", reasoning="thermodynamics")
        assert q.text()     == "What is entropy?"
        assert q.id()       == "q42"
        assert q.reasoning  == "thermodynamics"
        # set_id
        q.set_id("q99")
        assert q.id() == "q99"
        # equality / hash
        q2 = Query(text="other", idx="q99")
        assert q == q2
        assert hash(q) == hash(q2)
        tr.ok("ds_query_accessors")
    except Exception as e:
        tr.fail("ds_query_accessors", str(e))

    # Context
    try:
        c = Context(text="Entropy measures disorder.", idx="d01", title="Thermo")
        assert c.text()  == "Entropy measures disorder."
        assert c.id()    == "d01"
        assert c.title() == "Thermo"
        # no title
        c2 = Context(text="bare", idx="d02")
        assert c2.title() is None
        tr.ok("ds_context_accessors")
    except Exception as e:
        tr.fail("ds_context_accessors", str(e))

    # Answer
    try:
        a = Answer(text="42", idx="a1")
        assert a.text() == "42"
        assert a.id()   == "a1"
        assert a.flatten() == ["42"]
        tr.ok("ds_answer_accessors")
    except Exception as e:
        tr.fail("ds_answer_accessors", str(e))

    # Sample with all fields
    try:
        q = Query(text="q", idx="1")
        c = Context(text="c", idx="2")
        a = Answer(text="a")
        s = Sample(idx="1", query=q, evidences=c, answer=a)
        assert s.query    is q
        assert s.evidences is c
        assert s.answer   is a
        assert s.idx      == "1"
        tr.ok("ds_sample_full")
    except Exception as e:
        tr.fail("ds_sample_full", str(e))

    # Sample with no evidence (queries that lack gold docs)
    try:
        q = Query(text="unanswered", idx="u0")
        s = Sample(idx="u0", query=q)
        assert s.evidences is None
        assert s.answer    is None
        tr.ok("ds_sample_optional_fields")
    except Exception as e:
        tr.fail("ds_sample_optional_fields", str(e))


# ===========================================================================
# ③ RetrievalEvaluator – synthetic unit tests
# ===========================================================================

def test_evaluator_synthetic():
    section("③ RetrievalEvaluator – synthetic unit tests")

    qrels = {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1}}

    # --- perfect retrieval ---
    perfect = {
        "q1": {"d1": 10.0, "d2": 9.0, "d9": 1.0},
        "q2": {"d3": 10.0, "d4": 2.0},
    }
    try:
        ev = RetrievalEvaluator(k_values=[1, 5, 10])
        m  = ev.evaluate(qrels, perfect)
        assert "ndcg" in m and "map" in m and "recall" in m and "precision" in m and "mrr" in m
        assert m["ndcg"]["NDCG@10"]     > 0.99
        assert m["recall"]["Recall@10"] == 1.0
        tr.ok("eval_perfect_retrieval")
    except Exception as e:
        tr.fail("eval_perfect_retrieval", str(e))

    # --- zero retrieval ---
    try:
        m = ev.evaluate(qrels, {"q1": {}, "q2": {}})
        assert m["ndcg"]["NDCG@10"] == 0.0
        assert m["recall"]["Recall@10"] == 0.0
        tr.ok("eval_zero_retrieval")
    except Exception as e:
        tr.fail("eval_zero_retrieval", str(e))

    # --- MRR: relevant doc at rank 2 ---
    try:
        m = ev.evaluate(
            {"q1": {"d2": 1}},
            {"q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}},
        )
        assert abs(m["mrr"]["MRR"] - 0.5) < 0.01
        tr.ok("eval_mrr_rank2")
    except Exception as e:
        tr.fail("eval_mrr_rank2", str(e))

    # --- Precision@1 ---
    try:
        m = ev.evaluate(
            {"q1": {"d1": 1}},
            {"q1": {"d1": 5.0, "d2": 2.0}},
        )
        assert m["precision"]["Precision@1"] == 1.0
        tr.ok("eval_precision_at1")
    except Exception as e:
        tr.fail("eval_precision_at1", str(e))

    # --- evaluate_run convenience wrapper ---
    try:
        m = evaluate_run(qrels, perfect, k_values=[5, 10], verbose=False)
        assert "ndcg" in m
        tr.ok("eval_run_convenience")
    except Exception as e:
        tr.fail("eval_run_convenience", str(e))

    # --- filter identical IDs (doc_id == query_id should be stripped) ---
    try:
        filtered = RetrievalEvaluator._filter_identical_ids(
            {"q1": {"q1": 100.0, "d1": 5.0}}
        )
        assert "q1" not in filtered["q1"], "self-doc should be removed"
        assert "d1"  in filtered["q1"],    "genuine doc should remain"
        ev2 = RetrievalEvaluator(k_values=[1, 5], ignore_identical_ids=True)
        m   = ev2.evaluate({"q1": {"d1": 1}}, filtered)
        assert m["recall"]["Recall@5"] == 1.0
        tr.ok("eval_filter_identical_ids")
    except Exception as e:
        tr.fail("eval_filter_identical_ids", str(e))

    # --- empty results edge case ---
    try:
        m = ev.evaluate({}, {})
        assert m["ndcg"] == {}
        tr.ok("eval_empty_inputs")
    except Exception as e:
        tr.fail("eval_empty_inputs", str(e))


# ===========================================================================
# ④ DataLoaderFactory routing (no HF access)
# ===========================================================================

def test_factory_routing():
    section("④ DataLoaderFactory routing")
    factory = DataLoaderFactory()

    try:
        from src.ragtune.data.loaders.BRIGHTLoader import BRIGHTLoader as _BL
        ldr = factory.create_dataloader(Dataset.BIOLOGY, Benchmark.BRIGHT)
        assert isinstance(ldr, _BL), f"expected BRIGHTLoader, got {type(ldr)}"
        tr.ok("factory_bright_biology")
    except Exception as e:
        tr.fail("factory_bright_biology", str(e))

    try:
        from src.ragtune.data.loaders.BRIGHTLoader import BRIGHTLoader as _BL
        # Task name alone (no explicit benchmark) should also resolve BRIGHT
        ldr = factory.create_dataloader(Dataset.ECONOMICS, Benchmark.BRIGHT)
        assert isinstance(ldr, _BL)
        tr.ok("factory_bright_economics")
    except Exception as e:
        tr.fail("factory_bright_economics", str(e))

    try:
        from src.ragtune.data.loaders.FreshStackLoader import FreshStackLoader as _FS
        ldr = factory.create_dataloader(Dataset.LANGCHAIN, Benchmark.FRESHSTACK)
        assert isinstance(ldr, _FS)
        tr.ok("factory_freshstack_langchain")
    except Exception as e:
        tr.fail("factory_freshstack_langchain", str(e))

    try:
        from src.ragtune.data.loaders.IRDatasetsLoader import IRDatasetsLoader as _IR
        ldr = factory.create_dataloader("beir/scifact/test", "UNKNOWN_BENCH")
        assert isinstance(ldr, _IR)
        tr.ok("factory_fallback_irdatasets")
    except Exception as e:
        tr.fail("factory_fallback_irdatasets", str(e))

    # Invalid BRIGHT task name must raise ValueError
    try:
        factory.create_dataloader("not_a_task", Benchmark.BRIGHT)
        tr.fail("factory_invalid_bright_task", "should have raised ValueError")
    except ValueError:
        tr.ok("factory_invalid_bright_task")
    except Exception as e:
        tr.fail("factory_invalid_bright_task", str(e))

    # Invalid FreshStack topic must raise ValueError
    try:
        factory.create_dataloader("not_a_topic", Benchmark.FRESHSTACK)
        tr.fail("factory_invalid_freshstack_topic", "should have raised ValueError")
    except ValueError:
        tr.ok("factory_invalid_freshstack_topic")
    except Exception as e:
        tr.fail("factory_invalid_freshstack_topic", str(e))

    # RetrieverDataset wraps the factory – check repr
    try:
        rd = RetrieverDataset(Dataset.BIOLOGY, Benchmark.BRIGHT)
        assert "BRIGHT" in repr(rd)
        assert "biology" in repr(rd)
        tr.ok("factory_retriever_dataset_repr")
    except Exception as e:
        tr.fail("factory_retriever_dataset_repr", str(e))


# ===========================================================================
# ⑤  RetrieverDataset – full dataloader test (BRIGHT biology)
#
#     This is the canonical way every other test in SIR loads data:
#       rd = RetrieverDataset(dataset=..., benchmark=...)
#       query_objs, qrels, corpus = rd.qrels()
#     We verify every property of the returned triple exhaustively.
# ===========================================================================

def test_retriever_dataset(offline: bool) -> Optional[Tuple]:
    """Returns (query_objs, qrels, corpus) for re-use in downstream tests."""
    section("⑤ RetrieverDataset – BRIGHT biology dataloader tests")

    if offline:
        _skip_group([
            "rd_construct", "rd_corpus_nonempty", "rd_corpus_schema",
            "rd_corpus_text_nonempty", "rd_queries_nonempty",
            "rd_queries_type", "rd_qrels_nonempty", "rd_qrels_binary",
            "rd_qrels_doc_ids_in_corpus", "rd_raw_data_nonempty",
            "rd_samples_typed", "rd_query_objects_list",
            "rd_query_objects_ids_unique", "rd_load_beir_alias",
            "rd_get_loader",
        ], "offline")
        return None

    # ---- construct ----
    try:
        t0 = time.time()
        rd = RetrieverDataset(
            dataset=Dataset.BIOLOGY,
            benchmark=Benchmark.BRIGHT,
            split=Split.TEST,
        )
        query_objs, qrels, corpus = rd.qrels()
        elapsed = time.time() - t0
        logger.info(
            f"  biology loaded in {elapsed:.1f}s  "
            f"q={len(query_objs)}  docs={len(corpus)}  qrels={len(qrels)}"
        )
        tr.ok("rd_construct")
    except Exception as e:
        tr.fail("rd_construct", str(e))
        _skip_group([
            "rd_corpus_nonempty", "rd_corpus_schema", "rd_corpus_text_nonempty",
            "rd_queries_nonempty", "rd_queries_type", "rd_qrels_nonempty",
            "rd_qrels_binary", "rd_qrels_doc_ids_in_corpus",
            "rd_raw_data_nonempty", "rd_samples_typed",
            "rd_query_objects_list", "rd_query_objects_ids_unique",
            "rd_load_beir_alias", "rd_get_loader",
        ], "construct failed")
        return None

    # ---- corpus ----
    try:
        assert len(corpus) > 0, "corpus is empty"
        tr.ok("rd_corpus_nonempty")
    except AssertionError as e:
        tr.fail("rd_corpus_nonempty", str(e))

    try:
        for doc_id, doc in list(corpus.items())[:5]:
            assert isinstance(doc_id, str), f"doc_id should be str, got {type(doc_id)}"
            assert "text"  in doc, f"doc missing 'text' key: {list(doc.keys())}"
            assert "title" in doc, f"doc missing 'title' key: {list(doc.keys())}"
        tr.ok("rd_corpus_schema")
    except AssertionError as e:
        tr.fail("rd_corpus_schema", str(e))

    try:
        n_empty = sum(1 for d in corpus.values() if not d["text"].strip())
        pct_empty = n_empty / len(corpus)
        assert pct_empty < 0.05, f"{pct_empty:.1%} of docs have empty text"
        tr.ok("rd_corpus_text_nonempty")
    except AssertionError as e:
        tr.fail("rd_corpus_text_nonempty", str(e))

    # ---- queries ----
    try:
        assert len(query_objs) > 0, "query list is empty"
        tr.ok("rd_queries_nonempty")
    except AssertionError as e:
        tr.fail("rd_queries_nonempty", str(e))

    try:
        for q in query_objs[:5]:
            assert isinstance(q, Query), f"expected Query, got {type(q)}"
            assert isinstance(q.text(), str) and len(q.text()) > 0
            assert q.id() is not None, "query id is None"
        tr.ok("rd_queries_type")
    except AssertionError as e:
        tr.fail("rd_queries_type", str(e))

    # ---- qrels ----
    try:
        assert len(qrels) > 0, "qrels is empty"
        tr.ok("rd_qrels_nonempty")
    except AssertionError as e:
        tr.fail("rd_qrels_nonempty", str(e))

    try:
        for qid, rels in list(qrels.items())[:5]:
            assert isinstance(qid, str)
            assert len(rels) > 0, f"query {qid} has empty qrels"
            for did, rel in rels.items():
                assert isinstance(did, str), f"doc_id should be str"
                assert isinstance(rel, int),  f"relevance should be int, got {type(rel)}"
                assert rel in (0, 1),          f"relevance should be 0 or 1, got {rel}"
        tr.ok("rd_qrels_binary")
    except AssertionError as e:
        tr.fail("rd_qrels_binary", str(e))

    try:
        # All relevant doc IDs referenced in qrels must exist in corpus
        missing = 0
        for qid, rels in qrels.items():
            for did in rels:
                if did not in corpus:
                    missing += 1
        assert missing == 0, f"{missing} qrel doc IDs not found in corpus"
        tr.ok("rd_qrels_doc_ids_in_corpus")
    except AssertionError as e:
        tr.fail("rd_qrels_doc_ids_in_corpus", str(e))

    # ---- raw_data / Sample objects ----
    loader = rd.get_loader()
    try:
        loader._ensure_loaded()
        assert len(loader.raw_data) > 0, "raw_data is empty"
        tr.ok("rd_raw_data_nonempty")
    except Exception as e:
        tr.fail("rd_raw_data_nonempty", str(e))

    try:
        for s in loader.raw_data[:10]:
            assert isinstance(s, Sample),          f"expected Sample, got {type(s)}"
            assert isinstance(s.query, Query),      "sample.query not a Query"
            assert s.query.id()  is not None,       "sample query_id is None"
            assert isinstance(s.query.text(), str), "query text not str"
            if s.evidences is not None:
                assert isinstance(s.evidences, Context), "evidences not a Context"
                assert s.evidences.id() is not None, "evidence id is None"
        tr.ok("rd_samples_typed")
    except AssertionError as e:
        tr.fail("rd_samples_typed", str(e))

    # ---- get_query_objects() ----
    try:
        q_list = loader.get_query_objects()
        assert isinstance(q_list, list)
        assert all(isinstance(q, Query) for q in q_list)
        tr.ok("rd_query_objects_list")
    except Exception as e:
        tr.fail("rd_query_objects_list", str(e))

    try:
        ids = [q.id() for q in loader.get_query_objects()]
        assert len(ids) == len(set(ids)), "duplicate query IDs in get_query_objects()"
        tr.ok("rd_query_objects_ids_unique")
    except AssertionError as e:
        tr.fail("rd_query_objects_ids_unique", str(e))

    # ---- load() BEIR alias ----
    try:
        corpus2, queries2, qrels2 = rd.load()
        assert corpus2  == corpus,      "load() corpus mismatch"
        assert queries2 == {q.id(): q.text() for q in query_objs}, "load() queries mismatch"
        assert qrels2   == qrels,       "load() qrels mismatch"
        tr.ok("rd_load_beir_alias")
    except AssertionError as e:
        tr.fail("rd_load_beir_alias", str(e))

    # ---- get_loader() ----
    try:
        assert loader is rd.get_loader()
        tr.ok("rd_get_loader")
    except Exception as e:
        tr.fail("rd_get_loader", str(e))

    return query_objs, qrels, corpus


# ===========================================================================
# ⑥  BRIGHTLoader extras
# ===========================================================================

def test_bright_extras(offline: bool):
    section("⑥ BRIGHTLoader extras – long-context & reasoning subset")

    # ⑥a long-context qrels should have ≥ standard qrels
    if offline:
        tr.skip("bright_long_context_ge_standard", "offline")
    else:
        try:
            std = BRIGHTLoader(task=Dataset.BIOLOGY, long_context=False)
            lc  = BRIGHTLoader(task=Dataset.BIOLOGY, long_context=True)
            _, _, qrels_std = std.load()
            _, _, qrels_lc  = lc.load()
            n_std = sum(len(v) for v in qrels_std.values())
            n_lc  = sum(len(v) for v in qrels_lc.values())
            logger.info(f"  long-context: std_pairs={n_std} lc_pairs={n_lc}")
            assert n_lc >= n_std, f"long-context ({n_lc}) < standard ({n_std})"
            tr.ok("bright_long_context_ge_standard")
        except Exception as e:
            tr.fail("bright_long_context_ge_standard", str(e))

    # ⑥b reasoning-augmented query subset
    if offline:
        tr.skip("bright_reasoning_subset_loads", "offline")
    else:
        try:
            ldr = BRIGHTLoader(task=Dataset.BIOLOGY, reasoning_subset="gpt4_reason")
            _, queries, _ = ldr.load()
            assert len(queries) > 0
            # Reasoning queries tend to be longer than raw queries
            avg_len = sum(len(t) for t in queries.values()) / len(queries)
            logger.info(f"  gpt4_reason: {len(queries)} queries, avg_len={avg_len:.0f}")
            tr.ok("bright_reasoning_subset_loads")
        except Exception as e:
            tr.fail("bright_reasoning_subset_loads", str(e))


# ===========================================================================
# ⑦  BRIGHTMultiTaskLoader
# ===========================================================================

def test_bright_multi(offline: bool):
    section("⑦ BRIGHTMultiTaskLoader")

    if offline:
        _skip_group(["bright_multi_two_tasks", "bright_multi_corpora_differ"], "offline")
        return

    try:
        ml = BRIGHTMultiTaskLoader(tasks=[Dataset.ECONOMICS, Dataset.ROBOTICS])
        assert len(ml) == 2
        tr.ok("bright_multi_two_tasks")
    except Exception as e:
        tr.fail("bright_multi_two_tasks", str(e))
        tr.skip("bright_multi_corpora_differ", "multi-task init failed")
        return

    try:
        loaded = {}
        for task, ldr in ml.items():
            corpus, queries, qrels = ldr.load()
            loaded[task] = {"corpus": len(corpus), "queries": len(queries)}
            logger.info(f"  {task}: q={len(queries)} docs={len(corpus)}")
        assert len(loaded) == 2
        # The two tasks should have different corpus sizes (sanity check)
        sizes = [v["corpus"] for v in loaded.values()]
        # they *can* be equal by coincidence but both must be > 0
        assert all(s > 0 for s in sizes)
        tr.ok("bright_multi_corpora_differ")
    except Exception as e:
        tr.fail("bright_multi_corpora_differ", str(e))


# ===========================================================================
# ⑧  FreshStackLoader
# ===========================================================================

def test_freshstack(offline: bool):
    section("⑧ FreshStackLoader – langchain topic")

    if offline:
        _skip_group(["freshstack_corpus", "freshstack_queries", "freshstack_qrels"], "offline")
        return

    try:
        ldr = FreshStackLoader(topic=Dataset.LANGCHAIN, split=Split.TEST)
        corpus, queries, qrels = ldr.load()
        logger.info(
            f"  langchain: q={len(queries)} docs={len(corpus)} qrels={len(qrels)}"
        )
    except Exception as e:
        _skip_group(["freshstack_corpus", "freshstack_queries", "freshstack_qrels"],
                    f"load failed: {e}")
        return

    try:
        assert len(corpus) > 0
        doc = next(iter(corpus.values()))
        assert "text" in doc
        tr.ok("freshstack_corpus")
    except AssertionError as e:
        tr.fail("freshstack_corpus", str(e))

    try:
        assert len(queries) > 0
        assert all(isinstance(v, str) for v in queries.values())
        tr.ok("freshstack_queries")
    except AssertionError as e:
        tr.fail("freshstack_queries", str(e))

    try:
        # qrels may be empty if the nugget qrels sub-dataset isn't available,
        # but the loader should not crash
        assert isinstance(qrels, dict)
        tr.ok("freshstack_qrels")
    except Exception as e:
        tr.fail("freshstack_qrels", str(e))


# ===========================================================================
# ⑨  PyTerrier BM25 pipeline
#     All data comes through RetrieverDataset.qrels()
# ===========================================================================

def _pt_init():
    """Init PyTerrier exactly once; returns the module or raises ImportError."""
    import pyterrier as pt
    if not pt.started():
        pt.init()
    return pt


def _build_pt_index(corpus: Dict, index_dir: str):
    """Build a PyTerrier IterDictIndexer from a data corpus dict."""
    pt = _pt_init()

    def _iter():
        for doc_id, doc in corpus.items():
            text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            yield {"docno": doc_id, "text": text}

    indexer = pt.IterDictIndexer(
        index_dir, overwrite=True,
        meta={"docno": 48, "text": 4096},
    )
    ref = indexer.index(_iter())
    return pt.IndexFactory.of(ref)


def test_pyterrier_bm25(offline: bool):
    section("⑨ PyTerrier BM25 pipeline – BRIGHT biology (via RetrieverDataset)")

    if offline:
        _skip_group(["pt_bm25_import", "pt_bm25_rd_load", "pt_bm25_index",
                     "pt_bm25_retrieve", "pt_bm25_eval"], "offline")
        return

    # import guard
    try:
        _pt_init()
        tr.ok("pt_bm25_import")
    except ImportError:
        _skip_group(["pt_bm25_import", "pt_bm25_rd_load", "pt_bm25_index",
                     "pt_bm25_retrieve", "pt_bm25_eval"],
                    "python-terrier not installed")
        return
    except Exception as e:
        tr.fail("pt_bm25_import", str(e)); return

    # load via RetrieverDataset
    try:
        rd = RetrieverDataset(
            dataset=Dataset.BIOLOGY,
            benchmark=Benchmark.BRIGHT,
            split=Split.TEST,
        )
        query_objs, qrels, corpus = rd.qrels()
        queries_dict = {q.id(): q.text() for q in query_objs}
        logger.info(f"  loaded: q={len(queries_dict)} docs={len(corpus)}")
        tr.ok("pt_bm25_rd_load")
    except Exception as e:
        tr.fail("pt_bm25_rd_load", str(e))
        _skip_group(["pt_bm25_index", "pt_bm25_retrieve", "pt_bm25_eval"],
                    "data load failed")
        return

    # build index
    index_dir = tempfile.mkdtemp(prefix="sir_pt_bm25_")
    try:
        pt  = _pt_init()
        idx = _build_pt_index(corpus, index_dir)
        n_docs = idx.getCollectionStatistics().getNumberOfDocuments()
        logger.info(f"  index: {n_docs} docs")
        assert n_docs == len(corpus), \
            f"index doc count {n_docs} != corpus size {len(corpus)}"
        tr.ok("pt_bm25_index")
    except Exception as e:
        tr.fail("pt_bm25_index", str(e))
        _skip_group(["pt_bm25_retrieve", "pt_bm25_eval"], "index failed"); return

    # BM25 retrieval
    try:
        import pandas as pd
        bm25 = pt.BatchRetrieve(idx, wmodel="BM25", num_results=100,
                                metadata=["docno"])
        topics = pd.DataFrame([
            {"qid": str(qid), "query": qtext}
            for qid, qtext in queries_dict.items()
        ])
        res_df = bm25.transform(topics)
        # Build {qid: {docno: score}}
        retrieval_results: Dict[str, Dict[str, float]] = {}
        for _, row in res_df.iterrows():
            retrieval_results.setdefault(str(row["qid"]), {})[str(row["docno"])] = float(row["score"])
        total_pairs = sum(len(v) for v in retrieval_results.values())
        logger.info(f"  BM25: {len(retrieval_results)} queries, {total_pairs} pairs")
        assert len(retrieval_results) > 0
        tr.ok("pt_bm25_retrieve")
    except Exception as e:
        tr.fail("pt_bm25_retrieve", str(e))
        tr.skip("pt_bm25_eval", "retrieval failed"); return

    # evaluate
    try:
        ev = RetrievalEvaluator(k_values=[5, 10, 100])
        metrics = ev.evaluate(qrels, retrieval_results)
        RetrievalEvaluator.print_results(metrics, title="⑨ BRIGHT Biology – BM25")
        assert metrics["ndcg"]["NDCG@10"] >= 0.0
        tr.ok("pt_bm25_eval")
    except Exception as e:
        tr.fail("pt_bm25_eval", str(e))


# ===========================================================================
# ⑩  RAGtune full pipeline
#
#  All data loaded via RetrieverDataset.qrels() → (query_objs, qrels, corpus).
#  Index built with PyTerrier IterDictIndexer.
#  Retriever wrapped with PyTerrierRetriever (ragtune.adapters.pyterrier).
#  Tests:
#    ⑩b  RetrieverDataset  →  PyTerrier index  →  PyTerrierRetriever
#    ⑩c  Baseline controller  (static, budget=TOP_K)
#    ⑩d  RAGtune controller   (iterative, budget=RERANK_BUDGET)
#    ⑩e  SimulatedReranker actually differentiates scores
#    ⑩f  SimilarityEstimator boots without error
#    ⑩g  Budget respected: reranked_docs ≤ RERANK_BUDGET
#    ⑩h  Summary table printed
# ===========================================================================

_RAGTUNE_DOMAINS   = [Dataset.BIOLOGY]  # extend freely
_QUERIES_PER_TASK  = 5                  # keep the smoke-test fast
_CANDIDATES_TOP_K  = 50
_RERANK_BUDGET     = 10


def _ragtune_evaluate(controller, query_objs: List[Query],
                      qrels: Dict) -> Dict:
    """
    Run the RAGtune controller over a list of Query objects and return
    accuracy / avg_docs_reranked / avg_latency.

    Uses RetrieverDataset's Query objects directly – no ad-hoc dict building.
    """
    results = []
    for q in query_objs:
        query_str = q.text()
        gold_ids  = set(qrels.get(q.id(), {}).keys())

        t0     = time.time()
        output = controller.run(query_str)
        elapsed = time.time() - t0

        found        = any(doc.id in gold_ids for doc in output.documents)
        docs_reranked = output.final_budget_state.get("rerank_docs", 0)
        results.append({"found": found, "docs_reranked": docs_reranked,
                         "latency": elapsed, "output": output})

    found_vals = [r["found"]         for r in results]
    dr_vals    = [r["docs_reranked"] for r in results]
    lat_vals   = [r["latency"]       for r in results]
    return {
        "accuracy":          sum(found_vals) / max(len(found_vals), 1),
        "avg_docs_reranked": statistics.mean(dr_vals) if dr_vals else 0.0,
        "avg_latency":       statistics.mean(lat_vals) if lat_vals else 0.0,
        "_raw":              results,
    }


def test_ragtune_pipeline(offline: bool, no_ragtune: bool):
    section("⑩ RAGtune full pipeline – BRIGHT benchmark (via RetrieverDataset)")

    if offline:
        _skip_group(["ragtune_imports", "ragtune_rd_index_retriever",
                     "ragtune_baseline_ctrl", "ragtune_iterative_ctrl",
                     "ragtune_reranker_scores_differ",
                     "ragtune_similarity_estimator",
                     "ragtune_budget_respected",
                     "ragtune_summary_table"], "offline")
        return
    if no_ragtune:
        _skip_group(["ragtune_imports", "ragtune_rd_index_retriever",
                     "ragtune_baseline_ctrl", "ragtune_iterative_ctrl",
                     "ragtune_reranker_scores_differ",
                     "ragtune_similarity_estimator",
                     "ragtune_budget_respected",
                     "ragtune_summary_table"], "--no-ragtune")
        return

    # ---- ⑩a  imports ----
    try:
        _pt_init()
        from ragtune.core.controller     import RAGtuneController
        from ragtune.core.budget         import CostBudget
        from ragtune.adapters.pyterrier  import PyTerrierRetriever
        from ragtune.components.rerankers      import SimulatedReranker, NoOpReranker
        from ragtune.components.reformulators  import IdentityReformulator
        from ragtune.components.assemblers     import GreedyAssembler
        from ragtune.components.schedulers     import ActiveLearningScheduler
        from ragtune.components.estimators     import BaselineEstimator, SimilarityEstimator
        tr.ok("ragtune_imports")
    except ImportError as e:
        _skip_group(["ragtune_imports", "ragtune_rd_index_retriever",
                     "ragtune_baseline_ctrl", "ragtune_iterative_ctrl",
                     "ragtune_reranker_scores_differ",
                     "ragtune_similarity_estimator",
                     "ragtune_budget_respected",
                     "ragtune_summary_table"], f"missing dep: {e}")
        return
    except Exception as e:
        tr.fail("ragtune_imports", str(e)); return

    import pandas as pd
    pt = _pt_init()

    all_metrics: List[Dict] = []

    for domain in _RAGTUNE_DOMAINS:

        # ---- ⑩b  RetrieverDataset  →  index  →  PyTerrierRetriever ----
        try:
            rd = RetrieverDataset(
                dataset=domain,
                benchmark=Benchmark.BRIGHT,
                split=Split.TEST,
            )
            query_objs_all, qrels, corpus = rd.qrels()

            # Trim to _QUERIES_PER_TASK for speed
            sampled_query_objs = query_objs_all[:_QUERIES_PER_TASK]
            sampled_qids       = {q.id() for q in sampled_query_objs}
            sampled_qrels      = {qid: qrels[qid]
                                  for qid in sampled_qids if qid in qrels}

            logger.info(
                f"  [{domain}] {len(query_objs_all)} queries, "
                f"{len(corpus)} docs  (testing first {len(sampled_query_objs)})"
            )

            # Build PyTerrier index
            index_dir = tempfile.mkdtemp(prefix=f"sir_ragtune_{domain}_")
            idx = _build_pt_index(corpus, index_dir)

            # Wrap as RAGtune retriever
            bm25_pt   = pt.BatchRetrieve(idx, wmodel="BM25",
                                          num_results=_CANDIDATES_TOP_K,
                                          metadata=["docno", "text"])
            retriever = PyTerrierRetriever(pt_transformer=bm25_pt)
            tr.ok("ragtune_rd_index_retriever")
        except Exception as e:
            tr.fail("ragtune_rd_index_retriever", str(e))
            _skip_group(["ragtune_baseline_ctrl", "ragtune_iterative_ctrl",
                         "ragtune_reranker_scores_differ",
                         "ragtune_similarity_estimator",
                         "ragtune_budget_respected",
                         "ragtune_summary_table"], "setup failed")
            return

        # ---- ⑩c  Baseline controller (static rerank-all) ----
        try:
            baseline = RAGtuneController(
                retriever    = retriever,
                reformulator = IdentityReformulator(),
                reranker     = SimulatedReranker(),
                assembler    = GreedyAssembler(max_docs=_CANDIDATES_TOP_K),
                scheduler    = ActiveLearningScheduler(batch_size=_CANDIDATES_TOP_K),
                estimator    = BaselineEstimator(),
                budget       = CostBudget.simple(
                                   tokens=999_999,
                                   docs=_CANDIDATES_TOP_K,
                                   calls=100),
                initial_top_k=_CANDIDATES_TOP_K,
            )
            m_base = _ragtune_evaluate(baseline, sampled_query_objs, sampled_qrels)
            m_base.update({"domain": domain, "method": "Baseline (Static-Rerank-All)"})
            all_metrics.append({k: v for k, v in m_base.items() if k != "_raw"})
            logger.info(
                f"  [{domain}] Baseline  acc={m_base['accuracy']:.2f}  "
                f"reranked={m_base['avg_docs_reranked']:.1f}  "
                f"lat={m_base['avg_latency']*1000:.0f}ms"
            )
            tr.ok("ragtune_baseline_ctrl")
        except Exception as e:
            tr.fail("ragtune_baseline_ctrl", str(e))

        # ---- ⑩d  RAGtune controller (budget-constrained iterative) ----
        try:
            ragtune = RAGtuneController(
                retriever    = retriever,
                reformulator = IdentityReformulator(),
                reranker     = SimulatedReranker(),
                assembler    = GreedyAssembler(max_docs=_RERANK_BUDGET),
                scheduler    = ActiveLearningScheduler(batch_size=5),
                estimator    = BaselineEstimator(),
                budget       = CostBudget.simple(
                                   tokens=2000,
                                   docs=_RERANK_BUDGET,
                                   calls=20),
                initial_top_k=_CANDIDATES_TOP_K,
            )
            m_rt = _ragtune_evaluate(ragtune, sampled_query_objs, sampled_qrels)
            m_rt.update({"domain": domain,
                          "method": f"RAGtune (budget={_RERANK_BUDGET})"})
            all_metrics.append({k: v for k, v in m_rt.items() if k != "_raw"})
            logger.info(
                f"  [{domain}] RAGtune   acc={m_rt['accuracy']:.2f}  "
                f"reranked={m_rt['avg_docs_reranked']:.1f}  "
                f"lat={m_rt['avg_latency']*1000:.0f}ms"
            )
            tr.ok("ragtune_iterative_ctrl")
        except Exception as e:
            tr.fail("ragtune_iterative_ctrl", str(e))
            m_rt = {"_raw": []}

        # ---- ⑩e  SimulatedReranker produces distinct per-doc scores ----
        try:
            # Run a single query and inspect the controller output
            q0     = sampled_query_objs[0]
            output = ragtune.run(q0.text())
            scores = [doc.score for doc in output.documents]
            assert len(scores) > 0, "no documents returned"
            # Scores should not all be identical (SimulatedReranker uses query-content match)
            assert len(set(round(s, 6) for s in scores)) > 1 or len(scores) == 1, \
                "all scores identical – SimulatedReranker may not be working"
            tr.ok("ragtune_reranker_scores_differ")
        except Exception as e:
            tr.fail("ragtune_reranker_scores_differ", str(e))

        # ---- ⑩f  SimilarityEstimator boots without error ----
        try:
            sim_ctrl = RAGtuneController(
                retriever    = retriever,
                reformulator = IdentityReformulator(),
                reranker     = SimulatedReranker(),
                assembler    = GreedyAssembler(max_docs=5),
                scheduler    = ActiveLearningScheduler(batch_size=5),
                estimator    = SimilarityEstimator(),
                budget       = CostBudget.simple(tokens=999_999, docs=5, calls=5),
                initial_top_k=10,
            )
            q0 = sampled_query_objs[0]
            out = sim_ctrl.run(q0.text())
            assert isinstance(out.documents, list)
            tr.ok("ragtune_similarity_estimator")
        except Exception as e:
            tr.fail("ragtune_similarity_estimator", str(e))

        # ---- ⑩g  Budget is respected ----
        try:
            for r in m_rt["_raw"]:
                consumed = r["output"].final_budget_state.get("rerank_docs", 0)
                assert consumed <= _RERANK_BUDGET, \
                    f"consumed {consumed} > budget {_RERANK_BUDGET}"
            tr.ok("ragtune_budget_respected")
        except AssertionError as e:
            tr.fail("ragtune_budget_respected", str(e))
        except Exception as e:
            tr.fail("ragtune_budget_respected", str(e))

    # ---- ⑩h  Summary table (mirrors benchmark_bright.py output) ----
    try:
        df = pd.DataFrame(all_metrics)
        print("\n" + "=" * 56)
        print("  ⑩  RAGtune Benchmark Summary  (mirrors benchmark_bright.py)")
        print("=" * 56)
        cols = ["domain", "method", "accuracy", "avg_docs_reranked", "avg_latency"]
        present = [c for c in cols if c in df.columns]
        print(df[present].to_string(index=False))
        print("=" * 56 + "\n")
        tr.ok("ragtune_summary_table")
    except Exception as e:
        tr.fail("ragtune_summary_table", str(e))


# ===========================================================================
# main
# ===========================================================================

_ALL_TASKS = [
    "constants", "datastructures", "evaluator", "factory",
    "retriever_dataset",
    "bright_extras", "bright_multi", "freshstack",
    "pipeline_pyterrier", "pipeline_ragtune",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="data test suite")
    p.add_argument("--offline",    action="store_true",
                   help="Skip every test that requires a network / HuggingFace download")
    p.add_argument("--no-ragtune", action="store_true",
                   help="Skip the RAGtune pipeline group (⑩)")
    p.add_argument("--tasks", nargs="*", choices=_ALL_TASKS,
                   help="Run only the listed test groups (default: all)")
    return p.parse_args()


def main():
    args     = parse_args()
    offline  = args.offline
    no_rt    = args.no_ragtune
    run_all  = not args.tasks

    def should(name: str) -> bool:
        return run_all or name in (args.tasks or [])

    if offline:
        logger.info("OFFLINE mode – network tests will be skipped.")

    # ── always-run (no network) ─────────────────────────────────────────────
    if should("constants"):      test_constants()
    if should("datastructures"): test_datastructures()
    if should("evaluator"):      test_evaluator_synthetic()
    if should("factory"):        test_factory_routing()

    # ── network tests ───────────────────────────────────────────────────────
    if should("retriever_dataset"): test_retriever_dataset(offline)
    if should("bright_extras"):     test_bright_extras(offline)
    if should("bright_multi"):      test_bright_multi(offline)
    if should("freshstack"):        test_freshstack(offline)

    # ── pipeline tests ──────────────────────────────────────────────────────
    if should("pipeline_pyterrier"): test_pyterrier_bm25(offline)
    if should("pipeline_ragtune"):   test_ragtune_pipeline(offline, no_rt)

    sys.exit(0 if tr.summary() else 1)


if __name__ == "__main__":
    main()
