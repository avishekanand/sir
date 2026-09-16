"""
Integration test for the full tool retrieval pipeline.

Tests the complete flow:
    DataLoaderFactory → IndexFactory → ConfigLoader → RetrievalEvaluator

This single test verifies all components work together end-to-end.
"""

import os
import pytest
from ragtune.data.loaders import DataLoaderFactory
from ragtune.data.constants import Benchmark
from ragtune.indexing import IndexFactory
from ragtune.cli.config_loader import ConfigLoader
from ragtune.evaluation.RetrievalEvaluator import RetrievalEvaluator

# Import adapters to register retrievers, rerankers in the registry
import ragtune.adapters  # noqa: F401
import ragtune.components  # noqa: F401


class TestFullPipeline:
    """End-to-end integration test: load → index → retrieve → evaluate."""

    @staticmethod
    def _build_retriever(corpus):
        import tempfile, pyterrier as pt

        if not pt.java.started():
            pt.java.init()
        indexer = IndexFactory.create("pyterrier")
        index_path = os.path.join(tempfile.mkdtemp(), "idx")
        indexer.build_from_corpus(corpus, index_path=index_path)
        idx_ref = pt.IndexFactory.of(index_path)
        bm25 = pt.terrier.Retriever(
            idx_ref, wmodel="BM25", metadata=["docno", "text"], num_results=10
        )
        from ragtune.adapters.pyterrier import PyTerrierRetriever

        return PyTerrierRetriever(bm25)

    def test_toolret_pipeline(self):
        """Test full pipeline on ToolRet apibank (smallest subset)."""
        factory = DataLoaderFactory()
        loader = factory.create_dataloader(
            dataset_name="apibank", benchmark_name=Benchmark.TOOLRET, n_queries=5
        )
        corpus, queries, qrels = loader.load()
        assert len(corpus) > 0
        assert len(queries) > 0
        assert len(qrels) > 0

        retriever = self._build_retriever(corpus)

        # Test ConfigLoader default scenarios
        scenarios = ConfigLoader.create_controllers_from_env(retriever)
        assert len(scenarios) >= 1

        # Run first scenario (BM25 baseline) on 3 queries
        evaluator = RetrievalEvaluator(k_values=[10])
        results = {}
        for qid, qtext in list(queries.items())[:3]:
            output = scenarios[0][1].run(qtext)
            results[qid] = {
                doc.id: 1.0 / (rank + 1) for rank, doc in enumerate(output.documents)
            }
        assert len(results) > 0

        metrics = evaluator.evaluate(qrels, results, k_values=[10])
        assert "ndcg" in metrics
        assert "NDCG@10" in metrics["ndcg"]
        assert metrics["ndcg"]["NDCG@10"] >= 0

        # Run second scenario on one query
        qid, qtext = list(queries.items())[0]
        output = scenarios[1][1].run(qtext)
        assert output is not None
        assert len(output.documents) > 0

        print(
            f"ToolRet pipeline OK: {len(corpus)} docs, NDCG@10={metrics['ndcg']['NDCG@10']:.4f}"
        )

    def test_skillret_pipeline(self):
        """Test full pipeline on SkillRet test split (limited)."""
        factory = DataLoaderFactory()
        loader = factory.create_dataloader(
            dataset_name="test", benchmark_name=Benchmark.SKILLRET, n_queries=5
        )
        corpus, queries, qrels = loader.load()
        assert len(corpus) > 0
        assert len(queries) > 0
        assert len(qrels) > 0

        retriever = self._build_retriever(corpus)
        scenarios = ConfigLoader.create_controllers_from_env(retriever)
        assert len(scenarios) >= 1

        evaluator = RetrievalEvaluator(k_values=[10])
        results = {}
        for qid, qtext in list(queries.items())[:3]:
            output = scenarios[0][1].run(qtext)
            results[qid] = {
                doc.id: 1.0 / (rank + 1) for rank, doc in enumerate(output.documents)
            }
        metrics = evaluator.evaluate(qrels, results, k_values=[10])
        assert metrics["ndcg"]["NDCG@10"] >= 0

        print(
            f"SkillRet pipeline OK: {len(corpus)} skills, NDCG@10={metrics['ndcg']['NDCG@10']:.4f}"
        )

    def test_sra_pipeline(self):
        """Test full pipeline on SRA-Bench toolqa (limited)."""
        factory = DataLoaderFactory()
        loader = factory.create_dataloader(
            dataset_name="toolqa", benchmark_name=Benchmark.SRA_BENCH, n_queries=5
        )
        corpus, queries, qrels = loader.load()
        assert len(corpus) > 0
        assert len(queries) > 0
        assert len(qrels) > 0

        retriever = self._build_retriever(corpus)
        scenarios = ConfigLoader.create_controllers_from_env(retriever)
        assert len(scenarios) >= 1

        evaluator = RetrievalEvaluator(k_values=[10])
        results = {}
        for qid, qtext in list(queries.items())[:3]:
            output = scenarios[0][1].run(qtext)
            results[qid] = {
                doc.id: 1.0 / (rank + 1) for rank, doc in enumerate(output.documents)
            }
        metrics = evaluator.evaluate(qrels, results, k_values=[10])
        assert metrics["ndcg"]["NDCG@10"] >= 0

        print(
            f"SRA-Bench pipeline OK: {len(corpus)} skills, NDCG@10={metrics['ndcg']['NDCG@10']:.4f}"
        )

    def test_scenarios_from_env(self):
        """Test ConfigLoader multi-scenario from env."""
        import json
        from ragtune.components.retrievers import InMemoryRetriever

        mock_retriever = InMemoryRetriever(
            documents=[{"id": "d1", "content": "test doc"}]
        )

        os.environ.pop("SCENARIOS", None)
        scenarios = ConfigLoader.create_controllers_from_env(mock_retriever)
        assert len(scenarios) == 7
        assert "BM25" in scenarios[0][0]

        os.environ["SCENARIOS"] = json.dumps(
            [
                {
                    "name": "custom",
                    "pipeline": {
                        "components": {"reranker": {"type": "noop"}},
                        "budget": {"limits": {"rerank_docs": 5}},
                    },
                }
            ]
        )
        scenarios = ConfigLoader.create_controllers_from_env(mock_retriever)
        assert len(scenarios) == 1
        assert scenarios[0][0] == "custom"
        os.environ.pop("SCENARIOS")

    def test_data_loader_factory_cache_dir(self):
        """Test that DataLoaderFactory forwards cache_dir to loaders."""
        factory = DataLoaderFactory()
        loader = factory.create_dataloader(
            dataset_name="toolqa",
            benchmark_name=Benchmark.SRA_BENCH,
            n_queries=3,
            cache_dir="/tmp/test_cache",
        )
        assert loader.cache_dir == "/tmp/test_cache"
