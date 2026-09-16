"""
Unit tests for IndexFactory and ConfigLoader scenario system.
"""

import os
import pytest
from ragtune.indexing import IndexFactory, PyTerrierIndexer
from ragtune.registry import registry
from ragtune.cli.config_loader import ConfigLoader

# Import adapters and components to register them in the registry
import ragtune.adapters  # noqa: F401 — registers retrievers
import ragtune.components  # noqa: F401 — registers rerankers, estimators, etc.


class TestIndexFactory:
    def test_create_pyterrier(self):
        indexer = IndexFactory.create("pyterrier")
        assert isinstance(indexer, PyTerrierIndexer)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown indexer type"):
            IndexFactory.create("invalid_nonexistent")

    def test_registered_indexers(self):
        all_indexers = registry.list_all().get("indexer", {})
        assert "pyterrier" in all_indexers
        assert "faiss" in all_indexers


class TestConfigLoaderScenarios:
    def test_default_scenarios(self):
        os.environ.pop("SCENARIOS", None)
        scenarios = ConfigLoader._default_scenarios()
        assert len(scenarios) == 7
        assert scenarios[0]["name"] == "bm25_only"
        assert scenarios[1]["name"] == "crossenc_tight"
        assert scenarios[6]["name"] == "crossenc_sim_loose"

    def test_default_scenarios_components(self):
        scenarios = ConfigLoader._default_scenarios()
        for s in scenarios:
            assert "pipeline" in s
            assert "budget" in s["pipeline"]
            assert "components" in s["pipeline"]

    def test_custom_scenarios_from_env(self):
        os.environ["SCENARIOS"] = (
            '[{"name":"test","pipeline":{"components":{"reranker":{"type":"noop"}},"budget":{"limits":{"rerank_docs":5}}}}]'
        )
        # Test that default scenarios + inject retriever = controller
        from ragtune.components.retrievers import InMemoryRetriever

        mock_retriever = InMemoryRetriever(
            documents=[{"id": "d1", "content": "test doc"}]
        )
        scenarios = ConfigLoader.create_controllers_from_env(retriever=mock_retriever)
        assert len(scenarios) == 1
        assert scenarios[0][0] == "test"
        assert scenarios[0][1] is not None
        os.environ.pop("SCENARIOS")

    def test_empty_scenarios_defaults(self):
        os.environ.pop("SCENARIOS", None)
        from ragtune.components.retrievers import InMemoryRetriever

        mock_retriever = InMemoryRetriever(
            documents=[{"id": "d1", "content": "test doc"}]
        )
        scenarios = ConfigLoader.create_controllers_from_env(retriever=mock_retriever)
        assert len(scenarios) == 7
        assert "bm25" in scenarios[0][0]
        assert scenarios[0][1] is not None
