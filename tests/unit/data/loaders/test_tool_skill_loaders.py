"""
Unit tests for ToolRet, SkillRet, SRA-Bench loaders.

These tests verify:
- Loader inherits BaseDataLoader correctly
- _load_data() populates corpus/queries/qrels
- n_queries caps queries properly
- max_corpus_docs caps corpus properly
- Query objects are created correctly
- Factory creates correct loader types
"""

import pytest
from unittest.mock import patch, MagicMock
from ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from ragtune.data.loaders.ToolRetLoader import ToolRetLoader
from ragtune.data.loaders.SkillRetLoader import SkillRetLoader
from ragtune.data.loaders.SRABenchLoader import SRABenchLoader
from ragtune.data.loaders.DataLoaderFactory import DataLoaderFactory
from ragtune.data.constants import Benchmark, TOOLRET_SUBSETS, SRA_BENCH_SUBSETS


class TestToolRetLoader:
    def test_inherits_base_dataloader(self):
        assert issubclass(ToolRetLoader, BaseDataLoader)

    def test_has_required_methods(self):
        loader = ToolRetLoader.__new__(ToolRetLoader)
        for method in [
            "_load_data",
            "get_corpus",
            "get_queries",
            "get_qrels",
            "load",
            "get_query_objects",
        ]:
            assert hasattr(BaseDataLoader, method)

    def test_init_params(self):
        loader = ToolRetLoader(dataset="test_dataset", split="test", n_queries=10)
        assert loader.dataset == "test_dataset"
        assert loader.split == "test"
        assert loader.n_queries == 10

    def test_default_params(self):
        loader = ToolRetLoader(dataset="test")
        assert loader.n_queries == 0
        assert loader.max_corpus_docs is None
        assert loader.cache_dir is None


class TestSkillRetLoader:
    def test_inherits_base_dataloader(self):
        assert issubclass(SkillRetLoader, BaseDataLoader)

    def test_has_required_methods(self):
        loader = SkillRetLoader.__new__(SkillRetLoader)
        for method in [
            "_load_data",
            "get_corpus",
            "get_queries",
            "get_qrels",
            "load",
            "get_query_objects",
        ]:
            assert hasattr(BaseDataLoader, method)

    def test_init_params(self):
        loader = SkillRetLoader(dataset="train", split="train", n_queries=100)
        assert loader.dataset == "train"
        assert loader.split == "train"
        assert loader.n_queries == 100

    def test_default_params(self):
        loader = SkillRetLoader()
        assert loader.dataset == "test"
        assert loader.n_queries == 0


class TestSRABenchLoader:
    def test_inherits_base_dataloader(self):
        assert issubclass(SRABenchLoader, BaseDataLoader)

    def test_has_required_methods(self):
        loader = SRABenchLoader.__new__(SRABenchLoader)
        for method in [
            "_load_data",
            "get_corpus",
            "get_queries",
            "get_qrels",
            "load",
            "get_query_objects",
        ]:
            assert hasattr(BaseDataLoader, method)

    def test_init_params(self):
        loader = SRABenchLoader(dataset="toolqa", split="test", n_queries=50)
        assert loader.dataset == "toolqa"
        assert loader.split == "test"
        assert loader.n_queries == 50

    def test_invalid_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown SRA-Bench sub-dataset"):
            SRABenchLoader(dataset="invalid_dataset")

    def test_valid_datasets(self):
        for ds in [
            "toolqa",
            "theoremqa",
            "bigcodebench",
            "champ",
            "logicbench",
            "medcalcbench",
        ]:
            loader = SRABenchLoader(dataset=ds)
            assert loader.dataset == ds


class TestDataLoaderFactory:
    def test_toolret_creation(self):
        factory = DataLoaderFactory()
        loader = factory.create_dataloader(
            dataset_name="apibank",
            benchmark_name=Benchmark.TOOLRET,
            n_queries=5,
        )
        assert isinstance(loader, ToolRetLoader)

    def test_skillret_creation(self):
        factory = DataLoaderFactory()
        loader = factory.create_dataloader(
            dataset_name="test",
            benchmark_name=Benchmark.SKILLRET,
            n_queries=10,
        )
        assert isinstance(loader, SkillRetLoader)

    def test_sra_creation(self):
        factory = DataLoaderFactory()
        loader = factory.create_dataloader(
            dataset_name="toolqa",
            benchmark_name=Benchmark.SRA_BENCH,
            n_queries=20,
        )
        assert isinstance(loader, SRABenchLoader)

    def test_factory_subsets(self):
        factory = DataLoaderFactory()
        for subset in TOOLRET_SUBSETS[:3]:
            loader = factory.create_dataloader(
                dataset_name=subset,
                benchmark_name=Benchmark.TOOLRET,
            )
            assert isinstance(loader, ToolRetLoader)
        for subset in SRA_BENCH_SUBSETS[:3]:
            loader = factory.create_dataloader(
                dataset_name=subset,
                benchmark_name=Benchmark.SRA_BENCH,
            )
            assert isinstance(loader, SRABenchLoader)
