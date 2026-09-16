"""
Unit tests for DataLoaderFactory dispatch (src/ragtune/data/loaders/DataLoaderFactory.py).

Loaders load lazily (`BaseDataLoader._ensure_loaded`), so constructing one
touches no network — these tests only assert which class the factory picks.
"""

import pytest

from ragtune.data.constants import COIR_DATASETS, Benchmark, Dataset, Split
from ragtune.data.loaders.BRIGHTLoader import BRIGHTLoader
from ragtune.data.loaders.CoIRLoader import CoIRLoader
from ragtune.data.loaders.DataLoaderFactory import DataLoaderFactory


@pytest.fixture
def factory():
    return DataLoaderFactory()


@pytest.mark.parametrize("dataset_name", COIR_DATASETS)
def test_coir_dispatch_by_benchmark_name(factory, dataset_name):
    loader = factory.create_dataloader(
        dataset_name=dataset_name, benchmark_name=Benchmark.COIR
    )
    assert isinstance(loader, CoIRLoader)
    assert loader.dataset == f"CoIR-Retrieval/{dataset_name}"
    assert loader.split == Split.TEST


@pytest.mark.parametrize("dataset_name", COIR_DATASETS)
def test_coir_dispatch_by_dataset_name_alone(factory, dataset_name):
    """A bare CoIR dataset name routes to CoIRLoader without a benchmark hint."""
    loader = factory.create_dataloader(dataset_name=dataset_name, benchmark_name=None)
    assert isinstance(loader, CoIRLoader)


def test_coir_forwards_split_and_kwargs(factory):
    loader = factory.create_dataloader(
        dataset_name=Dataset.COSQA,
        benchmark_name=Benchmark.COIR,
        split=Split.TRAIN,
        max_queries=7,
        max_corpus_docs=99,
    )
    assert loader.split == Split.TRAIN
    assert loader.max_queries == 7
    assert loader.max_corpus_docs == 99


def test_coir_rejects_unknown_dataset(factory):
    with pytest.raises(ValueError, match="Unknown CoIR dataset"):
        factory.create_dataloader(
            dataset_name="not-a-coir-dataset", benchmark_name=Benchmark.COIR
        )


def test_bright_stackoverflow_not_shadowed_by_coir(factory):
    """'stackoverflow' (BRIGHT) must not be captured by 'stackoverflow-qa' (CoIR)."""
    loader = factory.create_dataloader(
        dataset_name=Dataset.STACKOVERFLOW, benchmark_name=Benchmark.BRIGHT
    )
    assert isinstance(loader, BRIGHTLoader)
