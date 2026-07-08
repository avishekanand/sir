"""
Unit tests for CRUMB benchmark wiring in DataLoaderFactory / constants.

Regression coverage for a gap where CRUMBLoader was implemented and unit
tested (test_crumb_loader.py) but never registered: Benchmark.CRUMB was
missing from constants.py and DataLoaderFactory had no CRUMB branch, so
benchmark="CRUMB" silently fell through to the ir_datasets fallback and
failed for any CRUMB task name (e.g. 'legal_qa').
"""

from unittest.mock import patch

from ragtune.data.constants import Benchmark, CRUMB_TASKS
from ragtune.data.loaders.CRUMBLoader import CRUMB_TASKS as LOADER_CRUMB_TASKS
from ragtune.data.loaders.DataLoaderFactory import DataLoaderFactory


def test_benchmark_crumb_constant_is_registered():
    assert Benchmark.CRUMB == "CRUMB"


def test_crumb_tasks_are_registered_in_constants():
    expected = [
        "paper_retrieval",
        "theorem_retrieval",
        "tip_of_the_tongue",
        "stack_exchange",
        "clinical_trial",
        "set_operation_entity_retrieval",
        "code_retrieval",
        "legal_qa",
    ]
    assert CRUMB_TASKS == expected


def test_crumb_loader_reexports_the_same_task_list():
    # CRUMBLoader must not keep its own copy of the task list -- constants.py
    # is the single source of truth, and CRUMBLoader re-exports it.
    assert LOADER_CRUMB_TASKS is CRUMB_TASKS


def test_factory_dispatches_crumb_to_crumb_loader():
    with patch("ragtune.data.loaders.CRUMBLoader.CRUMBLoader") as MockCRUMBLoader:
        DataLoaderFactory().create_dataloader(
            dataset_name="legal_qa",
            benchmark_name="CRUMB",
            split="test",
            cache_dir="/tmp/cache",
        )

    MockCRUMBLoader.assert_called_once_with(
        task="legal_qa",
        split="test",
        cache_dir="/tmp/cache",
    )


def test_factory_dispatch_is_case_insensitive():
    with patch("ragtune.data.loaders.CRUMBLoader.CRUMBLoader") as MockCRUMBLoader:
        DataLoaderFactory().create_dataloader(
            dataset_name="legal_qa",
            benchmark_name="crumb",
            split="test",
        )

    MockCRUMBLoader.assert_called_once()


def test_factory_requires_explicit_benchmark_for_crumb_task_names():
    # CRUMB task names (e.g. "legal_qa") are generic enough that another
    # benchmark could reuse one, so dispatch must key off benchmark_name
    # rather than matching dataset_name against CRUMB_TASKS alone.
    with patch("ragtune.data.loaders.CRUMBLoader.CRUMBLoader") as MockCRUMBLoader, \
         patch("ragtune.data.loaders.IRDatasetsLoader.IRDatasetsLoader") as MockIRDatasetsLoader:
        DataLoaderFactory().create_dataloader(
            dataset_name="legal_qa",
            benchmark_name="",
            split="test",
        )

    MockCRUMBLoader.assert_not_called()
    MockIRDatasetsLoader.assert_called_once()


def test_factory_does_not_fall_back_to_ir_datasets_for_crumb(caplog):
    with patch("ragtune.data.loaders.CRUMBLoader.CRUMBLoader"), \
         patch("ragtune.data.loaders.IRDatasetsLoader.IRDatasetsLoader") as MockIRDatasetsLoader:
        DataLoaderFactory().create_dataloader(
            dataset_name="legal_qa",
            benchmark_name="CRUMB",
            split="test",
        )

    MockIRDatasetsLoader.assert_not_called()
    assert not any("Unknown benchmark" in r.message for r in caplog.records)
