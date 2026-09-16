"""
Unit tests for the data-loading constants (src/ragtune/data/constants/constants.py).

Covers the CoIR additions: dataset names live on `Dataset`, are collected in
`COIR_DATASETS`, and the HuggingFace layout constants match CoIR's
non-standard config/split naming.
"""

from ragtune.data import constants
from ragtune.data.constants import (
    BRIGHT_TASKS,
    COIR_DATASETS,
    FRESHSTACK_TOPICS,
    Benchmark,
    Dataset,
    HFDatasets,
)


EXPECTED_COIR_DATASETS = [
    "stackoverflow-qa",
    "codefeedback-st",
    "apps",
    "cosqa",
    "synthetic-text2sql",
]


def test_coir_datasets_values():
    assert COIR_DATASETS == EXPECTED_COIR_DATASETS


def test_coir_datasets_built_from_dataset_constants():
    """The list must reference Dataset attributes, not bare string literals."""
    assert COIR_DATASETS == [
        Dataset.STACKOVERFLOW_QA,
        Dataset.CODEFEEDBACK_ST,
        Dataset.APPS,
        Dataset.COSQA,
        Dataset.SYNTHETIC_TEXT2SQL,
    ]


def test_coir_datasets_exported_from_package():
    assert "COIR_DATASETS" in constants.__all__
    assert constants.COIR_DATASETS is COIR_DATASETS


def test_coir_benchmark_registered():
    assert Benchmark.COIR == "coir"


def test_coir_hf_layout_constants():
    assert HFDatasets.COIR_ORG == "CoIR-Retrieval"
    # CoIR stores corpus/queries in a config *and* split of the same name.
    assert HFDatasets.COIR_CORPUS_CONFIG == HFDatasets.COIR_CORPUS_SPLIT == "corpus"
    assert HFDatasets.COIR_QUERIES_CONFIG == HFDatasets.COIR_QUERIES_SPLIT == "queries"


def test_dataset_name_lists_are_disjoint():
    """A dataset name must dispatch to exactly one loader in the factory."""
    assert not set(COIR_DATASETS) & set(BRIGHT_TASKS)
    assert not set(COIR_DATASETS) & set(FRESHSTACK_TOPICS)


def test_no_duplicate_coir_datasets():
    assert len(COIR_DATASETS) == len(set(COIR_DATASETS))
