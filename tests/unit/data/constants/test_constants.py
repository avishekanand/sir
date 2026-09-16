"""
Unit tests for the data-loading constants (src/ragtune/data/constants/constants.py).

Covers the CoIR additions: dataset names live on `Dataset`, are collected in
`COIR_DATASETS`, and the HuggingFace layout constants match CoIR's
non-standard config/split naming.
"""

import pytest

from ragtune.data import constants
from ragtune.data.constants import (
    BRIGHT_TASKS,
    COIR_DATASETS,
    FRESHSTACK_TOPICS,
    SRA_BENCH_SUBSETS,
    TOOLRET_SUBSETS,
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


@pytest.mark.parametrize(
    "other_name, other",
    [
        ("BRIGHT_TASKS", BRIGHT_TASKS),
        ("FRESHSTACK_TOPICS", FRESHSTACK_TOPICS),
        ("TOOLRET_SUBSETS", TOOLRET_SUBSETS),
        ("SRA_BENCH_SUBSETS", SRA_BENCH_SUBSETS),
    ],
)
def test_coir_dataset_names_are_disjoint_from_other_benchmarks(other_name, other):
    """
    DataLoaderFactory dispatches on bare dataset names, so an overlap would
    silently route a CoIR dataset to whichever branch is checked first.
    """
    assert not set(COIR_DATASETS) & set(other), other_name


def test_no_duplicate_coir_datasets():
    assert len(COIR_DATASETS) == len(set(COIR_DATASETS))
