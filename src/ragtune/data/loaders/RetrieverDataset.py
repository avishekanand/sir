"""
RetrieverDataset
================
High-level dataset class that wraps a data loader and exposes the
(queries, qrels, corpus) triple expected by retrieval evaluators.

This mirrors the DEXTER RetrieverDataset API so that SIR's evaluation
pipeline can consume it without modification.

Usage
-----
    from ragtune.data.loaders.RetrieverDataset import RetrieverDataset
    from ragtune.data.constants import Benchmark, Dataset, Split

    rd = RetrieverDataset(
        dataset=Dataset.BIOLOGY,
        benchmark=Benchmark.BRIGHT,
        split=Split.TEST,
    )
    queries, qrels, corpus = rd.qrels()
"""

import logging
from typing import Dict, List, Optional, Tuple

from ragtune.data.constants import Benchmark, Split
from ragtune.data.datastructures import Query, Context
from ragtune.data.loaders.DataLoaderFactory import DataLoaderFactory

logger = logging.getLogger(__name__)


class RetrieverDataset:
    """
    High-level dataset wrapper for retrieval evaluation.

    Parameters
    ----------
    dataset : str
        Dataset / task name (e.g. 'biology', 'langchain').
    benchmark : str
        Benchmark name (Benchmark.BRIGHT | Benchmark.FRESHSTACK | Benchmark.BEIR).
    split : str
        Data split (Split.TEST | Split.TRAIN | Split.DEV).
    long_context : bool
        BRIGHT-specific: use long-context gold labels.
    reasoning_subset : str | None
        BRIGHT-specific: load reasoning-augmented queries.
    cache_dir : str | None
        HuggingFace cache directory.
    extra_kwargs : dict
        Additional kwargs forwarded to the loader factory.
    """

    def __init__(
        self,
        dataset: str,
        benchmark: str,
        split: str = Split.TEST,
        long_context: bool = False,
        reasoning_subset: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **extra_kwargs,
    ):
        self.dataset = dataset
        self.benchmark = benchmark
        self.split = split

        self._loader = DataLoaderFactory().create_dataloader(
            dataset_name=dataset,
            benchmark_name=benchmark,
            split=split,
            long_context=long_context,
            reasoning_subset=reasoning_subset,
            cache_dir=cache_dir,
            **extra_kwargs,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def qrels(
        self,
    ) -> Tuple[List[Query], Dict[str, Dict[str, int]], Dict[str, Dict]]:
        """
        Returns (queries, qrels, corpus) in DEXTER / BEIR style.

        Returns
        -------
        queries : List[Query]
            Query objects with IDs.
        qrels : Dict[str, Dict[str, int]]
            {query_id: {doc_id: relevance_score}}
        corpus : Dict[str, Dict]
            {doc_id: {'text': ..., 'title': ...}}
        """
        queries = self._loader.get_query_objects()
        qrels = self._loader.get_qrels()
        corpus = self._loader.get_corpus()
        return queries, qrels, corpus

    def load(self):
        """Alias: returns (corpus, queries_dict, qrels) matching BEIR GenericDataLoader."""
        return self._loader.load()

    def get_loader(self):
        """Return the underlying BaseDataLoader."""
        return self._loader

    def __repr__(self):
        return (
            f"RetrieverDataset(benchmark={self.benchmark!r}, "
            f"dataset={self.dataset!r}, split={self.split!r})"
        )
