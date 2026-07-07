"""
DataLoader Factory
==================
Maps (benchmark, dataset) pairs to the appropriate loader class.

This is the single registration point – adding a new benchmark means
adding one branch here and implementing a BaseDataLoader subclass.
"""

import logging
from typing import Optional

from ragtune.data.constants import Benchmark, Dataset, BRIGHT_TASKS, FRESHSTACK_TOPICS, Split

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """Factory that creates the correct BaseDataLoader for a given benchmark/dataset."""

    def create_dataloader(
        self,
        dataset_name: str,
        benchmark_name: str,
        split: str = Split.TEST,
        long_context: bool = False,
        reasoning_subset: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Instantiate and return the appropriate data loader.

        Parameters
        ----------
        dataset_name : str
            Task / dataset name, e.g. 'biology', 'langchain', 'beir/scifact'.
        benchmark_name : str
            One of Benchmark.BRIGHT, Benchmark.FRESHSTACK, Benchmark.BEIR.
        split : str
            Data split.
        long_context : bool
            BRIGHT-specific long-context setting.
        reasoning_subset : str | None
            BRIGHT-specific reasoning-augmented query subset.
        cache_dir : str | None
            HuggingFace cache directory.
        **kwargs :
            Additional kwargs forwarded to the loader.

        Returns
        -------
        BaseDataLoader subclass instance.
        """

        benchmark_name = benchmark_name.upper() if benchmark_name else benchmark_name

        # ---- BRIGHT ----
        if benchmark_name == Benchmark.BRIGHT or dataset_name in BRIGHT_TASKS:
            from ragtune.data.loaders.BRIGHTLoader import BRIGHTLoader
            task = dataset_name
            logger.info(f"[Factory] Creating BRIGHTLoader(task={task!r})")
            return BRIGHTLoader(
                task=task,
                split=split,
                long_context=long_context,
                reasoning_subset=reasoning_subset,
                cache_dir=cache_dir,
            )

        # ---- FreshStack ----
        if benchmark_name == Benchmark.FRESHSTACK.upper() or dataset_name in FRESHSTACK_TOPICS:
            from ragtune.data.loaders.FreshStackLoader import FreshStackLoader
            logger.info(f"[Factory] Creating FreshStackLoader(topic={dataset_name!r})")
            return FreshStackLoader(
                topic=dataset_name,
                split=split,
                cache_dir=cache_dir,
            )

        # ---- BEIR via HuggingFace (mteb mirror) ----
        if benchmark_name == Benchmark.BEIR.upper():
            hf_name = kwargs.pop("hf_dataset_name", f"mteb/{dataset_name}")
            subset = kwargs.pop("subset", None)
            from ragtune.data.loaders.HuggingFaceLoader import HuggingFaceLoader
            logger.info(
                f"[Factory] Creating HuggingFaceLoader for BEIR "
                f"hf={hf_name!r} subset={subset!r}"
            )
            return HuggingFaceLoader(
                hf_dataset_name=hf_name,
                subset=subset,
                split=split,
                cache_dir=cache_dir,
                **kwargs,
            )

        # ---- ir_datasets fallback ----
        # dataset_name is expected to be a full ir_datasets path,
        # e.g. 'beir/scifact/test'
        logger.warning(
            f"[Factory] Unknown benchmark={benchmark_name!r} for dataset={dataset_name!r}. "
            "Attempting IRDatasetsLoader as fallback."
        )
        from ragtune.data.loaders.IRDatasetsLoader import IRDatasetsLoader
        return IRDatasetsLoader(
            dataset_id=dataset_name,
            split=split,
            cache_dir=kwargs.get("cache_dir", cache_dir),
        )
