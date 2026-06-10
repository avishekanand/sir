"""
BRIGHT Data Loader
==================
Loads BRIGHT (Reasoning-Intensive Retrieval) benchmark.

Source selection
----------------
1. HuggingFace ``datasets`` — via the shared ``fetch_hf_split`` /
   ``populate_corpus`` / ``populate_queries`` / ``build_raw_data``
   helpers in ``HuggingFaceLoader``.
2. ``ir_datasets``           — via the shared ``load_from_irds`` helper in
   ``IRDatasetsLoader``.

No HF or ir_datasets access code is duplicated here; all raw I/O
is delegated to those helpers.

HuggingFace schema
------------------
documents[task]:
    id             : str
    content        : str   (document text; alias "text" also accepted)

examples[task]:
    id             : str
    query          : str
    reasoning      : str   (LLM chain-of-thought, optional)
    excluded_ids   : List[str]
    gold_ids       : List[str]
    gold_ids_long  : List[str]
    gold_answer    : str

Reference: https://huggingface.co/datasets/xlangai/BRIGHT
"""

import logging
from typing import Dict, List, Optional

from src.ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from src.ragtune.data.loaders.HuggingFaceLoader import fetch_hf_split, populate_corpus, build_raw_data
from src.ragtune.data.loaders.IRDatasetsLoader import load_from_irds
from src.ragtune.data.datastructures.query import Query
from src.ragtune.data.datastructures.context import Context
from src.ragtune.data.datastructures.sample import Sample
from src.ragtune.data.constants import BRIGHT_TASKS, HFDatasets, Split

logger = logging.getLogger(__name__)


class BRIGHTLoader(BaseDataLoader):
    """
    Loads a single BRIGHT task (e.g. ``'biology'``, ``'stackoverflow'``).

    Source selection order
    ----------------------
    1. HuggingFace ``datasets`` — tried first.
    2. ``ir_datasets``           — fallback for air-gapped environments.

    Parameters
    ----------
    task : str
        One of the 12 BRIGHT task names (see ``constants.BRIGHT_TASKS``).
    split : str
        Only ``'test'`` is currently available for BRIGHT (default).
    long_context : bool
        If ``True``, use ``gold_ids_long`` for qrels instead of ``gold_ids``.
    reasoning_subset : str | None
        HF subset name for reasoning-augmented queries
        (e.g. ``'gpt4_reason'``, ``'llama3-70b_reason'``).
    cache_dir : str | None
        HuggingFace / ir_datasets cache directory.
    """

    TASK_TO_HF_SPLIT: Dict[str, str] = {task: task for task in BRIGHT_TASKS}

    def __init__(
        self,
        task: str,
        split: str = Split.TEST,
        long_context: bool = False,
        reasoning_subset: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        if task not in BRIGHT_TASKS:
            raise ValueError(
                f"Unknown BRIGHT task: {task!r}. Valid tasks: {BRIGHT_TASKS}"
            )
        super().__init__(dataset=task, split=split)
        self.task             = task
        self.long_context     = long_context
        self.reasoning_subset = reasoning_subset
        self.cache_dir        = cache_dir

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        logger.info(
            f"[BRIGHTLoader] task={self.task!r}  "
            f"long_context={self.long_context}  "
            f"reasoning_subset={self.reasoning_subset!r}"
        )
        if self._try_load_via_hf():
            return
        logger.warning("[BRIGHTLoader] HuggingFace failed; trying ir_datasets.")
        if self._try_load_via_ir_datasets():
            return
        raise RuntimeError(
            f"[BRIGHTLoader] All sources failed for task={self.task!r}. "
            "Install 'datasets' or 'ir-datasets'."
        )

    # ------------------------------------------------------------------
    # Source 1: HuggingFace — uses fetch_hf_split / populate_corpus /
    #           build_raw_data from HuggingFaceLoader
    # ------------------------------------------------------------------

    def _try_load_via_hf(self) -> bool:
        hf_split = self.TASK_TO_HF_SPLIT[self.task]

        try:
            # ---- Corpus ----
            doc_subset = (
                HFDatasets.BRIGHT_SUBSET_LONG_DOCUMENTS
                if self.long_context
                else HFDatasets.BRIGHT_SUBSET_DOCUMENTS
            )
            docs_ds = fetch_hf_split(
                HFDatasets.BRIGHT_EXAMPLES, doc_subset,
                hf_split, self.cache_dir,
            )
        except RuntimeError:
            # theoremqa_questions / theoremqa_theorems share a document pool
            base = hf_split.split("_")[0] if "_" in hf_split else None
            if base is None:
                logger.warning(f"[BRIGHTLoader/HF] Could not load corpus for {self.task!r}")
                return False
            try:
                doc_subset = (
                    HFDatasets.BRIGHT_SUBSET_LONG_DOCUMENTS
                    if self.long_context
                    else HFDatasets.BRIGHT_SUBSET_DOCUMENTS
                )
                docs_ds = fetch_hf_split(
                    HFDatasets.BRIGHT_EXAMPLES, doc_subset,
                    base, self.cache_dir,
                )
            except (ImportError, RuntimeError) as exc:
                logger.warning(f"[BRIGHTLoader/HF] Corpus load failed: {exc}")
                return False
        except ImportError as exc:
            logger.debug(f"[BRIGHTLoader/HF] datasets not installed: {exc}")
            return False

        try:
            # populate_corpus handles "content" → "text" alias via text_col fallback
            populate_corpus(
                self._corpus, docs_ds,
                id_col="id",
                text_col="content",   # BRIGHT uses "content"; populate_corpus
                title_col="title",    # falls back to row.get("text","") if missing
            )
            logger.info(f"[BRIGHTLoader/HF] Corpus: {len(self._corpus)} docs")

            examples_subset = (
                self.reasoning_subset or HFDatasets.BRIGHT_SUBSET_EXAMPLES
            )
            examples_ds = fetch_hf_split(
                HFDatasets.BRIGHT_EXAMPLES, examples_subset,
                hf_split, self.cache_dir,
            )

            gold_key  = "gold_ids_long" if self.long_context else "gold_ids"
            query_objs: Dict[str, Query] = {}

            for example in examples_ds:
                qid       = str(example["id"])
                query_text = example["query"]
                reasoning  = example.get("reasoning", None)

                query_obj = Query(text=query_text, idx=qid, reasoning=reasoning)
                self._queries[qid] = query_text
                query_objs[qid]    = query_obj

                gold_doc_ids: List[str] = example.get(gold_key, [])
                self._qrels[qid] = {str(d): 1 for d in gold_doc_ids}

            build_raw_data(self.raw_data, query_objs, self._qrels, self._corpus)
            logger.info(
                f"[BRIGHTLoader/HF] Queries: {len(self._queries)}  "
                f"Qrels: {len(self._qrels)}"
            )
            return True

        except Exception as exc:
            import traceback as _tb
            logger.warning(
                f"[BRIGHTLoader/HF] Failed: {exc}\n" + _tb.format_exc()
            )
            self._corpus.clear(); self._queries.clear()
            self._qrels.clear();  self.raw_data.clear()
            return False

    # ------------------------------------------------------------------
    # Source 2: ir_datasets — uses load_from_irds from IRDatasetsLoader
    # ------------------------------------------------------------------

    def _try_load_via_ir_datasets(self) -> bool:
        # Try both possible ir_datasets paths for BRIGHT
        for irds_id in (
            f"bright/{self.task}/{self.split}",
            f"beir/bright/{self.task}/{self.split}",
        ):
            query_objs: Dict[str, Query] = {}
            ok = load_from_irds(
                dataset_id = irds_id,
                corpus     = self._corpus,
                queries    = self._queries,
                query_objs = query_objs,
                qrels      = self._qrels,
                raw_data   = self.raw_data,
                cache_dir  = self.cache_dir,
            )
            if ok:
                logger.info(f"[BRIGHTLoader/irds] Loaded via {irds_id!r}")
                return True
        return False

    # ------------------------------------------------------------------
    # BRIGHT-specific extra
    # ------------------------------------------------------------------

    def get_excluded_ids(self) -> Dict[str, List[str]]:
        """
        Return the excluded_ids per query (docs to never retrieve because
        they contain the answer verbatim).  Only available via HuggingFace.
        """
        hf_split      = self.TASK_TO_HF_SPLIT[self.task]
        subset        = self.reasoning_subset or HFDatasets.BRIGHT_SUBSET_EXAMPLES
        examples_ds   = fetch_hf_split(
            HFDatasets.BRIGHT_EXAMPLES, subset, hf_split, self.cache_dir
        )
        return {
            str(ex["id"]): [
                str(e) for e in ex.get("excluded_ids", []) if e != "N/A"
            ]
            for ex in examples_ds
        }


class BRIGHTMultiTaskLoader:
    """
    Loads multiple (or all) BRIGHT tasks at once.

    Usage
    -----
    loader = BRIGHTMultiTaskLoader(tasks=["biology", "economics"])
    for task, task_loader in loader.items():
        corpus, queries, qrels = task_loader.load()
    """

    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        split: str = Split.TEST,
        long_context: bool = False,
        reasoning_subset: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        tasks = tasks or BRIGHT_TASKS
        self._loaders: Dict[str, BRIGHTLoader] = {
            task: BRIGHTLoader(
                task=task,
                split=split,
                long_context=long_context,
                reasoning_subset=reasoning_subset,
                cache_dir=cache_dir,
            )
            for task in tasks
        }

    def items(self):
        return self._loaders.items()

    def keys(self):
        return self._loaders.keys()

    def __getitem__(self, task: str) -> BRIGHTLoader:
        return self._loaders[task]

    def __len__(self):
        return len(self._loaders)
