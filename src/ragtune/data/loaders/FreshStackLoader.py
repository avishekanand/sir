"""
FreshStack Data Loader
======================
Loads FreshStack benchmark datasets.

Source selection
----------------
1. HuggingFace ``datasets`` — via ``fetch_hf_split`` from ``HuggingFaceLoader``
   and ``populate_corpus`` / ``build_raw_data`` helpers.
2. ``freshstack`` PyPI package — official BEIR-format wrapper.
3. ``ir_datasets``             — via ``load_from_irds`` from ``IRDatasetsLoader``.

No HF or ir_datasets I/O code is duplicated here.

Real HuggingFace schema
-----------------------
Corpus  (freshstack/corpus-oct-2024, config=<topic>, split="train")
    _id       : str   ← document id  (note underscore prefix)
    text      : str
    title     : str
    url       : str

Queries  (freshstack/queries-oct-2024, config=<topic>, split="test")
    query_id    : str
    query_title : str
    query_text  : str
    nuggets     : List[dict], each:
        _id                     : str
        relevant_corpus_ids     : List[str]   relevance = 1
        non_relevant_corpus_ids : List[str]   relevance = 0

qrels are derived inline from the nuggets list; there is no separate qrels split.

Reference: https://github.com/fresh-stack/freshstack
           https://huggingface.co/datasets/freshstack/corpus-oct-2024
           https://huggingface.co/datasets/freshstack/queries-oct-2024
"""

import logging
from typing import Dict, List, Optional, Tuple

from src.ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from src.ragtune.data.loaders.HuggingFaceLoader import fetch_hf_split, populate_corpus, build_raw_data
from src.ragtune.data.loaders.IRDatasetsLoader import load_from_irds
from src.ragtune.data.datastructures.query import Query
from src.ragtune.data.constants import FRESHSTACK_TOPICS, HFDatasets, Split

logger = logging.getLogger(__name__)


class FreshStackLoader(BaseDataLoader):
    """
    Loads a single FreshStack topic.

    Source selection order
    ----------------------
    1. HuggingFace ``datasets``   — tried first.
    2. ``freshstack`` PyPI package — fallback.
    3. ``ir_datasets``             — last resort.

    Parameters
    ----------
    topic : str
        One of ``'langchain'``, ``'yolo'``, ``'angular'``, ``'laravel'``,
        ``'godot'``.
    split : str
        Logical evaluation split (default ``'test'``).  The corpus always
        uses the HF ``'train'`` split; only the queries respect this value.
    cache_dir : str | None
        Optional HuggingFace / ir_datasets cache directory.
    """

    def __init__(
        self,
        topic: str,
        split: str = Split.TEST,
        cache_dir: Optional[str] = None,
    ):
        if topic not in FRESHSTACK_TOPICS:
            raise ValueError(
                f"Unknown FreshStack topic: {topic!r}. "
                f"Valid topics: {FRESHSTACK_TOPICS}"
            )
        super().__init__(dataset=topic, split=split)
        self.topic     = topic
        self.cache_dir = cache_dir
        # Nugget-level data for callers that need it
        self._nugget_qrels:     Dict[str, Dict[str, int]] = {}
        self._query_to_nuggets: Dict[str, List[str]]      = {}

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        logger.info(f"[FreshStackLoader] topic={self.topic!r}  split={self.split!r}")

        if self._try_load_via_hf():
            return
        logger.warning("[FreshStackLoader] HuggingFace failed; trying freshstack package.")
        if self._try_load_via_freshstack_package():
            return
        logger.warning("[FreshStackLoader] freshstack package unavailable; trying ir_datasets.")
        if self._try_load_via_ir_datasets():
            return

        raise RuntimeError(
            f"[FreshStackLoader] All sources failed for topic={self.topic!r}. "
            "Install 'datasets', 'freshstack', or 'ir-datasets'."
        )

    # ------------------------------------------------------------------
    # Source 1: HuggingFace — uses fetch_hf_split / populate_corpus /
    #           build_raw_data from HuggingFaceLoader
    # ------------------------------------------------------------------

    def _try_load_via_hf(self) -> bool:
        try:
            # ---- Corpus ----
            # FreshStack corpus lives in the HF "train" split for all topics
            corpus_rows = fetch_hf_split(
                HFDatasets.FRESHSTACK_CORPUS,
                self.topic,
                HFDatasets.FRESHSTACK_CORPUS_SPLIT,   # "train"
                self.cache_dir,
            )
            # FreshStack uses "_id" not "id"; populate_corpus checks both
            populate_corpus(
                self._corpus, corpus_rows,
                id_col="_id",      # real column name
                text_col="text",
                title_col="title",
            )
            logger.info(f"[FreshStackLoader/HF] Corpus: {len(self._corpus)} docs")

            # ---- Queries + inline qrels ----
            queries_rows = fetch_hf_split(
                HFDatasets.FRESHSTACK_QUERIES,
                self.topic,
                self.split,
                self.cache_dir,
            )
            query_objs: Dict[str, Query] = {}

            for row in queries_rows:
                qid   = str(row["query_id"])
                title = row.get("query_title", "")
                body  = row.get("query_text",  row.get("text", ""))
                text  = (title + " " + body).strip() if title else body

                self._queries[qid] = text
                query_objs[qid]    = Query(text=text, idx=qid)

                # Derive qrels from per-query nugget list
                nugget_ids: List[str] = []
                self._qrels.setdefault(qid, {})

                for nugget in row.get("nuggets", []):
                    nid = str(nugget.get("_id", nugget.get("id", "")))
                    if nid:
                        nugget_ids.append(nid)

                    for did in nugget.get("relevant_corpus_ids", []):
                        did_str = str(did)
                        self._qrels[qid][did_str] = 1
                        if nid:
                            self._nugget_qrels.setdefault(nid, {})[did_str] = 1

                    for did in nugget.get("non_relevant_corpus_ids", []):
                        did_str = str(did)
                        if did_str not in self._qrels[qid]:
                            self._qrels[qid][did_str] = 0
                        if nid:
                            self._nugget_qrels.setdefault(nid, {})[did_str] = 0

                if nugget_ids:
                    self._query_to_nuggets[qid] = nugget_ids

            build_raw_data(self.raw_data, query_objs, self._qrels, self._corpus)
            logger.info(
                f"[FreshStackLoader/HF] Queries: {len(self._queries)}  "
                f"Qrels: {sum(len(v) for v in self._qrels.values())} pairs"
            )
            return True

        except ImportError as exc:
            logger.debug(f"[FreshStackLoader/HF] datasets not installed: {exc}")
            return False
        except Exception as exc:
            import traceback as _tb
            logger.warning(
                f"[FreshStackLoader/HF] Failed: {exc}\n" + _tb.format_exc()
            )
            self._corpus.clear(); self._queries.clear()
            self._qrels.clear();  self.raw_data.clear()
            self._nugget_qrels.clear(); self._query_to_nuggets.clear()
            return False

    # ------------------------------------------------------------------
    # Source 2: freshstack PyPI package
    # ------------------------------------------------------------------

    def _try_load_via_freshstack_package(self) -> bool:
        try:
            from freshstack.datasets import DataLoader as _FSLoader
        except ImportError:
            logger.debug("[FreshStackLoader/pkg] 'freshstack' not installed.")
            return False

        try:
            fs = _FSLoader(
                queries_repo=HFDatasets.FRESHSTACK_QUERIES,
                corpus_repo =HFDatasets.FRESHSTACK_CORPUS,
                topic       =self.topic,
            )
            corpus, queries, _ = fs.load(split=self.split)
            nugget_qrels, qrels_query, query_to_nuggets = fs.load_qrels(split=self.split)

            for doc_id, doc in corpus.items():
                self._corpus[str(doc_id)] = {
                    "text":  doc.get("text",  ""),
                    "title": doc.get("title", ""),
                }
            query_objs: Dict[str, Query] = {}
            for qid, qtext in queries.items():
                qid_str = str(qid)
                self._queries[qid_str] = qtext
                query_objs[qid_str]    = Query(text=qtext, idx=qid_str)

            for qid, rel_docs in qrels_query.items():
                self._qrels[str(qid)] = {str(d): int(r) for d, r in rel_docs.items()}

            self._nugget_qrels = {
                str(nid): {str(d): int(r) for d, r in rels.items()}
                for nid, rels in nugget_qrels.items()
            }
            self._query_to_nuggets = {
                str(qid): [str(n) for n in nids]
                for qid, nids in query_to_nuggets.items()
            }

            build_raw_data(self.raw_data, query_objs, self._qrels, self._corpus)
            logger.info(
                f"[FreshStackLoader/pkg] Corpus: {len(self._corpus)}  "
                f"Queries: {len(self._queries)}"
            )
            return True

        except Exception as exc:
            logger.warning(f"[FreshStackLoader/pkg] Failed: {exc}")
            self._corpus.clear(); self._queries.clear()
            self._qrels.clear();  self.raw_data.clear()
            return False

    # ------------------------------------------------------------------
    # Source 3: ir_datasets — uses load_from_irds from IRDatasetsLoader
    # ------------------------------------------------------------------

    def _try_load_via_ir_datasets(self) -> bool:
        irds_id    = f"freshstack/{self.topic}/{self.split}"
        query_objs: Dict[str, Query] = {}
        return load_from_irds(
            dataset_id = irds_id,
            corpus     = self._corpus,
            queries    = self._queries,
            query_objs = query_objs,
            qrels      = self._qrels,
            raw_data   = self.raw_data,
            cache_dir  = self.cache_dir,
        )

    # ------------------------------------------------------------------
    # Public extra: nugget-level qrels
    # ------------------------------------------------------------------

    def load_nugget_qrels(
        self,
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Dict[str, List[str]]]:
        """
        Return nugget-level evaluation data.

        Returns
        -------
        nugget_qrels : ``{nugget_id: {doc_id: relevance}}``
        qrels_query  : ``{query_id:  {doc_id: relevance}}``  (= self._qrels)
        query_to_nuggets : ``{query_id: [nugget_id, ...]}``
        """
        self._ensure_loaded()
        return self._nugget_qrels, self._qrels, self._query_to_nuggets
