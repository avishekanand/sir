"""
IRDatasets Loader
=================
Wraps the ``ir_datasets`` package for datasets available through it.

Also exposes a reusable module-level helper used by BRIGHTLoader and
FreshStackLoader so that ir_datasets access logic is never duplicated:

    IRDatasetsLoader.load_from_irds(dataset_id, corpus, queries, query_objs,
                                     qrels, raw_data, cache_dir)
        → bool  (True on success, False if ir_datasets unavailable / id unknown)

See: https://ir-datasets.com
"""

import logging
import os
from typing import Dict, List, Optional

from ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from ragtune.data.datastructures.query import Query
from ragtune.data.datastructures.context import Context
from ragtune.data.datastructures.sample import Sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helper – callable from any loader without instantiation
# ---------------------------------------------------------------------------

def load_from_irds(
    dataset_id: str,
    corpus: Dict[str, Dict],
    queries: Dict[str, str],
    query_objs: Dict[str, Query],
    qrels: Dict[str, Dict[str, int]],
    raw_data: List[Sample],
    cache_dir: Optional[str] = None,
) -> bool:
    """
    Load a dataset from ``ir_datasets`` and populate the provided dicts/lists
    in-place.  Returns ``True`` on success, ``False`` if the package is not
    installed or the dataset ID is not found.

    Parameters
    ----------
    dataset_id : str
        ir_datasets path, e.g. ``"beir/scifact/test"`` or
        ``"bright/biology/test"``.
    corpus : dict
        Target corpus dict ``{doc_id: {"text": ..., "title": ...}}``.
    queries : dict
        Target queries dict ``{query_id: query_text}``.
    query_objs : dict
        Target Query-objects dict ``{query_id: Query}``.
    qrels : dict
        Target qrels dict ``{query_id: {doc_id: relevance}}``.
    raw_data : list
        Target list for ``Sample`` objects.
    cache_dir : str | None
        Sets ``IR_DATASETS_HOME`` env var if provided.

    Returns
    -------
    bool
        ``True`` if data was loaded successfully.
    """
    try:
        import ir_datasets
    except ImportError:
        logger.debug("[IRDatasetsLoader] 'ir_datasets' not installed.")
        return False

    if cache_dir:
        os.environ.setdefault("IR_DATASETS_HOME", cache_dir)

    try:
        ds = ir_datasets.load(dataset_id)
    except Exception as exc:
        logger.warning(f"[IRDatasetsLoader] ir_datasets.load({dataset_id!r}) failed: {exc}")
        return False

    try:
        # ---- Corpus ----
        if ds.has_docs():
            for doc in ds.docs_iter():
                doc_id = str(doc.doc_id)
                text   = getattr(doc, "text",  getattr(doc, "body", ""))
                title  = getattr(doc, "title", "")
                corpus[doc_id] = {"text": text, "title": title}

        # ---- Queries ----
        if ds.has_queries():
            for q in ds.queries_iter():
                qid  = str(q.query_id)
                text = getattr(q, "text", "")
                queries[qid]    = text
                query_objs[qid] = Query(text=text, idx=qid)

        # ---- Qrels ----
        if ds.has_qrels():
            for qrel in ds.qrels_iter():
                qid = str(qrel.query_id)
                did = str(qrel.doc_id)
                rel = int(qrel.relevance)
                qrels.setdefault(qid, {})[did] = rel

        # ---- Samples ----
        for qid, query_obj in query_objs.items():
            gold = [d for d, r in qrels.get(qid, {}).items() if r > 0]
            if gold:
                for doc_id in gold:
                    info = corpus.get(doc_id, {})
                    raw_src.ragtune.data.append(Sample(
                        idx      = qid,
                        query    = query_obj,
                        evidences= Context(
                            text  = info.get("text",  ""),
                            idx   = doc_id,
                            title = info.get("title", ""),
                        ),
                    ))
            else:
                raw_src.ragtune.data.append(Sample(idx=qid, query=query_obj, evidences=None))

        logger.info(
            f"[IRDatasetsLoader] {dataset_id!r}: "
            f"{len(corpus)} docs, {len(queries)} queries"
        )
        return True

    except Exception as exc:
        logger.warning(f"[IRDatasetsLoader] Processing {dataset_id!r} failed: {exc}")
        # Clear partial state so callers can try the next source cleanly
        corpus.clear(); queries.clear(); query_objs.clear()
        qrels.clear();  raw_src.ragtune.data.clear()
        return False


# ---------------------------------------------------------------------------
# IRDatasetsLoader – BaseDataLoader subclass
# ---------------------------------------------------------------------------

class IRDatasetsLoader(BaseDataLoader):
    """
    Loads any dataset available via the ``ir_datasets`` package.

    Parameters
    ----------
    dataset_id : str
        The ir_datasets dataset ID, e.g. ``'beir/scifact/test'``.
    split : str
        Logical split label (informational; ir_datasets IDs embed the split).
    cache_dir : str | None
        Optional cache directory (sets ``IR_DATASETS_HOME``).
    """

    def __init__(
        self,
        dataset_id: str,
        split: str = "test",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(dataset=dataset_id, split=split)
        self.dataset_id = dataset_id
        self.cache_dir  = cache_dir

    def _load_data(self) -> None:
        query_objs: Dict[str, Query] = {}
        ok = load_from_irds(
            dataset_id = self.dataset_id,
            corpus     = self._corpus,
            queries    = self._queries,
            query_objs = query_objs,
            qrels      = self._qrels,
            raw_data   = self.raw_data,
            cache_dir  = self.cache_dir,
        )
        if not ok:
            raise RuntimeError(
                f"[IRDatasetsLoader] Could not load {self.dataset_id!r}. "
                "Is 'ir-datasets' installed and the dataset ID correct?"
            )
