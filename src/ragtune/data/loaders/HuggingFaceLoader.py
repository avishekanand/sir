"""
Generic HuggingFace Data Loader
================================
Loads any HuggingFace dataset that follows the BEIR-style schema:
    corpus split   : id, text, title
    queries split  : id, text
    qrels  split   : query-id, corpus-id, score

Also exposes reusable class-level helpers used by BRIGHTLoader and
FreshStackLoader so that HF access logic is never duplicated:

    HuggingFaceLoader.fetch_hf_split(repo, config, split, cache_dir)
        → HuggingFace Dataset object (or raises ImportError / RuntimeError)

    HuggingFaceLoader.populate_corpus(corpus_dict, rows, id_col, text_col, title_col)
        → populates a {doc_id: {text, title}} dict in-place

    HuggingFaceLoader.populate_queries(queries_dict, query_objs, rows,
                                        id_col, text_col)
        → populates queries + Query objects in-place

    HuggingFaceLoader.populate_qrels(qrels_dict, rows,
                                      qid_col, did_col, score_col)
        → populates qrels in-place

    HuggingFaceLoader.build_raw_data(raw_data_list, query_objs,
                                      qrels_dict, corpus_dict,
                                      min_relevance)
        → appends Sample objects to raw_data_list
"""

import logging
from typing import Any, Dict, List, Optional

from ragtune.data.loaders.BaseDataLoader import BaseDataLoader
from ragtune.data.datastructures.query import Query
from ragtune.data.datastructures.context import Context
from ragtune.data.datastructures.sample import Sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers – callable from any loader without instantiation
# ---------------------------------------------------------------------------

def fetch_hf_split(
    repo: str,
    config: Optional[str],
    split: str,
    cache_dir: Optional[str] = None,
) -> Any:
    """
    Load a single HuggingFace dataset split and return the Dataset object.

    Parameters
    ----------
    repo : str
        HuggingFace dataset repo id, e.g. ``"xlangai/BRIGHT"``.
    config : str | None
        Dataset config / subset name, e.g. ``"examples"``.
        Pass ``None`` to load the default config.
    split : str
        Split name, e.g. ``"test"`` or ``"biology"``.
    cache_dir : str | None
        Optional HuggingFace cache directory.

    Returns
    -------
    datasets.Dataset
        The loaded split.

    Raises
    ------
    ImportError
        If the ``datasets`` package is not installed.
    RuntimeError
        If the split cannot be loaded for any other reason.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. "
            "Install with: pip install datasets"
        )
    kwargs: Dict = {}
    if config:
        kwargs["name"] = config
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    try:
        return load_dataset(repo, split=split, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load HuggingFace split "
            f"repo={repo!r} config={config!r} split={split!r}: {exc}"
        ) from exc


def populate_corpus(
    corpus: Dict[str, Dict],
    rows,
    id_col: str = "id",
    text_col: str = "text",
    title_col: str = "title",
) -> None:
    """
    Iterate over HF dataset rows and populate a corpus dict in-place.

    Each entry: ``corpus[doc_id] = {"text": ..., "title": ...}``
    Rows that produce an empty doc_id are silently skipped.
    """
    for row in rows:
        doc_id = str(row.get(id_col, row.get("_id", row.get("id", ""))))
        if not doc_id:
            continue
        corpus[doc_id] = {
            "text":  row.get(text_col,  row.get("text",    "")),
            "title": row.get(title_col, row.get("title",   "")),
        }


def populate_queries(
    queries: Dict[str, str],
    query_objs: Dict[str, Query],
    rows,
    id_col: str = "id",
    text_col: str = "text",
) -> None:
    """
    Iterate over HF dataset rows and populate queries + Query objects in-place.
    """
    for row in rows:
        qid  = str(row[id_col])
        text = str(row.get(text_col, ""))
        queries[qid]    = text
        query_objs[qid] = Query(text=text, idx=qid)


def populate_qrels(
    qrels: Dict[str, Dict[str, int]],
    rows,
    qid_col: str = "query-id",
    did_col: str = "corpus-id",
    score_col: str = "score",
) -> None:
    """
    Iterate over HF qrels rows and populate a qrels dict in-place.
    """
    for row in rows:
        qid   = str(row[qid_col])
        did   = str(row[did_col])
        score = int(row.get(score_col, 1))
        qrels.setdefault(qid, {})[did] = score


def build_raw_data(
    raw_data: List[Sample],
    query_objs: Dict[str, Query],
    qrels: Dict[str, Dict[str, int]],
    corpus: Dict[str, Dict],
    min_relevance: int = 1,
) -> None:
    """
    Build Sample objects from already-populated query_objs / qrels / corpus
    and append them to ``raw_data``.

    A document is considered a gold evidence if its relevance score is
    >= ``min_relevance`` (default 1).
    """
    for qid, query_obj in query_objs.items():
        gold_docs = [
            did for did, rel in qrels.get(qid, {}).items()
            if rel >= min_relevance
        ]
        if gold_docs:
            for doc_id in gold_docs:
                doc_info = corpus.get(doc_id, {})
                raw_data.append(Sample(
                    idx      = qid,
                    query    = query_obj,
                    evidences= Context(
                        text  = doc_info.get("text",  ""),
                        idx   = doc_id,
                        title = doc_info.get("title", ""),
                    ),
                ))
        else:
            raw_data.append(Sample(idx=qid, query=query_obj, evidences=None))


# ---------------------------------------------------------------------------
# HuggingFaceLoader – BaseDataLoader subclass for BEIR-schema HF datasets
# ---------------------------------------------------------------------------

class HuggingFaceLoader(BaseDataLoader):
    """
    Generic loader for HuggingFace datasets that follow the BEIR schema:

    * corpus  split: id, text, title
    * queries split: id, text
    * qrels   split: query-id, corpus-id, score

    All column names are configurable.  Used directly for BEIR-mirror datasets
    (e.g. ``mteb/<beir_dataset>``) and as the building-block for benchmark-
    specific loaders (BRIGHTLoader, FreshStackLoader) which call the module-level
    helpers above instead of repeating the HF access logic.

    Parameters
    ----------
    hf_dataset_name : str
        HuggingFace repo id, e.g. ``'mteb/BrightRetrieval'``.
    subset : str | None
        Dataset config name passed as ``name=`` to ``load_dataset``.
    corpus_split, queries_split, qrels_split : str
        HF split names for each part of the dataset.
    id_col, text_col, title_col : str
        Corpus column names.
    query_id_col, query_text_col : str
        Queries column names.
    qrel_qid_col, qrel_did_col, qrel_score_col : str
        Qrels column names.
    split : str
        Logical evaluation split (stored on the loader, not used for HF calls
        directly since each part has its own split parameter above).
    cache_dir : str | None
        HuggingFace cache directory.
    """

    def __init__(
        self,
        hf_dataset_name: str,
        subset: Optional[str] = None,
        corpus_split: str = "corpus",
        queries_split: str = "queries",
        qrels_split: str = "qrels",
        id_col: str = "id",
        text_col: str = "text",
        title_col: str = "title",
        query_id_col: str = "id",
        query_text_col: str = "text",
        qrel_qid_col: str = "query-id",
        qrel_did_col: str = "corpus-id",
        qrel_score_col: str = "score",
        split: str = "test",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(dataset=hf_dataset_name, split=split)
        self.hf_dataset_name = hf_dataset_name
        self.subset          = subset
        self.corpus_split    = corpus_split
        self.queries_split   = queries_split
        self.qrels_split     = qrels_split
        self.id_col          = id_col
        self.text_col        = text_col
        self.title_col       = title_col
        self.query_id_col    = query_id_col
        self.query_text_col  = query_text_col
        self.qrel_qid_col    = qrel_qid_col
        self.qrel_did_col    = qrel_did_col
        self.qrel_score_col  = qrel_score_col
        self.cache_dir       = cache_dir

    def _load_data(self) -> None:
        logger.info(
            f"[HuggingFaceLoader] Loading {self.hf_dataset_name!r} "
            f"subset={self.subset!r}"
        )

        # ---- Corpus ----
        corpus_rows = fetch_hf_split(
            self.hf_dataset_name, self.subset,
            self.corpus_split, self.cache_dir,
        )
        populate_corpus(
            self._corpus, corpus_rows,
            id_col=self.id_col,
            text_col=self.text_col,
            title_col=self.title_col,
        )

        # ---- Queries ----
        query_objs: Dict[str, Query] = {}
        queries_rows = fetch_hf_split(
            self.hf_dataset_name, self.subset,
            self.queries_split, self.cache_dir,
        )
        populate_queries(
            self._queries, query_objs, queries_rows,
            id_col=self.query_id_col,
            text_col=self.query_text_col,
        )

        # ---- Qrels ----
        qrels_rows = fetch_hf_split(
            self.hf_dataset_name, self.subset,
            self.qrels_split, self.cache_dir,
        )
        populate_qrels(
            self._qrels, qrels_rows,
            qid_col=self.qrel_qid_col,
            did_col=self.qrel_did_col,
            score_col=self.qrel_score_col,
        )

        # ---- Samples ----
        build_raw_data(self.raw_data, query_objs, self._qrels, self._corpus)

        logger.info(
            f"[HuggingFaceLoader] {len(self._queries)} queries, "
            f"{len(self._corpus)} docs"
        )
