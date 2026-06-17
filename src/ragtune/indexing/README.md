# Indexing

This module builds and queries search indexes from any SIR data loader's corpus output. It supports sparse (BM25) and dense (embedding) indexing, with a unified API across backends so the rest of the pipeline doesn't need to know which one is in use.

## Quick start

```python
from ragtune.indexing import IndexFactory
from src.ragtune.data.loaders.RetrieverDataset import RetrieverDataset

# Load a corpus from any benchmark
rd = RetrieverDataset(dataset="biology", benchmark="BRIGHT", split="test")
_, _, corpus = rd.qrels()   # {doc_id: {"text": ..., "title": ...}}

# Build a sparse (BM25) index
indexer = IndexFactory.create("pyterrier")
indexer.build_from_corpus(corpus, index_path="indexes/biology-bm25")

# Query it
results = indexer.search("what is bm25?", top_k=5, index_path="indexes/biology-bm25")
for r in results:
    print(r.doc_id, r.score)
```

Or via the YAML-driven script — see [`scripts/run_indexing.py`](../../../scripts/run_indexing.py) and [`configs/bright_indexing.yaml`](configs/bright_indexing.yaml):

```bash
python scripts/run_indexing.py src/ragtune/indexing/configs/bright_indexing.yaml \
    --query "what is the role of mitochondria in apoptosis?" --top-k 5
```

If the index already exists at `index_path`, the corpus is **not** re-downloaded and the build step is skipped — `--query` alone is enough to test retrieval against a previously built index.

---

## The rule: sparse vs dense

- **`type: "sparse"`** → always `PyTerrierIndexer` (BM25). No encoder, no `model`/`params` needed.
- **`type: "dense"`** → you must pick a `backend` (`faiss` | `numpy` | `flex`) and a `model.name`. Everything about *how* it runs (`device`, `batch_size`, `max_length`, `use_fp16`, pooling, prefixes, ...) goes in `params`.

This is enforced in [`factory.py`](factory.py)'s `IndexFactory.from_config()`.

---

## Architecture

```
BaseIndexer (base.py)
 ├─ build_from_corpus(corpus, index_path)   primary API — BEIR-style dict in
 ├─ build(collection_path, format, fields)  file-based convenience wrapper
 ├─ exists(index_path)                      True if already built
 ├─ load(index_path)                        live index object for retrieval
 └─ search(query, top_k, index_path)        exact top_k hits → List[SearchResult]

 ├─ PyTerrierIndexer (pyterrier_indexer.py)      sparse BM25, python-terrier
 ├─ PyseriniIndexer  (pyserini_indexer.py)       sparse BM25, stub — not implemented
 └─ DenseIndexer (dense_indexer.py)              template: encode → batch → normalize
     ├─ FaissIndexer   storage: faiss.IndexFlatIP        (needs faiss-cpu)
     └─ NumpyIndexer   storage: plain .npy matrix         (no extra deps)
     FlexIndexer (flex_indexer.py)               storage: pyterrier_dr FlexIndex
                                                  (needs pyterrier-dr)
```

All four concrete indexers are **exact** (brute-force) search — no approximate-NN tradeoffs (HNSW/IVF/PQ are intentionally out of scope here).

### Why two dense storage backends (Faiss vs Numpy)?

| | `FaissIndexer` | `NumpyIndexer` |
|---|---|---|
| Storage | `faiss.IndexFlatIP` | raw `.npy` matrix |
| Dependency | `faiss-cpu` | none beyond numpy |
| Search | `faiss_index.search()` | `vectors @ query_vec` |
| When to use | larger corpora, already have faiss installed | small/medium corpora, minimal-dependency environments |

Both inherit nearly everything (encoding, batching, normalization, the `build_from_corpus`/`search` orchestration) from `DenseIndexer` — they only implement `_save_vectors()`, `exists()`, `load()`, `_search_vectors()`. Adding a third storage backend means implementing just those four hooks.

### Why a separate `FlexIndexer`?

`FlexIndexer` wraps [pyterrier_dr](https://github.com/Mandeep-Rathee/pyterrier_dr)'s `FlexIndex`, which gives you:
- A wider encoder zoo out of the box (BGE-M3 with simultaneous dense+sparse+ColBERT vectors, TasB, Ance, TctColBert)
- Swappable retrieval backend *independent of how it was indexed*: `get_retriever(index_path, backend="np"|"torch"|"faiss_hnsw"|"faiss_flat")`
- Native composition into PyTerrier pipelines via `>>` (e.g. chaining into a reranker)

It requires PyTerrier + pyterrier-dr; `FaissIndexer`/`NumpyIndexer` don't.

---

## Encoders (`encoders/`)

`FlexIndexer` needs an encoder model. Encoder resolution follows pyterrier_dr's own convention: **each model family gets its own pooling/instruction-prefix logic** — there's no generic one-size-fits-all encoder, because pooling strategy (CLS vs mean vs last-token) and instruction format are inherently model-specific.

```
encoders/
 ├─ __init__.py     resolve_encoder(model_name) — routing logic, see below
 ├─ qwen.py         Qwen3Encoder — last-token pooling, instruction prefix on
 │                   queries only, fp16, max_length=8192 default
 └─ generic_hf.py   GenericHFEncoder — configurable pooling/max_length/prefixes,
                     fallback for any HF model ID without a dedicated file
```

**`resolve_encoder()` routing order:**
1. Exact shorthand (`"bge-m3"`, `"tasb"`, `"ance"`, `"tct"`) → pyterrier_dr's own built-in classes (already correct for these families, reused as-is)
2. **Family substring match** — any `model_name` containing `"qwen3"` (case-insensitive) → our `Qwen3Encoder`, with the real checkpoint ID passed through. This means `"qwen3"`, `"Qwen/Qwen3-Embedding-0.6B"`, `-4B`, `-8B` all route correctly — not just the bare shorthand.
3. Anything else → `GenericHFEncoder` (pyterrier_dr's own `HgfBiEncoder` is too rigid — it can't take `max_length`, pooling choice, or instruction prefixes, which is what motivated writing this fallback).

Why this exists at all: pyterrier-dr on PyPI (0.7.0) has **no Qwen3 support**, and its generic HF encoder can't be configured for max_length/pooling/prefixes. Both gaps are filled here without reimplementing the families pyterrier_dr already gets right.

---

## Config schema

```yaml
index:
  data:                          # which corpus/queries to load
    benchmark: "BRIGHT"
    dataset: "biology"
    split: "test"
    long_context: false

  type: "dense"                  # "sparse" | "dense"
  backend: "flex"                # dense only: "faiss" | "numpy" | "flex"
  index_path: "indexes/biology-qwen3"

  model:
    name: "Qwen/Qwen3-Embedding-0.6B"   # WHICH model — identity only

  params:                        # HOW it runs — forwarded as-is to the
    device: "cuda"                # resolved encoder; valid keys differ per
    batch_size: 32                # backend/family (see encoders/ above)
    max_length: 8192
    use_fp16: true
```

`model`/`params` are deliberately split: `model.name` is just identity, `params` is everything about execution. `params` stays a free-form dict (`IndexConfig.params: Dict[str, Any]`) rather than enumerating every possible key on the Pydantic model, because valid keys genuinely differ per family — `Qwen3Encoder` takes `use_fp16`/`task_description`, `GenericHFEncoder` takes `pooling`/`query_prefix`/`doc_prefix`.

See [`configs/bright_indexing.yaml`](configs/bright_indexing.yaml) for all six backend options (sparse BM25, FAISS, NumPy, FlexIndex+Qwen3, FlexIndex+BGE-M3, FlexIndex+raw HF model), each fully commented.

---

## Adding a new indexer

1. Subclass `BaseIndexer` (or `DenseIndexer` if it's vector-based — you only need `_save_vectors`/`exists`/`load`/`_search_vectors`).
2. Decorate with `@registry.indexer("your-key")`.
3. Import the module in `__init__.py` so the decorator runs and registers it.
4. It's now usable via `IndexFactory.create("your-key", ...)` and from YAML via `backend: "your-key"`.

## Adding a new encoder family (for `FlexIndexer`)

1. Add a file in `encoders/` with a class subclassing `pyterrier_dr.BiEncoder`, implementing `encode_queries()`/`encode_docs()` with whatever pooling/prefix logic that family needs.
2. Register it in `encoders/__init__.py`'s `FAMILY_ENCODERS` (substring match) or `BUILTIN_ENCODERS` (exact match, for pyterrier_dr's own classes).

---

## Files

| File | Purpose |
|------|---------|
| `base.py` | `BaseIndexer` ABC, `SearchResult` |
| `factory.py` | `IndexFactory.create()` / `.from_config()` |
| `pyterrier_indexer.py` | `PyTerrierIndexer` — sparse BM25 |
| `pyserini_indexer.py` | `PyseriniIndexer` — stub, not yet implemented |
| `dense_indexer.py` | `DenseIndexer` template, `FaissIndexer`, `NumpyIndexer` |
| `flex_indexer.py` | `FlexIndexer` — pyterrier_dr FlexIndex |
| `encoders/` | `resolve_encoder()`, `Qwen3Encoder`, `GenericHFEncoder` |
| `configs/bright_indexing.yaml` | example config covering all backends |

Tests: `tests/unit/indexing/` (98 tests — heavy deps like faiss/pyterrier/transformers are mocked; one real end-to-end BM25 build+search has been run manually against the BRIGHT biology corpus).
