# Spec: AutoIndex — learned representation programs for BM25

**Status:** Draft · **Branch:** `feat/autoindex` · **Module:** `src/ragtune/indexing/autoindex/`

**Reference:** O'Nuallain et al., *AutoIndex: Learning Representation Programs for Retrieval*,
arXiv:2607.18603 (2026). Reference implementation: https://github.com/auto-index/autoindex.

## 1. Problem

RAGtune currently tunes the *query side* of retrieval (reformulation, reranking, scheduling,
budgets) but always indexes documents exactly as the loader returns them. For long or
heterogeneous corpora (CRUMB: theorems, clinical trials, code, legal QA, …), how a document
is *represented* to BM25 (what text goes in, how it's chunked, what vocabulary is added
or removed) matters as much as any query-side component, and we have no way to tune it.

AutoIndex learns a **representation program** θ: executable Python that maps raw documents to
the chunks BM25 indexes. The retriever stays fixed. θ is searched by two LLM agents, with
validation Recall@100 as the objective. At serving time θ runs once, at index-build time, and
makes no LLM calls.

## 2. Algorithm (as in the paper)

```
θ₀ ← identity program (one chunk per document, raw text)
for t = 1..T:
    1. INDEX     run θ_{t-1} on corpus → chunks → BM25 (Lucene, k1=1.5, b=0.75), MaxP over chunks
    2. ANALYZE   Analysis Agent gets a stratified slice of validation queries:
                   • regressions vs θ₀  (hit@100 under θ₀, miss under θ_{t-1})
                   • misses with their top-ranked wrong docs (hard negatives)
                   • worst-ranked successes
                 and investigates them with read-only tools
                   bm25_retrieve(query, top_k) · read_file(file, filter_id, max_chars) · grep_search(pattern, file)
                 It must make ≥ min_tool_turns tool turns, then emits a <summary>.
    3. SYNTHESIZE Code Agent sees the summary + current θ + search history and proposes N candidate
                 programs (phase 1: N mechanically distinct ideas; phase 2: one program per idea, in parallel).
    4. SELECT    each candidate is validated (sample run + contract), executed under a timeout,
                 indexed, and scored on validation queries. "Proven" ⇔ ΔR@100 ≥ threshold.
                 If ≥ 2 proven: the LLM merges them; the merge is adopted only if it beats the best single candidate.
                 Otherwise the best proven candidate is adopted. If none is proven, θ is unchanged.
return the best-validation checkpoint θ*  (scored once on held-out queries by the caller)
```

Defaults (paper `config.yaml`): `T=5`, `N=4`, `threshold=1e-5`, `analysis_max_turns=5`,
`min_tool_turns=3`, `temperature=1.0`, `preprocess_timeout=900s`, candidate pool `k=1000` chunks
before MaxP, validation:evaluation query split `1:2`.

**Ablation switches** (paper §ablations): `use_analysis` (off: the Code Agent sees aggregate
metrics only) and `use_history` (off: no search history in prompts).

## 3. Design in RAGtune

| File | Responsibility |
|---|---|
| `program.py` | `Document`, `Chunk`, `RepresentationProgram` ABC, `IdentityProgram`; `load_program(code)`: AST safety screen + `exec`; `run_program()`: doc-id hashing, timeout, contract checks |
| `bm25.py` | `ChunkBM25Index`: tokenizer, Lucene BM25, MaxP doc aggregation. Uses `bm25s` if installed, otherwise a pure-Python inverted index (same scores) |
| `metrics.py` | Fractional Recall@{10,100} and binary nDCG@10, plus per-query hit/rank/retrieved |
| `tools.py` | The three Analysis-Agent tools over the in-memory corpus and *validation* queries only |
| `history.py` | `SearchHistory`: per-iteration scores and per-candidate outcomes, rendered for prompts |
| `llm.py` | `LLMClient` protocol + `LiteLLMClient`; token-usage accounting per agent |
| `prompts.py` | System and user prompts (adapted from the reference repo) |
| `agents.py` | `AnalysisAgent`, `CodeAgent` |
| `loop.py` | `AutoIndexConfig`, `AutoIndex.fit()`, `AutoIndexResult` |
| `indexer.py` | `@registry.indexer("autoindex")`: applies a learned program at build time, `search()` via `ChunkBM25Index` |
| `examples/autoindex_crumb.py` | End-to-end CRUMB run: val/eval split, fit, held-out baseline vs learned |

Data uses RAGtune's loader conventions: `corpus: {doc_id: {"text", "title"}}`,
`queries: {qid: text}`, `qrels: {qid: {doc_id: rel}}`. A non-empty `title` goes into
`Document.metadata["title"]`.

### Generated-program contract

```python
from typing import List
from ragtune.indexing.autoindex.program import Document, Chunk, RepresentationProgram

class Preprocessor(RepresentationProgram):
    name = "..."
    def preprocess(self, docs: List[Document]) -> List[Chunk]: ...
```

- ≥ 1 chunk per document, every `chunk.doc_id` must be an input `doc_id`, and `chunk_id`s must be unique.
- Programs see doc IDs as opaque hashes (`doc_<sha256[:16]>`) and are mapped back afterwards,
  so IDs can't leak relevance signal (same as the paper).

### Safety

LLM-written code runs in-process, as in the reference implementation. Before `exec`,
`load_program` does an AST screen that rejects imports of process, network, and filesystem
modules (`os`, `subprocess`, `socket`, `shutil`, `pathlib`, `requests`, `urllib`, `http`, …),
calls to `open`/`exec`/`eval`/`compile`/`__import__`, and dunder attribute access. This is a
guardrail, **not a sandbox**. Run AutoIndex in a container for untrusted models.
Execution is bounded by `preprocess_timeout_s`. A thread can't be killed, so a program that
times out is abandoned, not stopped.

### Data hygiene

- Agents only ever see validation queries. Held-out queries are never passed to `fit()`.
- Adoption and checkpoint selection use validation metrics only. The reference repo's loop
  compares held-out scores across iterations when choosing a global best; we deliberately
  don't, which matches the paper text ("best validation checkpoint").

## 4. Out of scope

- Retrievers other than BM25 (the paper fixes the retriever).
- Persisting the BM25 matrix; `indexer.load()` rebuilds BM25 from the stored chunks.
- A `ragtune` CLI subcommand (follow-up; the example script covers end-to-end use).

## 5. Testing

Unit tests with fake LLM clients, no network:

- `test_program.py`: contract violations, safety screen, doc-id hashing round trip, timeout.
- `test_bm25.py`: hand-computed Lucene BM25 scores, MaxP aggregation, `bm25s` parity (skipped if not installed).
- `test_metrics.py`: recall/nDCG on hand cases.
- `test_tools.py`: held-out queries unreachable, `filter_id` lookup, grep caps.
- `test_history.py`: prompt rendering.
- `test_agents.py`: tool loop + min-tool-turn nudge + forced summary; idea/code parsing.
- `test_loop.py`: adoption on improvement, no adoption on regression, synthesis accept/reject,
  erroring candidates, `use_analysis=False`, **budget edge case**: `n_iterations=0` returns θ₀.
- `test_autoindex_indexer.py`: build → persist → load → search round trip.
