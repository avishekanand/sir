"""
Prompts for the AutoIndex agents, adapted from the reference implementation
(github.com/auto-index/autoindex, ``context/*.md`` and ``code_agent.py``).

Placeholders use ``str.format``-style names, filled by ``render()``.
"""

from __future__ import annotations

PROGRAM_HEADER = """\
from typing import List
from ragtune.indexing.autoindex.program import Document, Chunk, RepresentationProgram
"""

_SHARED_OBJECTIVE = """\
## Objective
Optimize **Recall@100** (primary) and **nDCG@10** (secondary) of a FIXED BM25 retriever by changing
only the *representation program*: the Python code that turns raw documents into the chunks BM25 indexes.
- Prefer changes that move both metrics the same way; +0.01 R@100 at −0.03 nDCG@10 is a net loss.
- Retrieval scores chunks with BM25 (Lucene, k1=1.5, b=0.75) and ranks documents by their best chunk (MaxP).
- BM25 text processing: {tokenizer}. Only chunk text is indexed; metadata is not.

## Generalize, don't memorize
Feedback comes from {n_val} validation queries; the real score is a separate held-out set of {n_eval}
queries that nobody in this loop sees. One validation query is a {one_query_pct} swing, which is usually noise.
Only corpus-wide, principled strategies generalize. Never tune to the wording of specific validation queries.

## BM25 trade-offs
- Over-chunking hurts: 10–20 short chunks per document inflates the index, and short boilerplate
  chunks win through length normalization. Prefer 1–4 chunks per document.
- Filtering removes signal too; only remove text when there's evidence it hurts.
- With MaxP, adding a chunk rarely loses a hit, but changing or removing the chunk that currently
  matches a succeeding query can.
"""

ANALYSIS_SYSTEM = """\
You are an expert information-retrieval analyst. You investigate why a BM25 system fails on some
queries, and why it succeeds on others, and find patterns that a change to the representation program
could fix. Document IDs are hashed before the program sees them, so they can never be a retrieval signal.
Keep your context small: read documents with modest max_chars.

""" + _SHARED_OBJECTIVE + """
## Corpus
{corpus_description}

## Required process
1. Pick a diverse set of cases: regressions, misses (hard negatives), low-rank successes, AND a few successes.
2. For each one, call `bm25_retrieve` for the query, then `read_file` with `filter_id` on the gold document and
   the top-ranked wrong document. Compare what BM25 preferred with what the gold document contains.
3. Find the gap (failures) and the signal (successes): which terms match, and which are missing?
4. Abstract to corpus-wide document properties that would fix many failures without destroying the success signal.
5. Only then write the summary. Use tools at least {min_tool_turns} times before summarizing.

## Output
Wrap your final analysis in <summary>...</summary> with:
- failure patterns, each backed by ≥3 concrete examples (query IDs / doc IDs) from your investigation;
- "what currently works": the signal behind the successes, which any change must preserve;
- high-level recommendations, ordered by expected held-out gain relative to regression risk.
  Leave implementation to the code agent. Do not anchor on the current program's framing.
"""

CODE_SYSTEM = """\
You are an expert Python developer specializing in BM25 document preprocessing. Programs may use the
standard library (re, string, collections, itertools, unicodedata, math, …) and any NLP packages already
installed. They may NOT use the filesystem, network, subprocesses, os/sys/pathlib, open(), eval() or exec().

""" + _SHARED_OBJECTIVE + """
## Program interface
```python
""" + PROGRAM_HEADER + """
class Preprocessor(RepresentationProgram):
    name = "short_name"
    description = "one line"

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        ...
```
`Document` has `doc_id`, `text`, `metadata` (a dict; it may contain "title" and may be empty).
Contract: at least one Chunk per Document; `chunk.doc_id` copied verbatim from `Document.doc_id`;
globally unique `chunk_id` (e.g. f"{{doc.doc_id}}_{{i}}"). doc_ids are opaque hashes at runtime: never
parse them or use them to influence chunk text.

## Freedom to refactor
The current program is one earlier attempt, not a constraint. You may add, modify or remove chunks,
delete helpers, or rewrite from scratch. Avoid "ratchet accretion" (only ever adding helpers). If a
strategy has plateaued, propose a mechanically different one; variants of a failing idea usually fail too.
"""

IDEAS_USER = """\
{analysis_section}## Current program
```python
{current_code}
```

{history_section}
Propose exactly {n} hypothesis IDEAS for improving the program. Do NOT write code yet.
The {n} ideas must be mechanically distinct: different *operations* on the text (e.g. add a chunk,
remove/clean text, expand/normalize vocabulary, change the unit of indexing), not parameter variations
of one idea, and not repeats of anything in the search history.

Use exactly this format for each idea:

### H1: <one-line description>
Rationale: <why this should improve held-out retrieval, citing the analysis>
Mechanism: <add chunk | modify chunk | remove text | expand tokens | rewrite | ...>
"""

CODE_USER = """\
Write a complete representation program implementing this hypothesis:

### {cid}: {description}
Rationale: {rationale}

## Current program (reference only; you need not keep it)
```python
{current_code}
```

{analysis_section}Output ONLY one ```python``` block containing the whole program. It must start with:
```python
""" + PROGRAM_HEADER + """```
and define `class Preprocessor(RepresentationProgram)` with `preprocess(self, docs)`.
"""

SYNTHESIS_USER = """\
{analysis_section}## Candidates that each improved validation Recall@100
{proven_section}
## Current program
```python
{current_code}
```

Combine ALL of these candidates into one program that keeps the best idea from each.
Output ONLY one ```python``` block with the complete program (same header and interface as above).
"""

NO_ANALYSIS_SECTION = """\
## Current validation metrics
- Recall@100: {recall_at_100:.4f}
- nDCG@10: {ndcg_at_10:.4f}

"""


def render(template: str, **kwargs) -> str:
    return template.format(**kwargs)
