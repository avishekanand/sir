"""
AutoIndex: learned representation programs for BM25.

Implements O'Nuallain et al., "AutoIndex: Learning Representation Programs for
Retrieval" (arXiv:2607.18603). See ``specs/spec-autoindex.md``.
"""

from ragtune.indexing.autoindex.program import (
    IDENTITY_PROGRAM_CODE,
    Chunk,
    Document,
    IdentityProgram,
    ProgramError,
    RepresentationProgram,
    corpus_to_documents,
    load_program,
    run_program,
)
from ragtune.indexing.autoindex.bm25 import ChunkBM25Index, DocHit, Tokenizer
from ragtune.indexing.autoindex.metrics import EvalSummary, QueryOutcome, evaluate_index
from ragtune.indexing.autoindex.llm import LiteLLMClient, LLMClient, UsageTracker
from ragtune.indexing.autoindex.loop import AutoIndex, AutoIndexConfig, AutoIndexResult

__all__ = [
    "IDENTITY_PROGRAM_CODE",
    "Chunk",
    "Document",
    "IdentityProgram",
    "ProgramError",
    "RepresentationProgram",
    "corpus_to_documents",
    "load_program",
    "run_program",
    "ChunkBM25Index",
    "DocHit",
    "Tokenizer",
    "EvalSummary",
    "QueryOutcome",
    "evaluate_index",
    "LiteLLMClient",
    "LLMClient",
    "UsageTracker",
    "AutoIndex",
    "AutoIndexConfig",
    "AutoIndexResult",
]
