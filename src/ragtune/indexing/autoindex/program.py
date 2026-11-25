"""
Representation programs: executable document -> chunk transformations.

A representation program is a ``RepresentationProgram`` subclass, conventionally
named ``Preprocessor``, whose ``preprocess(docs)`` returns the chunks BM25 indexes.
AutoIndex writes these programs as source code; ``load_program`` screens and
executes that source and ``run_program`` runs a program over a corpus while
enforcing the contract.
"""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import importlib.util
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Document:
    """A raw corpus document as seen by a representation program."""

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """An indexable unit. ``doc_id`` links it back to its source Document."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RepresentationProgram(ABC):
    """Base class for all representation programs.

    Contract for ``preprocess``:
      - return at least one Chunk per Document;
      - every ``chunk.doc_id`` is one of the input ``doc_id`` values;
      - ``chunk_id`` values are unique.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """Transform raw documents into chunks for BM25 indexing."""


class IdentityProgram(RepresentationProgram):
    """θ₀: one chunk per document containing the raw text (the paper's baseline)."""

    name = "identity"
    description = "One chunk per document, raw text."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return [Chunk(chunk_id=f"{d.doc_id}_0", doc_id=d.doc_id, text=d.text) for d in docs]


IDENTITY_PROGRAM_CODE = '''\
from typing import List
from ragtune.indexing.autoindex.program import Document, Chunk, RepresentationProgram


class Preprocessor(RepresentationProgram):
    name = "identity"
    description = "One chunk per document, raw text."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return [Chunk(chunk_id=f"{d.doc_id}_0", doc_id=d.doc_id, text=d.text) for d in docs]
'''


class ProgramError(Exception):
    """A representation program failed to load, run, or honour its contract."""


# ---------------------------------------------------------------------------
# Loading (AST screen + exec)
# ---------------------------------------------------------------------------

_BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "ctypes",
    "multiprocessing", "threading", "signal", "requests", "urllib", "http", "ftplib",
    "smtplib", "telnetlib", "asyncio", "pickle", "marshal", "builtins", "tempfile",
    "glob", "io",
})
_BLOCKED_CALLS = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "setattr", "delattr",
})


# Introspection dunders that give a route from ordinary objects to builtins/globals.
_BLOCKED_DUNDERS = frozenset({
    "__globals__", "__builtins__", "__subclasses__", "__bases__", "__base__", "__mro__",
    "__class__", "__dict__", "__code__", "__closure__", "__import__", "__getattribute__",
    "__loader__", "__spec__", "__reduce__", "__reduce_ex__", "__func__", "__self__",
    "__frame__", "f_globals", "f_back", "gi_frame", "tb_frame",
})


def _is_dunder(name: str) -> bool:
    return name in _BLOCKED_DUNDERS


def screen_program_source(code: str) -> None:
    """Reject source that imports blocked modules, calls blocked builtins, or uses introspection dunders.

    This is a guardrail against accidental damage by generated code, not a sandbox.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ProgramError(f"SyntaxError: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            if name.split(".")[0] in _BLOCKED_MODULES:
                raise ProgramError(f"import of '{name}' is not allowed in representation programs")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALLS:
                raise ProgramError(f"call to '{node.func.id}()' is not allowed in representation programs")
        if isinstance(node, ast.Attribute) and _is_dunder(node.attr):
            raise ProgramError(f"attribute access '{node.attr}' is not allowed")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _is_dunder(node.value):
            raise ProgramError(f"string {node.value!r} is not allowed")


def load_program(code: str) -> RepresentationProgram:
    """Screen and exec ``code``; return an instance of its ``Preprocessor`` class."""
    screen_program_source(code)
    module = types.ModuleType("_autoindex_program")
    try:
        exec(compile(code, "<autoindex_program>", "exec"), module.__dict__)
    except Exception as e:
        raise ProgramError(f"{type(e).__name__} while loading program: {e}") from e

    cls = getattr(module, "Preprocessor", None)
    if not isinstance(cls, type) or not issubclass(cls, RepresentationProgram):
        raise ProgramError("program must define `class Preprocessor(RepresentationProgram)`")
    return cls()


# ---------------------------------------------------------------------------
# Running (doc-id hashing, timeout, contract)
# ---------------------------------------------------------------------------

# Third-party packages a program might reasonably want. Only the importable ones are advertised.
_OPTIONAL_PACKAGES = ("nltk", "spacy", "numpy", "regex", "unidecode", "inflect", "rapidfuzz", "Stemmer", "sklearn")
_SPACY_MODELS = ("en_core_web_sm", "en_core_web_md", "en_core_web_lg")


def available_packages() -> List[str]:
    """Optional third-party packages (and spaCy models) importable in this environment."""
    def has(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    found = [p for p in _OPTIONAL_PACKAGES if has(p)]
    if "spacy" in found:
        found += [f"spacy model '{m}'" for m in _SPACY_MODELS if has(m)]
    return found


def corpus_to_documents(corpus: Dict[str, Dict]) -> List[Document]:
    """Convert a RAGtune/BEIR corpus dict into Documents (title kept in metadata)."""
    docs = []
    for doc_id, entry in corpus.items():
        title = (entry.get("title") or "").strip()
        docs.append(Document(doc_id=str(doc_id), text=entry.get("text", ""),
                             metadata={"title": title} if title else {}))
    return docs


def hash_doc_id(doc_id: str) -> str:
    return "doc_" + hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:16]


def _anonymize(docs: List[Document]) -> Tuple[List[Document], Dict[str, str]]:
    reverse: Dict[str, str] = {}
    out = []
    for d in docs:
        h = hash_doc_id(d.doc_id)
        reverse[h] = d.doc_id
        out.append(Document(doc_id=h, text=d.text, metadata=dict(d.metadata)))
    return out, reverse


def _check_contract(chunks: Any, valid_ids: Dict[str, str]) -> None:
    if not isinstance(chunks, list):
        raise ProgramError(f"preprocess() must return a list, got {type(chunks).__name__}")
    seen_chunk_ids = set()
    covered = set()
    for c in chunks:
        if not isinstance(c, Chunk):
            raise ProgramError(f"preprocess() returned a non-Chunk item: {type(c).__name__}")
        if c.doc_id not in valid_ids:
            raise ProgramError(f"chunk {c.chunk_id!r} has unknown doc_id {c.doc_id!r}; "
                               "copy Document.doc_id verbatim")
        if c.chunk_id in seen_chunk_ids:
            raise ProgramError(f"duplicate chunk_id {c.chunk_id!r}")
        if not isinstance(c.text, str):
            raise ProgramError(f"chunk {c.chunk_id!r} text is not a str")
        seen_chunk_ids.add(c.chunk_id)
        covered.add(c.doc_id)
    missing = len(valid_ids) - len(covered)
    if missing:
        raise ProgramError(f"{missing} document(s) produced no chunks; every document needs at least one")


def run_program(
    program: RepresentationProgram,
    docs: List[Document],
    timeout_s: Optional[float] = None,
    anonymize_ids: bool = True,
) -> List[Chunk]:
    """Run ``program`` over ``docs`` and return contract-checked chunks with original doc_ids.

    With ``anonymize_ids`` the program sees hashed doc_ids, mapped back afterwards.
    A program that exceeds ``timeout_s`` raises ProgramError; its thread is abandoned.
    """
    if anonymize_ids:
        inputs, reverse = _anonymize(docs)
    else:
        inputs, reverse = docs, {d.doc_id: d.doc_id for d in docs}

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(program.preprocess, inputs)
    try:
        chunks = future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False, cancel_futures=True)
        raise ProgramError(f"preprocess() timed out after {timeout_s}s")
    except ProgramError:
        ex.shutdown(wait=False)
        raise
    except Exception as e:
        ex.shutdown(wait=False)
        raise ProgramError(f"{type(e).__name__} in preprocess(): {e}") from e
    ex.shutdown(wait=True)

    _check_contract(chunks, reverse)
    for c in chunks:
        c.doc_id = reverse[c.doc_id]
    return chunks


def validate_program_code(code: str, sample_docs: List[Document]) -> Optional[str]:
    """Load ``code`` and run it on a small sample. Returns an error message, or None if OK."""
    try:
        run_program(load_program(code), sample_docs, timeout_s=60)
        return None
    except ProgramError as e:
        return str(e)
