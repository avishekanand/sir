import time

import pytest

from ragtune.indexing.autoindex.program import (
    IDENTITY_PROGRAM_CODE,
    Chunk,
    Document,
    IdentityProgram,
    ProgramError,
    RepresentationProgram,
    corpus_to_documents,
    hash_doc_id,
    load_program,
    run_program,
    screen_program_source,
    validate_program_code,
)

HEADER = (
    "from typing import List\n"
    "from ragtune.indexing.autoindex.program import Document, Chunk, RepresentationProgram\n"
)


def _code(body: str) -> str:
    return HEADER + "\nclass Preprocessor(RepresentationProgram):\n" + body


DOCS = [Document("a", "alpha text"), Document("b", "beta text")]


def test_identity_program_one_chunk_per_doc():
    chunks = run_program(IdentityProgram(), DOCS)
    assert [(c.doc_id, c.text) for c in chunks] == [("a", "alpha text"), ("b", "beta text")]


def test_identity_code_loads_and_matches_class():
    prog = load_program(IDENTITY_PROGRAM_CODE)
    assert isinstance(prog, RepresentationProgram)
    assert [c.text for c in run_program(prog, DOCS)] == ["alpha text", "beta text"]


def test_program_sees_hashed_ids_and_output_is_remapped():
    seen = []

    class Spy(RepresentationProgram):
        def preprocess(self, docs):
            seen.extend(d.doc_id for d in docs)
            return [Chunk(f"{d.doc_id}_0", d.doc_id, d.text) for d in docs]

    chunks = run_program(Spy(), DOCS)
    assert seen == [hash_doc_id("a"), hash_doc_id("b")]
    assert [c.doc_id for c in chunks] == ["a", "b"]


def test_anonymize_can_be_disabled():
    chunks = run_program(IdentityProgram(), DOCS, anonymize_ids=False)
    assert chunks[0].chunk_id == "a_0"


@pytest.mark.parametrize("body,msg", [
    ("    def preprocess(self, docs):\n        return [Chunk('x', 'nope', 't')]\n", "unknown doc_id"),
    ("    def preprocess(self, docs):\n        return [Chunk('x', d.doc_id, d.text) for d in docs]\n", "duplicate chunk_id"),
    ("    def preprocess(self, docs):\n        return [Chunk('x', docs[0].doc_id, 't')]\n", "produced no chunks"),
    ("    def preprocess(self, docs):\n        return 'oops'\n", "must return a list"),
    ("    def preprocess(self, docs):\n        raise ValueError('boom')\n", "ValueError in preprocess"),
])
def test_contract_violations_raise(body, msg):
    with pytest.raises(ProgramError, match=msg):
        run_program(load_program(_code(body)), DOCS)


@pytest.mark.parametrize("snippet", [
    "import os\n",
    "import subprocess\n",
    "from pathlib import Path\n",
    "from urllib.request import urlopen\n",
    "x = open('f')\n",
    "x = eval('1')\n",
    "x = ().__class__\n",
    "x = getattr(len, '__globals__', None)\n",
])
def test_screen_rejects_dangerous_source(snippet):
    with pytest.raises(ProgramError):
        screen_program_source(snippet)


def test_screen_allows_common_idioms():
    screen_program_source(
        "import re, collections, string\n"
        "class A:\n    def __init__(self):\n        super().__init__()\n"
        "x = getattr(A, 'name', None)\n"
        "if __name__ == '__main__':\n    pass\n"
    )


def test_load_requires_preprocessor_class():
    with pytest.raises(ProgramError, match="Preprocessor"):
        load_program(HEADER + "x = 1\n")


def test_load_reports_syntax_error():
    with pytest.raises(ProgramError, match="SyntaxError"):
        load_program("def broken(:\n")


def test_timeout_raises():
    class Slow(RepresentationProgram):
        def preprocess(self, docs):
            time.sleep(0.5)
            return []

    with pytest.raises(ProgramError, match="timed out"):
        run_program(Slow(), DOCS, timeout_s=0.05)


def test_validate_program_code_returns_message_or_none():
    assert validate_program_code(IDENTITY_PROGRAM_CODE, DOCS) is None
    assert "not allowed" in validate_program_code("import os\n", DOCS)


def test_corpus_to_documents_keeps_title_in_metadata():
    docs = corpus_to_documents({"1": {"text": "t", "title": "T"}, "2": {"text": "u", "title": ""}})
    assert docs[0].metadata == {"title": "T"}
    assert docs[1].metadata == {}


def test_available_packages_reports_only_importable(monkeypatch):
    import importlib.util

    from ragtune.indexing.autoindex import program as prog

    present = {"numpy", "spacy", "en_core_web_sm"}
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name in present else None)
    assert prog.available_packages() == ["spacy", "numpy", "spacy model 'en_core_web_sm'"]
