"""Fakes and fixtures for AutoIndex tests. No network, no real LLM calls."""

from typing import Callable, List, Optional

import pytest

from ragtune.indexing.autoindex.llm import LLMMessage, ToolCall
from ragtune.indexing.autoindex.program import IDENTITY_PROGRAM_CODE

HEADER = (
    "from typing import List\n"
    "from ragtune.indexing.autoindex.program import Document, Chunk, RepresentationProgram\n\n"
)


def program(body_expr: str) -> str:
    """A program whose single chunk per doc has text ``body_expr`` (may reference ``d``)."""
    return HEADER + (
        "class Preprocessor(RepresentationProgram):\n"
        "    def preprocess(self, docs: List[Document]) -> List[Chunk]:\n"
        f"        return [Chunk(f'{{d.doc_id}}_0', d.doc_id, {body_expr}) for d in docs]\n"
    )


TITLE_PROGRAM = program("d.metadata.get('title', '') + ' ' + d.text")
SPLIT_PROGRAM = program("d.text.replace('_', ' ')")
BOTH_PROGRAM = program("d.metadata.get('title', '') + ' ' + d.text.replace('_', ' ')")
EMPTY_PROGRAM = program("'nothing'")


def fenced(code: str) -> str:
    return f"Here you go:\n```python\n{code}```\n"


class ScriptedLLM:
    """Replies via ``responder(messages, tools)``; records every call."""

    def __init__(self, responder: Callable[[List[dict], Optional[list]], LLMMessage]):
        self.responder = responder
        self.calls: List[List[dict]] = []

    def complete(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        return self.responder(messages, tools)


def make_code_llm(idea_to_code: dict, synthesis_code: Optional[str] = None) -> ScriptedLLM:
    """Ideas = keys of ``idea_to_code`` (H1, H2, ...); code reply looked up by idea id."""

    def respond(messages, tools):
        user = messages[-1]["content"]
        if "hypothesis IDEAS" in user:
            return LLMMessage("\n".join(f"### {k}: idea {k}\nRationale: because {k}\nMechanism: x\n"
                                        for k in idea_to_code))
        if "Combine ALL" in user:
            return LLMMessage(fenced(synthesis_code) if synthesis_code else "no code")
        for k, code in idea_to_code.items():
            if f"### {k}:" in user:
                return LLMMessage(fenced(code))
        return LLMMessage("???")

    return ScriptedLLM(respond)


def make_analysis_llm(n_tool_turns: int = 3) -> ScriptedLLM:
    tools = [
        ToolCall("c1", "bm25_retrieve", '{"query": "alpha", "top_k": 3}'),
        ToolCall("c2", "read_file", '{"file_path": "documents.jsonl", "filter_id": "d0"}'),
        ToolCall("c3", "grep_search", '{"pattern": "alpha", "file_path": "documents.jsonl"}'),
    ]

    def respond(messages, tools_arg):
        done = sum(1 for m in messages if m["role"] == "assistant" and m.get("tool_calls"))
        if tools_arg and done < n_tool_turns:
            return LLMMessage("", [tools[done % len(tools)]])
        return LLMMessage("<summary>Titles are not indexed; underscores glue words.</summary>")

    return ScriptedLLM(respond)


@pytest.fixture
def toy_data():
    """Queries 0-3 need the title, 4-7 need underscore splitting, 8-9 already hit."""
    corpus, queries, qrels = {}, {}, {}
    for i in range(10):
        text = f"common filler text number{i}"
        if 4 <= i < 8:
            text += f" glued_word{i}"
        corpus[f"d{i}"] = {"text": text, "title": f"title{i}"}
    for i in range(4):
        queries[f"q{i}"] = f"title{i}"
    for i in range(4, 8):
        queries[f"q{i}"] = f"word{i}"
    for i in range(8, 10):
        queries[f"q{i}"] = f"number{i}"
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(10)}
    return corpus, queries, qrels


@pytest.fixture
def identity_code():
    return IDENTITY_PROGRAM_CODE
