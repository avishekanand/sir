from ragtune.indexing.autoindex.agents import (
    AnalysisAgent,
    CodeAgent,
    PromptContext,
    extract_code,
    parse_ideas,
)
from ragtune.indexing.autoindex.bm25 import ChunkBM25Index
from ragtune.indexing.autoindex.llm import LLMMessage, ToolCall
from ragtune.indexing.autoindex.metrics import EvalSummary, QueryOutcome, evaluate_index
from ragtune.indexing.autoindex.program import Chunk
from ragtune.indexing.autoindex.tools import AnalysisToolbox

from .conftest import TITLE_PROGRAM, ScriptedLLM, fenced, make_analysis_llm, make_code_llm

CTX = PromptContext(n_val=10, n_eval=20, tokenizer="lowercased")


def _toolbox(toy_data):
    corpus, queries, qrels = toy_data
    idx = ChunkBM25Index([Chunk(f"{d}_0", d, e["text"]) for d, e in corpus.items()], backend="python")
    return AnalysisToolbox(idx, corpus, queries, qrels), idx


def test_prompt_context_fills_counts():
    text = CTX.fill("{n_val}/{n_eval} {one_query_pct} {tokenizer}")
    assert text == "10/20 +10.00% lowercased"


def test_parse_ideas_and_extract_code():
    ideas = parse_ideas("### H1: add title\nRationale: titles\nmatter\nMechanism: add\n### H2: split\nRationale: r2\n")
    assert [(i.id, i.description) for i in ideas] == [("H1", "add title"), ("H2", "split")]
    assert ideas[0].rationale == "titles\nmatter"
    assert extract_code(fenced(TITLE_PROGRAM)).startswith("from typing")
    assert extract_code("```python\nx = 1\n```") is None


def test_select_targets_stratifies():
    def o(q, hit, rank=None):
        return QueryOutcome(q, 0, 1.0 if hit else 0.0, 0, [rank] if hit else [], ["w1", "w2"])

    cur = EvalSummary(0, 0, 0, [o("a", False), o("b", True, 50), o("c", True, 2), o("d", False)])
    init = EvalSummary(0, 0, 0, [o("a", True, 1), o("b", True, 1), o("c", True, 1), o("d", False)])
    agent = AnalysisAgent(make_analysis_llm(), CTX)
    t = agent.select_targets(cur, init)
    assert [x.query_id for x in t["regressions"]] == ["a"]
    assert [x.query_id for x in t["misses"]] == ["a", "d"]
    assert [x.query_id for x in t["successes"]] == ["b", "c"]


def test_analysis_runs_tools_then_summarizes(toy_data):
    box, idx = _toolbox(toy_data)
    s = evaluate_index(idx, box.val_queries, box.val_qrels)
    llm = make_analysis_llm(n_tool_turns=3)
    res = AnalysisAgent(llm, CTX).analyze(s, s, box, "code", history_text="## Search history")
    assert res.summary.startswith("Titles are not indexed")
    assert res.tool_turns == 3
    assert box.calls == ["bm25_retrieve", "read_file", "grep_search"]
    assert "## Search history" in llm.calls[0][1]["content"]
    assert any(m["role"] == "tool" for m in res.messages)


def test_analysis_nudges_when_no_tools_and_no_summary(toy_data):
    box, idx = _toolbox(toy_data)
    s = evaluate_index(idx, box.val_queries, box.val_qrels)
    replies = iter([LLMMessage("thinking..."), LLMMessage("", [ToolCall("x", "bm25_retrieve", '{"query":"a"}')]),
                    LLMMessage("<summary>ok</summary>")])
    llm = ScriptedLLM(lambda m, t: next(replies))
    res = AnalysisAgent(llm, CTX, min_tool_turns=1).analyze(s, s, box, "code")
    assert res.summary == "ok"
    assert any("tool-using turns" in m["content"] for m in res.messages if m["role"] == "user")


def test_analysis_forces_summary_after_max_turns(toy_data):
    box, idx = _toolbox(toy_data)
    s = evaluate_index(idx, box.val_queries, box.val_qrels)

    def respond(messages, tools):
        if tools:
            return LLMMessage("", [ToolCall("x", "bm25_retrieve", '{"query":"a"}')])
        assert not any(m["role"] == "tool" for m in messages)  # flattened for tool-less call
        return LLMMessage("<summary>forced</summary>")

    res = AnalysisAgent(ScriptedLLM(respond), CTX, max_turns=2).analyze(s, s, box, "code")
    assert res.summary == "forced" and res.tool_turns == 2


def test_code_agent_propose_two_phase():
    llm = make_code_llm({"H1": TITLE_PROGRAM, "H2": "not code"})
    cands = CodeAgent(llm, CTX).propose("analysis", "current", n=2, history_text="HIST")
    assert [c.id for c in cands] == ["H1", "H2"]
    assert cands[0].code.startswith("from typing") and cands[1].code == ""
    assert "HIST" in llm.calls[0][1]["content"]
    assert len(llm.calls) == 3  # 1 ideas call + 2 code calls


def test_code_agent_no_ideas_returns_empty():
    assert CodeAgent(ScriptedLLM(lambda m, t: LLMMessage("nothing")), CTX).propose("", "c", 4) == []
