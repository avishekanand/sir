"""
AutoIndex agents.

- ``AnalysisAgent`` investigates a stratified slice of validation queries with
  read-only tools and returns a structured failure summary.
- ``CodeAgent`` proposes N mechanically distinct ideas, writes one program per
  idea in parallel, and can merge several winning programs into one.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ragtune.indexing.autoindex import prompts
from ragtune.indexing.autoindex.llm import LLMClient
from ragtune.indexing.autoindex.metrics import EvalSummary
from ragtune.indexing.autoindex.tools import TOOL_SCHEMAS, AnalysisToolbox

logger = logging.getLogger(__name__)

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
_CODE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_IDEA_RE = re.compile(r"^###\s+(H\d+)\s*:\s*(.+?)\s*$", re.MULTILINE)


@dataclass
class PromptContext:
    """Values shared by both agents' system prompts."""

    n_val: int
    n_eval: int
    tokenizer: str
    corpus_description: str = "(no corpus description provided)"
    min_tool_turns: int = 3
    packages: str = "none"

    def fill(self, template: str) -> str:
        pct = f"+{100.0 / self.n_val:.2f}%" if self.n_val else "n/a"
        return prompts.render(
            template, n_val=self.n_val, n_eval=self.n_eval or "unknown", one_query_pct=pct,
            tokenizer=self.tokenizer, corpus_description=self.corpus_description,
            min_tool_turns=self.min_tool_turns, packages=self.packages,
        )


# ---------------------------------------------------------------------------
# Analysis Agent
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    summary: str
    tool_turns: int
    messages: List[dict] = field(default_factory=list)


class AnalysisAgent:
    def __init__(
        self,
        llm: LLMClient,
        ctx: PromptContext,
        max_turns: int = 5,
        min_tool_turns: int = 3,
        n_regressions: int = 5,
        n_hard_negatives: int = 5,
        n_successes: int = 8,
    ):
        self.llm = llm
        self.system = ctx.fill(prompts.ANALYSIS_SYSTEM)
        self.max_turns = max_turns
        self.min_tool_turns = min_tool_turns
        self.n_regressions = n_regressions
        self.n_hard_negatives = n_hard_negatives
        self.n_successes = n_successes

    def select_targets(self, current: EvalSummary, initial: EvalSummary) -> Dict[str, list]:
        """The paper's stratified slice: regressions vs θ₀, misses with top wrong docs, worst-ranked hits."""
        init = initial.by_qid()
        regressions = [o for o in current.per_query
                       if o.query_id in init and init[o.query_id].hit and not o.hit][: self.n_regressions]
        misses = [o for o in current.per_query if not o.hit][: self.n_hard_negatives]
        hits = sorted((o for o in current.per_query if o.hit), key=lambda o: o.rank or 0, reverse=True)
        return {"regressions": regressions, "misses": misses, "successes": hits[: self.n_successes]}

    def _initial_message(self, current, initial, targets, toolbox, current_code, history_text) -> str:
        def q(o):
            return json.dumps(toolbox.val_queries.get(o.query_id, ""))[:300]

        def gold(o):
            return [d for d, r in toolbox.val_qrels.get(o.query_id, {}).items() if r > 0]

        lines = []
        if history_text:
            lines += [history_text, ""]
        lines += [
            f"## Current validation metrics ({len(current.per_query)} queries)",
            f"- Recall@100: {current.recall_at_100:.4f} (initial program: {initial.recall_at_100:.4f})",
            f"- nDCG@10: {current.ndcg_at_10:.4f} (initial program: {initial.ndcg_at_10:.4f})",
            "", "## Current program", "```python", current_code, "```", "",
            "## Analysis targets (investigate failures AND successes)",
        ]
        lines.append(f"### Regressions vs the initial program ({len(targets['regressions'])})")
        for o in targets["regressions"] or []:
            lines.append(f"- [{o.query_id}] {q(o)} | gold={gold(o)} | retrieved top5={o.retrieved_doc_ids[:5]}")
        lines.append(f"### Misses / hard negatives ({len(targets['misses'])})")
        for o in targets["misses"]:
            wrong = [d for d in o.retrieved_doc_ids if d not in gold(o)][:3]
            lines.append(f"- [{o.query_id}] {q(o)} | gold={gold(o)} | top wrong docs={wrong}")
        lines.append(f"### Successes, worst rank first ({len(targets['successes'])})")
        for o in targets["successes"]:
            lines.append(f"- [{o.query_id}] rank={o.rank} {q(o)}")
        lines += ["", "Use the tools (bm25_retrieve, read_file, grep_search), then write <summary>...</summary>."]
        return "\n".join(lines)

    def analyze(
        self,
        current: EvalSummary,
        initial: EvalSummary,
        toolbox: AnalysisToolbox,
        current_code: str,
        history_text: Optional[str] = None,
    ) -> AnalysisResult:
        targets = self.select_targets(current, initial)
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self._initial_message(current, initial, targets, toolbox,
                                                               current_code, history_text)},
        ]
        tool_turns = 0
        summary = None
        for _ in range(self.max_turns):
            msg = self.llm.complete(messages, tools=TOOL_SCHEMAS)
            entry: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in msg.tool_calls
                ]
            messages.append(entry)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = toolbox.dispatch(tc.name, args if isinstance(args, dict) else {})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                tool_turns += 1
                continue

            m = _SUMMARY_RE.search(msg.content or "")
            if m:
                summary = m.group(1).strip()
                break
            if tool_turns < self.min_tool_turns:
                messages.append({"role": "user", "content": (
                    f"You have completed only {tool_turns}/{self.min_tool_turns} tool-using turns. "
                    "Investigate more queries with the tools before summarizing.")})
                continue
            break

        if summary is None:
            summary = self._force_summary(messages)
        return AnalysisResult(summary=summary, tool_turns=tool_turns, messages=messages)

    def _force_summary(self, messages: List[dict]) -> str:
        """Ask for a summary with tools disabled (tool traffic flattened to plain text)."""
        for m in messages:
            if m["role"] == "assistant":
                found = _SUMMARY_RE.search(m.get("content") or "")
                if found:
                    return found.group(1).strip()
        flat = []
        for m in messages:
            if m["role"] == "tool":
                flat.append({"role": "user", "content": f"[tool result]\n{m['content']}"})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                calls = ", ".join(f"{tc['function']['name']}({tc['function']['arguments']})" for tc in m["tool_calls"])
                flat.append({"role": "assistant", "content": f"{m.get('content') or ''}\n[called {calls}]".strip()})
            else:
                flat.append(m)
        flat.append({"role": "user", "content": (
            "Tools are now disabled. Write your FINAL analysis from what you have already found, "
            "citing concrete query/doc IDs, wrapped in <summary>...</summary>.")})
        text = self.llm.complete(flat).content or ""
        found = _SUMMARY_RE.search(text)
        return (found.group(1) if found else text).strip() or "Analysis produced no summary."


# ---------------------------------------------------------------------------
# Code Agent
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    id: str
    description: str
    rationale: str = ""
    code: str = ""


def parse_ideas(text: str) -> List[Candidate]:
    headers = list(_IDEA_RE.finditer(text))
    ideas = []
    for i, h in enumerate(headers):
        body = text[h.end(): headers[i + 1].start() if i + 1 < len(headers) else len(text)]
        rat = re.search(r"Rationale:\s*(.+?)(?:\n[A-Z][a-z]+:|\n###|\Z)", body, re.DOTALL)
        ideas.append(Candidate(id=h.group(1), description=h.group(2).strip(),
                               rationale=rat.group(1).strip() if rat else ""))
    return ideas


def extract_code(text: str) -> Optional[str]:
    m = _CODE_RE.search(text or "")
    if not m or "class Preprocessor" not in m.group(1):
        return None
    return m.group(1).strip()


class CodeAgent:
    def __init__(self, llm: LLMClient, ctx: PromptContext, max_workers: int = 4):
        self.llm = llm
        self.system = ctx.fill(prompts.CODE_SYSTEM)
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers

    def _ask(self, user: str) -> str:
        return self.llm.complete([{"role": "system", "content": self.system},
                                  {"role": "user", "content": user}]).content or ""

    @staticmethod
    def _analysis_section(analysis: str, limit: Optional[int] = None) -> str:
        if not analysis:
            return ""
        text = analysis if limit is None else analysis[:limit]
        return f"## Analysis summary\n{text}\n\n"

    def propose(self, analysis: str, current_code: str, n: int, history_text: str = "") -> List[Candidate]:
        """Phase 1: n ideas in one call. Phase 2: one program per idea, in parallel."""
        text = self._ask(prompts.render(
            prompts.IDEAS_USER, analysis_section=self._analysis_section(analysis),
            current_code=current_code, history_section=history_text, n=n))
        ideas = parse_ideas(text)[:n]
        if not ideas:
            logger.warning("Code agent produced no parseable ideas")
            return []

        def write(idea: Candidate) -> Candidate:
            reply = self._ask(prompts.render(
                prompts.CODE_USER, cid=idea.id, description=idea.description, rationale=idea.rationale,
                current_code=current_code, analysis_section=self._analysis_section(analysis, 2000)))
            idea.code = extract_code(reply) or ""
            return idea

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(ideas))) as ex:
            futures = [ex.submit(write, idea) for idea in ideas]
            out = []
            for idea, fut in zip(ideas, futures):
                try:
                    out.append(fut.result())
                except Exception as e:
                    logger.warning("Code generation failed for %s: %s", idea.id, e)
        return out

    def synthesize(
        self,
        analysis: str,
        proven: Sequence[tuple],  # (Candidate, delta_recall_100, delta_ndcg_10)
        current_code: str,
    ) -> Optional[str]:
        section = "".join(
            f"### {c.id}: {c.description}\n- ΔR@100 {d_r:+.4f}, ΔnDCG@10 {d_n:+.4f}\n```python\n{c.code}\n```\n\n"
            for c, d_r, d_n in proven
        )
        reply = self._ask(prompts.render(
            prompts.SYNTHESIS_USER, analysis_section=self._analysis_section(analysis),
            proven_section=section, current_code=current_code))
        return extract_code(reply)
