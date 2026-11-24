"""
Minimal chat-completion interface for the AutoIndex agents.

``LLMClient`` is the protocol the agents depend on; tests pass fakes.
``LiteLLMClient`` is the default implementation and records token usage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, as returned by OpenAI-style APIs


@dataclass
class LLMMessage:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def complete(self, messages: List[dict], tools: Optional[List[dict]] = None) -> LLMMessage:
        ...


@dataclass
class UsageTracker:
    """Per-agent call counts, token usage and latency."""

    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def record(self, agent: str, prompt_tokens: int, completion_tokens: int, seconds: float) -> None:
        s = self.stats.setdefault(agent, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0})
        s["calls"] += 1
        s["prompt_tokens"] += prompt_tokens
        s["completion_tokens"] += completion_tokens
        s["seconds"] += seconds

    def total_tokens(self) -> int:
        return int(sum(s["prompt_tokens"] + s["completion_tokens"] for s in self.stats.values()))


class LiteLLMClient:
    """LLMClient backed by ``litellm.completion`` (one retry on failure)."""

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        usage: Optional[UsageTracker] = None,
        agent: str = "llm",
        retry_wait_s: float = 5.0,
    ):
        self.model = model
        self.temperature = temperature
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.usage = usage
        self.agent = agent
        self.retry_wait_s = retry_wait_s

    def complete(self, messages: List[dict], tools: Optional[List[dict]] = None) -> LLMMessage:
        import litellm

        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature)
        for k in ("api_base", "api_key", "timeout"):
            if getattr(self, k) is not None:
                kwargs[k] = getattr(self, k)
        if tools:
            kwargs["tools"] = tools

        t0 = time.time()
        try:
            response = litellm.completion(**kwargs)
        except Exception as e:
            logger.warning("LLM call failed (%s: %s); retrying once", type(e).__name__, e)
            time.sleep(self.retry_wait_s)
            response = litellm.completion(**kwargs)

        usage = getattr(response, "usage", None)
        if self.usage is not None:
            self.usage.record(
                self.agent,
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
                time.time() - t0,
            )
        msg = response.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (getattr(msg, "tool_calls", None) or [])
        ]
        return LLMMessage(content=msg.content or "", tool_calls=calls)
