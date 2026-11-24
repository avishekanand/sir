from types import SimpleNamespace

import pytest

litellm = pytest.importorskip("litellm")

from ragtune.indexing.autoindex.llm import LiteLLMClient, UsageTracker  # noqa: E402


def _response(content="hi", tool_calls=None, pt=10, ct=5):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct))


def test_parses_tool_calls_and_records_usage(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        tc = SimpleNamespace(id="t1", function=SimpleNamespace(name="bm25_retrieve", arguments='{"query":"x"}'))
        return _response(content=None, tool_calls=[tc])

    monkeypatch.setattr(litellm, "completion", fake)
    usage = UsageTracker()
    msg = LiteLLMClient("m", usage=usage, agent="analysis").complete([{"role": "user", "content": "q"}], tools=[{}])
    assert msg.content == "" and msg.tool_calls[0].name == "bm25_retrieve"
    assert seen["tools"] == [{}] and "api_base" not in seen
    assert usage.stats["analysis"]["calls"] == 1 and usage.total_tokens() == 15


def test_retries_once(monkeypatch):
    attempts = []

    def flaky(**kw):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("rate limited")
        return _response()

    monkeypatch.setattr(litellm, "completion", flaky)
    assert LiteLLMClient("m", retry_wait_s=0).complete([]).content == "hi"
    assert len(attempts) == 2
