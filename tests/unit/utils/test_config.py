"""
Unit tests for ConfigLoader's two stores.

ConfigLoader keeps defaults.yaml in _config and prompts.yaml in _prompts, with
separate accessors for each. The stores must stay isolated: prompt optimization
(specs/spec-prompt-optimization.md) swaps templates between evaluations by
writing to the prompt store, and a leak in either direction would silently
change the other component's behaviour.
"""

import pytest

from ragtune.utils.config import ConfigLoader, config


@pytest.fixture
def cfg():
    """
    The loader is a singleton, so snapshot and restore both stores.

    Without this a test that sets a key leaks into every later test in the
    session — including the tuning tests, which read retrieval.* defaults.
    """
    import copy

    saved_config = copy.deepcopy(ConfigLoader._config)
    saved_prompts = copy.deepcopy(ConfigLoader._prompts)
    yield config
    ConfigLoader._config = saved_config
    ConfigLoader._prompts = saved_prompts
    config._config = saved_config
    config._prompts = saved_prompts


def test_get_reads_defaults(cfg):
    assert cfg.get("retrieval.original_query_depth") == 10


def test_get_missing_key_returns_default(cfg):
    assert cfg.get("retrieval.does_not_exist") is None
    assert cfg.get("nope.nope.nope", "fallback") == "fallback"


def test_set_creates_intermediate_dicts(cfg):
    cfg.set("brand.new.nested.key", 7)
    assert cfg.get("brand.new.nested.key") == 7


def test_set_overrides_existing_value(cfg):
    cfg.set("retrieval.original_query_depth", 123)
    assert cfg.get("retrieval.original_query_depth") == 123


def test_get_prompt_reads_prompts_file(cfg):
    prompt = cfg.get_prompt("reformulation.llm_rewrite")
    assert isinstance(prompt, dict)
    assert "{query}" in prompt["user"]


def test_set_prompt_roundtrips(cfg):
    payload = {"system": "sys", "user": "Query: {query}\nMake {m} variants."}
    cfg.set_prompt("reformulation.llm_rewrite", payload)
    assert cfg.get_prompt("reformulation.llm_rewrite") == payload


def test_set_prompt_creates_intermediate_dicts(cfg):
    cfg.set_prompt("brand.new.prompt", {"system": "s", "user": "u"})
    assert cfg.get_prompt("brand.new.prompt") == {"system": "s", "user": "u"}


def test_stores_are_isolated(cfg):
    """set() must not be reachable via get_prompt(), and vice versa."""
    cfg.set("isolated.key", "from_config")
    cfg.set_prompt("isolated.key", "from_prompts")

    assert cfg.get("isolated.key") == "from_config"
    assert cfg.get_prompt("isolated.key") == "from_prompts"


def test_singleton_shares_state(cfg):
    """A second handle sees writes made through the first."""
    cfg.set_prompt("shared.probe", {"user": "{query}"})
    assert ConfigLoader().get_prompt("shared.probe") == {"user": "{query}"}
