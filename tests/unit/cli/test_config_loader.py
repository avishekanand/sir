"""ConfigLoader budget overrides: KEY=VALUE parsing, limit removal and replacement."""

import pytest

from ragtune.cli.config_loader import ConfigLoader


def test_parse_limit_overrides():
    parsed = ConfigLoader.parse_limit_overrides(["rerank.latency_ms=500", "tokens=none", " docs = 3 "])
    assert parsed == {"rerank.latency_ms": 500.0, "tokens": None, "docs": 3.0}


@pytest.mark.parametrize("item, message", [
    ("tokens", "Invalid limit"), ("tokens=", "Invalid limit"), ("=5", "Invalid limit"),
    ("tokens=abc", "Invalid limit"), ("rerenk.docs=1", "Unknown budget component"),
])
def test_parse_limit_overrides_rejects_malformed(item, message):
    with pytest.raises(ValueError, match=message):
        ConfigLoader.parse_limit_overrides([item])


def _config(limits):
    return {"pipeline": {
        "components": {
            "retriever": {"type": "in-memory", "params": {"documents": []}},
            "reranker": {"type": "noop"},
        },
        "budget": {"limits": limits},
    }}


def test_create_controller_overrides_remove_and_replace():
    import ragtune.components  # noqa: F401  populate registry

    config = _config({"tokens": 4000, "rerank_docs": 50})
    merged = ConfigLoader.create_controller(config, {"tokens": None, "rerank.docs": 5.0})
    assert merged.budget.limits == {"rerank_docs": 50, "rerank.docs": 5.0}

    replaced = ConfigLoader.create_controller(config, {"rerank.latency_ms": 500.0}, replace_limits=True)
    assert replaced.budget.limits == {"rerank.latency_ms": 500.0}

    assert config["pipeline"]["budget"]["limits"] == {"tokens": 4000, "rerank_docs": 50}  # not mutated
