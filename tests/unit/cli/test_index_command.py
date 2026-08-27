"""Regression tests: `ragtune index` must resolve the indexer via IndexConfig.

The command used to read `pipeline.index.framework`, an attribute that does
not exist on IndexConfig (the schema models index selection as
type: "sparse" | "dense" plus optional backend/model). Any invocation of
`ragtune index` crashed with AttributeError before reaching the build step.

The fix routes resolution through IndexFactory.from_config(), which already
implements the documented sparse/dense rules.
"""

import yaml
from typer.testing import CliRunner

from ragtune.cli.main import app
from ragtune.indexing.factory import IndexFactory

runner = CliRunner()


def _make_config(tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text('[{"doc_id": "d1", "content": "hello world", "source": "s"}]')
    cfg = {
        "pipeline": {
            "name": "index-cmd-test",
            "data": {
                "collection_path": str(corpus),
                "collection_format": "json",
            },
            "index": {"type": "sparse", "index_path": str(tmp_path / "idx")},
        }
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


class _StubIndexer:
    def __init__(self, calls):
        self._calls = calls

    def build(self, **kwargs):
        self._calls["build_kwargs"] = kwargs
        return True


def test_index_command_resolves_via_from_config(tmp_path, monkeypatch):
    """The command must ask IndexFactory.from_config() for the indexer."""
    calls = {}
    monkeypatch.setattr(
        IndexFactory,
        "from_config",
        staticmethod(lambda ic: (calls.__setitem__("config", ic), _StubIndexer(calls))[1]),
    )
    cfg_path = _make_config(tmp_path)

    result = runner.invoke(app, ["index", str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert "config" in calls, "IndexFactory.from_config was never called"
    assert calls["config"].type == "sparse"


def test_index_command_passes_index_path_to_build(tmp_path, monkeypatch):
    """build() must receive the configured index_path."""
    calls = {}
    monkeypatch.setattr(
        IndexFactory,
        "from_config",
        staticmethod(lambda ic: _StubIndexer(calls)),
    )
    cfg_path = _make_config(tmp_path)

    result = runner.invoke(app, ["index", str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert calls["build_kwargs"]["index_path"] == str(tmp_path / "idx")


def test_index_command_dense_requires_backend(tmp_path, monkeypatch):
    """type=dense without backend must fail with a clear message, not AttributeError."""
    calls = {}
    monkeypatch.setattr(
        IndexFactory,
        "from_config",
        staticmethod(
            lambda ic: (_ for _ in ()).throw(ValueError("index.backend is required"))
        ),
    )
    corpus = tmp_path / "corpus.json"
    corpus.write_text('[{"doc_id": "d1", "content": "x", "source": "s"}]')
    cfg = {
        "pipeline": {
            "name": "dense-missing-backend",
            "data": {"collection_path": str(corpus), "collection_format": "json"},
            "index": {"type": "dense", "index_path": str(tmp_path / "idx")},
        }
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    result = runner.invoke(app, ["index", str(cfg_path)])

    assert result.exit_code == 1
    assert "backend" in result.output.lower()
