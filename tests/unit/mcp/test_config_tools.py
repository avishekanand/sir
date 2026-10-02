"""ragtune.mcp config tools: template, read/write, dotted updates, validation, visualization."""

import yaml

VALID = {"pipeline": {"name": "t", "components": {
    "retriever": {"type": "in-memory", "params": {"documents": []}},
    "reranker": {"type": "noop"},
}}}


def test_template_matches_ragtune_init_and_can_be_written(mcp_server):
    from ragtune.cli.main import default_config_template

    result = mcp_server.call("config_template", output_path="configs/new.yaml")
    assert result["config"] == default_config_template()
    assert result["written_to"] == "configs/new.yaml"
    assert yaml.safe_load((mcp_server.root / "configs/new.yaml").read_text()) == default_config_template()
    assert "exists" in mcp_server.error("config_template", output_path="configs/new.yaml")


def test_update_config_sets_removes_and_diffs(mcp_server):
    mcp_server.call("write_config", config_path="p.yaml", config=VALID)
    result = mcp_server.call(
        "update_config", config_path="p.yaml",
        updates={"pipeline.components.reranker.type": "simulated", "pipeline.budget.limits.rerank_docs": 3},
        remove=["pipeline.name"],
    )
    assert "-      type: noop" in result["diff"] and "+      type: simulated" in result["diff"]
    on_disk = yaml.safe_load((mcp_server.root / "p.yaml").read_text())
    assert on_disk["pipeline"]["budget"]["limits"]["rerank_docs"] == 3
    assert "name" not in on_disk["pipeline"]
    assert "does not exist" in mcp_server.error("update_config", config=VALID, remove=["pipeline.nope"])


def test_update_config_indexes_lists(mcp_server):
    config = {"pipeline": {"components": {"estimator": [{"type": "baseline"}, {"type": "utility"}]}}}
    result = mcp_server.call("update_config", config=config, updates={"pipeline.components.estimator.1.type": "similarity"})
    assert result["config"]["pipeline"]["components"]["estimator"][1]["type"] == "similarity"
    assert "written_to" not in result  # inline config: nothing is written


def test_validate_reports_registry_schema_and_index_problems(mcp_server):
    assert mcp_server.call("validate_config", config=VALID) == {"valid": True, "problems": []}

    bad = {"pipeline": {"components": {"reranker": {"type": "nope"}},
                        "index": {"params": {"index_path": "missing_index"}}}}
    problems = mcp_server.call("validate_config", config=yaml.safe_dump(bad))["problems"]
    assert any("'nope' not found in registry for category 'reranker'" in p for p in problems)
    assert any("missing_index" in p for p in problems)
    assert not any("missing_index" in p for p in mcp_server.call(
        "validate_config", config=bad, allow_missing_index=True)["problems"])

    schema = mcp_server.call("validate_config", config={"pipeline": {"budget": {"limits": "lots"}}})
    assert schema["valid"] is False and schema["problems"][0].startswith("Schema:")


def test_visualize_and_config_input_errors(mcp_server):
    diagram = mcp_server.call("visualize_config", config=VALID)["diagram"]
    assert "RETRIEVER" in diagram and "RAGtune Pipeline: t" in diagram
    assert "exactly one" in mcp_server.error("validate_config")
    assert "outside the workspace root" in mcp_server.error("read_config", config_path="../x.yaml")


def test_template_resource(mcp_server):
    text = mcp_server.session(lambda c: c.read_resource("ragtune://config/template")).contents[0].text
    assert yaml.safe_load(text)["pipeline"]["name"] == "My First RAGtune Pipeline"
