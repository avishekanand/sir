"""Config tools: create, read, edit, validate and visualize pipeline YAML."""

import copy
import difflib
import io
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError
from rich.console import Console

from mcp.server.mcpserver import MCPServer

from ragtune.cli.config_loader import ConfigLoader
from ragtune.cli.main import default_config_template
from ragtune.mcp._common import DESTRUCTIVE, READ_ONLY, WRITES, ConfigInput, add_tools, parse_config_input
from ragtune.mcp.state import ServerState


def _dump(config: Dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


def _walk(config: Dict[str, Any], dotted: str, create: bool):
    """Return (container, last_key) for a dotted path; integer segments index lists."""
    parts = dotted.split(".")
    node: Any = config
    for i, part in enumerate(parts[:-1]):
        key: Any = int(part) if isinstance(node, list) else part
        try:
            node = node[key]
        except (KeyError, IndexError, TypeError):
            if not create or isinstance(node, list):
                raise KeyError(f"{'.'.join(parts[:i + 1])!r} does not exist in the config") from None
            node[key] = {}
            node = node[key]
    last = parts[-1]
    return node, int(last) if isinstance(node, list) else last


def register(mcp: MCPServer, state: ServerState) -> None:
    def write_yaml(path: str, config: Dict[str, Any], overwrite: bool) -> str:
        target = state.resolve(path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"{path!r} exists; pass overwrite=True to replace it")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_dump(config))
        return state.relative(target)

    def problems_of(config: Dict[str, Any], allow_missing_index: bool) -> List[str]:
        try:
            return ConfigLoader.find_problems(config, allow_missing_index, base_dir=state.root)
        except ValidationError as e:
            return [f"Schema: {err['loc']}: {err['msg']}" for err in e.errors()]

    def config_template(output_path: Optional[str] = None, overwrite: bool = False) -> Dict[str, Any]:
        """The starter config from `ragtune init`. Pass output_path to also write it as YAML.

        Adapt it with update_config, then check it with validate_config.
        """
        config = default_config_template()
        result: Dict[str, Any] = {"config": config, "yaml": _dump(config)}
        if output_path:
            result["written_to"] = write_yaml(output_path, config, overwrite)
        return result

    def read_config(config_path: str) -> Dict[str, Any]:
        """Load a pipeline YAML file from the workspace."""
        return {"config": parse_config_input(state.resolve, config_path=config_path)}

    def write_config(
        config_path: str, config: ConfigInput, overwrite: bool = False, allow_missing_index: bool = True
    ) -> Dict[str, Any]:
        """Write a config (dict or YAML text) to a file, reporting any validation problems.

        Problems do not block the write, so a work-in-progress config can be saved.
        """
        data = parse_config_input(state.resolve, config=config)
        return {"written_to": write_yaml(config_path, data, overwrite),
                "problems": problems_of(data, allow_missing_index)}

    def update_config(
        updates: Optional[Dict[str, Any]] = None,
        remove: Optional[List[str]] = None,
        config_path: Optional[str] = None,
        config: Optional[ConfigInput] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set or remove values by dotted path and return the result with a unified diff.

        Example updates: {"pipeline.components.reranker.type": "cross-encoder",
        "pipeline.budget.limits.rerank_docs": 20}. Integer segments index lists
        ("pipeline.components.estimator.0.type"). The non-interactive
        equivalent of `ragtune visualize --edit`. Writes to output_path when
        given; with only config_path, the file is updated in place.
        """
        original = parse_config_input(state.resolve, config_path, config)
        modified = copy.deepcopy(original)
        for dotted, value in (updates or {}).items():
            node, key = _walk(modified, dotted, create=True)
            node[key] = value
        for dotted in remove or []:
            node, key = _walk(modified, dotted, create=False)
            try:
                del node[key]
            except (KeyError, IndexError):
                raise KeyError(f"{dotted!r} does not exist in the config") from None
        diff = "".join(difflib.unified_diff(
            _dump(original).splitlines(keepends=True), _dump(modified).splitlines(keepends=True),
            fromfile="original", tofile="modified",
        ))
        result: Dict[str, Any] = {"config": modified, "diff": diff}
        target = output_path or config_path
        if target:
            result["written_to"] = write_yaml(target, modified, overwrite=True)
        return result

    def validate_config(
        config_path: Optional[str] = None,
        config: Optional[ConfigInput] = None,
        allow_missing_index: bool = False,
    ) -> Dict[str, Any]:
        """Run `ragtune validate`: schema, registered component types, index path existence."""
        problems = problems_of(parse_config_input(state.resolve, config_path, config), allow_missing_index)
        return {"valid": not problems, "problems": problems}

    def visualize_config(config_path: Optional[str] = None, config: Optional[ConfigInput] = None) -> Dict[str, Any]:
        """ASCII flow diagram of the pipeline and its budget, as `ragtune visualize` prints it."""
        from ragtune.cli.visualize import PipelineFlowRenderer

        console = Console(file=io.StringIO(), record=True, width=110)
        console.print(PipelineFlowRenderer(parse_config_input(state.resolve, config_path, config)).render())
        return {"diagram": console.export_text()}

    add_tools(mcp, READ_ONLY, read_config, validate_config, visualize_config)
    add_tools(mcp, WRITES, config_template)
    add_tools(mcp, DESTRUCTIVE, write_config, update_config)

    @mcp.resource("ragtune://config/template", mime_type="application/yaml")
    def config_template_resource() -> str:
        """The starter pipeline config written by `ragtune init`."""
        return _dump(default_config_template())
