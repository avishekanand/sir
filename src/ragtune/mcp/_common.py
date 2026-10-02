"""Helpers shared by every RAGtune MCP tool module."""

import copy
import dataclasses
import enum
import functools
import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import yaml
from pydantic import BaseModel

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

READ_ONLY = ToolAnnotations(readOnlyHint=True)
WRITES = ToolAnnotations(readOnlyHint=False, destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True)

ConfigInput = Union[Dict[str, Any], str]


def tool_errors(fn: Callable) -> Callable:
    """Re-raise any failure as ToolError so its message reaches the agent.

    MCPServer only forwards the text of ToolError; anything else is reported
    as a bare "Error executing tool", leaving the caller nothing to fix.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    return wrapper


def add_tools(mcp: MCPServer, annotations: ToolAnnotations, *fns: Callable) -> None:
    """Register functions as tools; the name and docstring become the tool's."""
    for fn in fns:
        mcp.add_tool(tool_errors(fn), annotations=annotations)


def to_jsonable(obj: Any) -> Any:
    """Convert RAGtune/pydantic/numpy/dataclass values into plain JSON types."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, BaseModel):
        return to_jsonable(obj.model_dump())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):  # numpy scalars and arrays
        return to_jsonable(obj.tolist())
    return repr(obj)


def parse_config_input(
    resolve: Callable[..., Path],
    config_path: Optional[str] = None,
    config: Optional[ConfigInput] = None,
) -> Dict[str, Any]:
    """Return a pipeline config from exactly one of a file path or an inline dict/YAML string."""
    if (config_path is None) == (config is None):
        raise ValueError("Pass exactly one of 'config_path' or 'config'.")
    if config_path is not None:
        data = yaml.safe_load(resolve(config_path, must_exist=True).read_text())
    elif isinstance(config, str):
        data = yaml.safe_load(config)
    else:
        data = copy.deepcopy(config)
    if not isinstance(data, dict):
        raise ValueError("A config must be a mapping, e.g. {'pipeline': {...}}.")
    return data


def merge_limits(
    limits: Dict[str, float], overrides: Optional[Dict[str, Optional[float]]]
) -> Dict[str, float]:
    """Apply limit overrides; a None value removes that limit."""
    merged = dict(limits)
    for key, value in (overrides or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = float(value)
    return merged
