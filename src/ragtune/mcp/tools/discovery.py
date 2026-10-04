"""Discovery tools: what is installed, registered and configured."""

import importlib.metadata
import importlib.util
import inspect
import platform
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

from ragtune.mcp._common import READ_ONLY, WRITES, add_tools, to_jsonable
from ragtune.mcp.state import ServerState
from ragtune.registry import registry
from ragtune.utils.config import config as runtime_config

OPTIONAL_PACKAGES = (
    "pyterrier", "pyterrier_dr", "faiss", "pytrec_eval", "sentence_transformers",
    "torch", "litellm", "datasets", "ir_datasets", "freshstack", "querygym",
)


def _category(name: str) -> str:
    categories = registry.list_all()
    key = name.lower().rstrip("s") if name.lower() not in categories else name.lower()
    if key not in categories:
        raise ValueError(f"Unknown category {name!r}. Valid: {sorted(categories)}")
    return key


def _parameters(cls: Any) -> List[Dict[str, Any]]:
    target = cls.__init__ if inspect.isclass(cls) else cls
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return []
    params = []
    for p in signature.parameters.values():
        if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        entry: Dict[str, Any] = {"name": p.name, "required": p.default is p.empty}
        if p.annotation is not p.empty:
            entry["type"] = inspect.formatannotation(p.annotation)
        if p.default is not p.empty:
            entry["default"] = to_jsonable(p.default)
        params.append(entry)
    return params


def _own_doc(cls: Any) -> str:
    """The component's own docstring; inspect.getdoc would inherit e.g. ABC's."""
    return inspect.cleandoc(vars(cls).get("__doc__") or "") if inspect.isclass(cls) else inspect.getdoc(cls) or ""


def _summary(cls: Any) -> str:
    return _own_doc(cls).split("\n")[0]


def register(mcp: MCPServer, state: ServerState) -> None:
    def server_info() -> Dict[str, Any]:
        """Versions, workspace root, optional dependencies, and open handles/jobs.

        Call this first: 'optional_packages' shows which retrievers, indexers
        and evaluators can actually run in this environment.
        """
        def version(dist: str) -> Optional[str]:
            try:
                return importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                return None

        return {
            "ragtune_version": version("ragtune"),
            "mcp_version": version("mcp"),
            "python": platform.python_version(),
            "workspace_root": str(state.root),
            "optional_packages": {m: importlib.util.find_spec(m) is not None for m in OPTIONAL_PACKAGES},
            "registry_import_errors": state.import_errors,
            "open_pipelines": sorted(state.pipelines),
            "loaded_datasets": sorted(state.datasets),
            "jobs": [j.id for j in state.jobs.list()],
        }

    def list_components(category: Optional[str] = None) -> Dict[str, Any]:
        """List registered components with their constructor parameters.

        Categories: retriever, reranker, reformulator, assembler, scheduler,
        estimator, feedback, indexer. The 'name' is the 'type' string used in
        pipeline configs, e.g. components.reranker.type: "cross-encoder".
        """
        all_components = registry.list_all()
        categories = [_category(category)] if category else sorted(all_components)
        return {
            cat: [
                {"name": name, "summary": _summary(cls), "parameters": _parameters(cls)}
                for name, cls in sorted(all_components[cat].items())
            ]
            for cat in categories
        }

    def describe_component(category: str, name: str) -> Dict[str, Any]:
        """Full docstring, source location and constructor parameters of one component."""
        cat = _category(category)
        components = registry.list_all()[cat]
        if name not in components:
            raise ValueError(f"No {cat} named {name!r}. Available: {sorted(components)}")
        cls = components[name]
        try:
            source = inspect.getsourcefile(cls)
        except TypeError:
            source = None
        return {
            "category": cat,
            "name": name,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "source_file": source,
            "doc": _own_doc(cls),
            "parameters": _parameters(cls),
        }

    def get_settings(key: Optional[str] = None, prompts: bool = False) -> Dict[str, Any]:
        """Read runtime defaults (config/defaults.yaml) or prompt templates (config/prompts.yaml).

        'key' is a dotted path such as 'retrieval.max_pool_size'. These values
        steer the controller, e.g. retrieval.original_query_depth is the
        first-stage depth when a pipeline sets no initial_top_k.
        """
        source = runtime_config._prompts if prompts else runtime_config._config
        if key is None:
            return {"settings": to_jsonable(source)}
        getter = runtime_config.get_prompt if prompts else runtime_config.get
        missing = object()
        value = getter(key, missing)
        if value is missing:
            raise KeyError(f"No setting {key!r}. Top-level keys: {sorted(source)}")
        return {"key": key, "value": to_jsonable(value)}

    def set_setting(key: str, value: Any) -> Dict[str, Any]:
        """Override a runtime default for this server process, e.g. retrieval.max_pool_size=100.

        Affects every pipeline run afterwards; it is not written to disk.
        """
        previous = runtime_config.get(key)
        runtime_config.set(key, value)
        return {"key": key, "previous": to_jsonable(previous), "value": to_jsonable(value)}

    def list_default_scenarios() -> Dict[str, Any]:
        """The 7 built-in benchmark scenarios (BM25 baseline + 6 cross-encoder variants).

        Each entry is a pipeline config without a retriever, ready for
        evaluate_scenarios or create_pipeline.
        """
        from ragtune.cli.config_loader import ConfigLoader

        return {"scenarios": ConfigLoader._default_scenarios()}

    add_tools(mcp, READ_ONLY, server_info, list_components, describe_component, get_settings, list_default_scenarios)
    add_tools(mcp, WRITES, set_setting)

    def job_status(job_id: str, tail_lines: int = 40) -> Dict[str, Any]:
        """Status of a background job; includes the result (tasks) or a log tail (processes)."""
        return state.jobs.get(job_id).describe(tail_lines=tail_lines)

    def list_jobs() -> Dict[str, Any]:
        """All background jobs started by this server, newest last."""
        return {"jobs": [job.describe() for job in state.jobs.list()]}

    def cancel_job(job_id: str) -> Dict[str, Any]:
        """Stop a running subprocess job (scripts, tests, CLI). In-process tasks cannot be stopped."""
        return state.jobs.cancel(job_id).describe()

    add_tools(mcp, READ_ONLY, job_status, list_jobs)
    add_tools(mcp, WRITES, cancel_job)
