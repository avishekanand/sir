"""Per-server state: workspace root and object handles."""

import itertools
import threading
from pathlib import Path
from typing import Any, Dict


class ServerState:
    """Everything one server instance owns. Tests build a fresh one per server."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.pipelines: Dict[str, Any] = {}
        self.datasets: Dict[str, Any] = {}
        self.import_errors: Dict[str, str] = {}  # optional modules that failed to load
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def resolve(self, path: str, must_exist: bool = False) -> Path:
        """Resolve a path against the workspace root, refusing anything outside it."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"{path!r} is outside the workspace root {self.root}")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"{path!r} does not exist (resolved to {candidate})")
        return candidate

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.root)) if path != self.root else "."

    def new_id(self, prefix: str) -> str:
        with self._lock:
            return f"{prefix}-{next(self._ids)}"

    def lookup(self, table: Dict[str, Any], kind: str, handle: str) -> Any:
        if handle not in table:
            raise KeyError(f"Unknown {kind} {handle!r}. Open: {sorted(table)}")
        return table[handle]
