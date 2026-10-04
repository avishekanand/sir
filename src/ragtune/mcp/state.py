"""Per-server state: workspace root, object handles, and background jobs."""

import itertools
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ragtune.mcp._common import to_jsonable

FINISHED = ("succeeded", "failed", "cancelled", "timed_out")


@dataclass
class Job:
    id: str
    kind: str  # "task" (in-process thread) or "process" (subprocess)
    description: str
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    returncode: Optional[int] = None
    log_path: Optional[Path] = None
    process: Optional[subprocess.Popen] = field(default=None, repr=False)

    def finish(self, status: str, **updates: Any) -> None:
        if self.status in FINISHED:  # a cancel already settled this job
            return
        for key, value in updates.items():
            setattr(self, key, value)
        self.status = status
        self.finished_at = time.time()

    def describe(self, tail_lines: int = 0) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "job_id": self.id,
            "kind": self.kind,
            "description": self.description,
            "status": self.status,
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 2),
        }
        if self.returncode is not None:
            info["returncode"] = self.returncode
        if self.error:
            info["error"] = self.error
        if self.status == "succeeded" and self.kind == "task":
            info["result"] = to_jsonable(self.result)
        if self.log_path is not None:
            info["log_path"] = str(self.log_path)
            if tail_lines > 0 and self.log_path.exists():
                lines = self.log_path.read_text(errors="replace").splitlines()
                info["log_tail"] = "\n".join(lines[-tail_lines:])
        return info


class JobManager:
    """Runs long work off the request path so MCP calls return immediately."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self._jobs: Dict[str, Job] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def _new(self, kind: str, description: str) -> Job:
        with self._lock:
            job = Job(id=f"job-{next(self._ids)}", kind=kind, description=description)
            self._jobs[job.id] = job
        return job

    def start_task(self, description: str, fn: Callable[[], Any]) -> Job:
        job = self._new("task", description)

        def run():
            try:
                job.finish("succeeded", result=fn())
            except Exception as e:
                job.finish("failed", error=f"{type(e).__name__}: {e}")

        threading.Thread(target=run, name=job.id, daemon=True).start()
        return job

    def start_process(
        self,
        description: str,
        argv: List[str],
        cwd: Path,
        env: Optional[Dict[str, str]] = None,
        timeout_s: float = 3600,
    ) -> Job:
        job = self._new("process", description)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        job.log_path = self.log_dir / f"{job.id}.log"
        log = open(job.log_path, "w")
        job.process = subprocess.Popen(
            argv,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdin=subprocess.DEVNULL,  # interactive prompts fail fast instead of hanging
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # lets cancel/timeout stop the whole process group
        )

        def watch():
            try:
                code = job.process.wait(timeout=timeout_s)
                job.finish("succeeded" if code == 0 else "failed", returncode=code)
            except subprocess.TimeoutExpired:
                self._kill(job)
                job.finish("timed_out", returncode=job.process.wait(), error=f"exceeded {timeout_s}s")
            finally:
                log.close()

        threading.Thread(target=watch, name=f"{job.id}-watch", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job_id {job_id!r}. Known: {sorted(self._jobs)}")
        return self._jobs[job_id]

    def list(self) -> List[Job]:
        return list(self._jobs.values())

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status in FINISHED:
            return job
        if job.kind != "process":
            raise ValueError("In-process tasks cannot be interrupted; wait for them to finish.")
        self._kill(job)
        job.finish("cancelled")
        return job

    @staticmethod
    def _kill(job: Job) -> None:
        try:
            os.killpg(job.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


class ServerState:
    """Everything one server instance owns. Tests build a fresh one per server."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.pipelines: Dict[str, Any] = {}
        self.datasets: Dict[str, Any] = {}
        self.jobs = JobManager(self.root / "logs" / "mcp_jobs")
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

    def run_or_background(self, background: bool, description: str, fn: Callable[[], Any]) -> Any:
        """Run fn now, or as a background job whose result job_status returns."""
        if not background:
            return fn()
        job = self.jobs.start_task(description, fn)
        return {"job_id": job.id, "status": job.status, "next": "poll job_status(job_id)"}
