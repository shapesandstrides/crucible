"""Run records on disk.

The schema is designed as if it will sync to a server, because it will.
Records are plain JSON so a third party holding only the file can read it
without our code.
"""

import os
import threading
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from shapesandstrides.correctness import CorrectnessReport
from shapesandstrides.metrics import DeviceInfo, DispatchTrace, MemoryMetrics, RuntimeContext
from shapesandstrides.types import ComparisonResult, TimingResult

SCHEMA_VERSION = 1


class Provenance(BaseModel):
    """How a record was produced.

    Harmless while records are local. It matters the moment they sync,
    because a shared catalog needs to know which numbers its own code
    computed and which arrived from somewhere else.
    """

    entry_point: str = "library"
    attested: bool = True
    tool_version: str = "0.0.1"


class RunRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    timestamp: float = Field(default_factory=time.time)
    kernel_name: str
    kernel_hash: str
    variant: str | None = None
    tags: list[str] = []

    provenance: Provenance = Field(default_factory=Provenance)
    device: DeviceInfo = Field(default_factory=DeviceInfo)
    memory: MemoryMetrics = Field(default_factory=MemoryMetrics)
    context: RuntimeContext = Field(default_factory=RuntimeContext)
    dispatch: DispatchTrace = Field(default_factory=DispatchTrace)

    correctness: CorrectnessReport | None = None
    timing: TimingResult | None = None
    comparison: ComparisonResult | None = None

    duration_s: float | None = None
    notes: str | None = None


_id_lock = threading.Lock()
_last_id_ns = 0


def new_run_id() -> str:
    """Chronologically sortable, collision-resistant.

    ``time.time_ns()`` resolution is coarser than a tight call loop on some
    platforms (observed on Windows), so two calls can land on the same
    nanosecond. A random suffix alone would then break ties in an order
    unrelated to generation order, defeating "ids must sort chronologically".
    A per-process monotonic counter guarantees each id's timestamp component
    strictly increases, so lexicographic sort always matches call order.
    """
    global _last_id_ns
    with _id_lock:
        now = time.time_ns()
        if now <= _last_id_ns:
            now = _last_id_ns + 1
        _last_id_ns = now
    return f"run-{now:019d}-{uuid.uuid4().hex[:6]}"


def default_root() -> Path:
    return Path(os.environ.get("SHAPESANDSTRIDES_HOME", Path.home() / ".shapesandstrides"))


def _runs_dir(root: Path | None) -> Path:
    return (Path(root) if root else default_root()) / "runs"


def save_run(record: RunRecord, root: Path | None = None) -> Path:
    d = _runs_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{record.run_id}.json"
    # Write to a temp file and replace, so an interrupted write cannot
    # leave a half-record that later reads treat as corrupt.
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(record.model_dump_json(indent=2))
    tmp.replace(p)
    return p


class CorruptRecordError(RuntimeError):
    """A stored run file failed schema validation.

    Distinct from FileNotFoundError so a caller (the CLI in particular)
    can tell "no such run" from "this run's JSON is corrupt or from an
    incompatible schema version" instead of letting a bare ValidationError
    traceback out.
    """


def load_run(run_id: str, root: Path | None = None) -> RunRecord:
    p = _runs_dir(root) / f"{run_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"no run {run_id!r} under {_runs_dir(root)}")
    try:
        return RunRecord.model_validate_json(p.read_text())
    except ValidationError as e:
        raise CorruptRecordError(f"run record at {p} failed validation: {e}") from e


def list_runs(root: Path | None = None, limit: int | None = None) -> list[RunRecord]:
    """Newest first. A corrupt file is skipped, never fatal."""
    d = _runs_dir(root)
    if not d.exists():
        return []
    out: list[RunRecord] = []
    for p in sorted(d.glob("*.json"), reverse=True):
        try:
            out.append(RunRecord.model_validate_json(p.read_text()))
        except Exception:
            continue
        if limit is not None and len(out) >= limit:
            break
    return out
