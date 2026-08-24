"""One line to adopt.

    @verify(against="torch.add")
    def my_add(x, y): ...

That is the whole integration. The decorator attaches a spec and returns the
function unchanged — a kernel that behaves differently because it is being
verified is a kernel nobody will annotate in production.

Everything else here exists to find those marks and act on them: in a pytest
run, or from `shapesandstrides verify <path>` in CI, where the exit code is
what turns a report nobody reads into a gate.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from shapesandstrides.correctness import CorrectnessReport, check
from shapesandstrides.reference import OracleKind, resolve
from shapesandstrides.shapes import ShapeTier

MARK = "__shapesandstrides_verify__"

# Files we never import while walking a directory. Importing a virtualenv or a
# build tree is slow at best and destructive at worst.
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "build", "dist",
    ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "site-packages", "node_modules", ".eggs",
}


@dataclass(frozen=True)
class VerifySpec:
    """What the decorator recorded. Enough to run a check, and nothing more."""

    against: object
    tier: ShapeTier = ShapeTier.FAST
    dtypes: tuple[str, ...] | None = None
    op_name: str | None = None
    device: str = "cuda"
    max_elements: int | None = None
    tolerance: tuple[float, float] | None = None
    fused_ops: tuple[str, ...] | None = None
    seed: int | None = None
    kernel_name: str = ""

    def resolved_op_name(self) -> str:
        """The tolerance profile to use.

        An explicit name wins. Otherwise a torch operator names itself —
        ``"torch.add"`` means the ``add`` profile — and anything else falls
        back to the kernel's own name, which is at least a stable label.
        """
        if self.op_name:
            return self.op_name
        ref = resolve(self.against)
        if ref.kind is OracleKind.TORCH_OP:
            return ref.label.rsplit(".", 1)[-1]
        return self.kernel_name or "unknown"


def verify(
    against: object,
    *,
    tier: ShapeTier = ShapeTier.FAST,
    dtypes: Iterable[str] | None = None,
    op_name: str | None = None,
    device: str = "cuda",
    max_elements: int | None = None,
    tolerance: tuple[float, float] | None = None,
    fused_ops: Iterable[str] | None = None,
    seed: int | None = None,
) -> Callable[[Callable], Callable]:
    """Mark a kernel for verification. Returns the function unchanged.

    ``against`` is resolved immediately rather than lazily, so a typo in an
    operator name fails when the module is imported instead of an hour into a
    CI run.
    """
    resolve(against)  # fail fast, at decoration time

    def deco(fn: Callable) -> Callable:
        setattr(
            fn,
            MARK,
            VerifySpec(
                against=against,
                tier=tier,
                dtypes=tuple(dtypes) if dtypes else None,
                op_name=op_name,
                device=device,
                max_elements=max_elements,
                tolerance=tolerance,
                fused_ops=tuple(fused_ops) if fused_ops else None,
                seed=seed,
                kernel_name=getattr(fn, "__name__", "kernel"),
            ),
        )
        return fn

    return deco


def spec_of(fn: object) -> VerifySpec | None:
    return getattr(fn, MARK, None)


def is_marked(fn: object) -> bool:
    return spec_of(fn) is not None


def verify_kernel(
    fn: Callable,
    *,
    device: str | None = None,
    tier: ShapeTier | None = None,
) -> CorrectnessReport:
    """Run the check described by `fn`'s mark.

    ``device`` and ``tier`` may be overridden at call time: the decorator is
    written once in the source tree, but the same tree runs on a laptop, a CI
    runner and a datacenter host.
    """
    spec = spec_of(fn)
    if spec is None:
        raise ValueError(
            f"{getattr(fn, '__name__', fn)!r} is not marked for verification. "
            f"Add @verify(against=...) to it."
        )

    kwargs: dict = {}
    if spec.seed is not None:
        kwargs["seed"] = spec.seed
    if spec.tolerance is not None:
        kwargs["tolerance_override"] = spec.tolerance
    if spec.fused_ops is not None:
        kwargs["fused_ops"] = list(spec.fused_ops)

    return check(
        fn,
        reference=spec.against,
        tier=tier or spec.tier,
        dtypes=list(spec.dtypes) if spec.dtypes else None,
        op_name=spec.resolved_op_name(),
        device=device or spec.device,
        max_elements=spec.max_elements,
        **kwargs,
    )


# ------------------------------------------------------------------ discovery


def discover_in_module(module) -> list[tuple[str, Callable]]:
    """Marked callables defined at module level, in declaration order."""
    found = []
    for name, obj in vars(module).items():
        if callable(obj) and is_marked(obj):
            found.append((name, obj))
    return found


def _import_file(path: Path):
    """Import a .py file by path under a unique module name.

    Deliberately simple: a file that depends on being imported as part of its
    package may fail here. That is reported as an import error against that
    file rather than silently skipped, because a kernel we could not load is
    not a kernel we verified.
    """
    mod_name = "_sas_discovered_" + path.stem + "_" + str(abs(hash(str(path.resolve()))))
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    # A kernel file usually sits next to its helpers; make those importable.
    parent = str(path.parent.resolve())
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass
    return module


def discover_in_path(target: str | Path) -> list[tuple[str, Callable]]:
    """Find marked kernels in a file or, recursively, a directory."""
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")

    files: list[Path] = []
    if path.is_file():
        files = [path]
    else:
        for p in sorted(path.rglob("*.py")):
            if SKIP_DIRS & set(p.parts):
                continue
            files.append(p)

    found: list[tuple[str, Callable]] = []
    for f in files:
        module = _import_file(f)
        found.extend(discover_in_module(module))
    return found
