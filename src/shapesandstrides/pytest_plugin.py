"""Pytest integration: marked kernels become collected tests.

Registered as a `pytest11` entry point, so installing the package is enough —
there is no plugin to enable and no conftest to write.

Kernels do not live in test files, so we collect them from their own modules.
By default any file matching ``*kernels*.py`` is scanned for `@verify` marks;
override with the ``sas_kernel_files`` ini option. This mirrors how pytest
finds ``test_*.py``: a convention, not magic, and configurable when the
convention does not fit.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

DEFAULT_PATTERNS = ["*kernels*.py", "*_kernel.py"]


def _fmt(x: float | None) -> str:
    """``None`` is a real answer here, not a missing one: a shape mismatch or
    an all-non-finite output has no error magnitude to report."""
    return "n/a" if x is None else f"{x:.3e}"


def pytest_addoption(parser):
    parser.addini(
        "sas_kernel_files",
        "Glob patterns for files holding @verify-marked kernels.",
        type="args",
        default=DEFAULT_PATTERNS,
    )
    group = parser.getgroup("shapesandstrides")
    group.addoption(
        "--sas-device",
        action="store",
        default=None,
        help="Override the device declared by @verify (e.g. cpu).",
    )
    group.addoption(
        "--sas-tier",
        action="store",
        default=None,
        help="Override the shape tier declared by @verify.",
    )


def _patterns(config) -> list[str]:
    return list(config.getini("sas_kernel_files")) or DEFAULT_PATTERNS


def pytest_collect_file(file_path: Path, parent):
    # Never shadow ordinary test collection.
    if file_path.name.startswith("test_") or file_path.name.endswith("_test.py"):
        return None
    if any(fnmatch.fnmatch(file_path.name, p) for p in _patterns(parent.config)):
        return KernelFile.from_parent(parent, path=file_path)
    return None


class KernelVerificationError(AssertionError):
    """A marked kernel failed its check."""


class KernelFile(pytest.File):
    def collect(self):
        from shapesandstrides.verify import _import_file, discover_in_module

        try:
            module = _import_file(Path(str(self.path)))
        except Exception as e:
            # A kernel file we cannot import is not a kernel file we verified,
            # so this surfaces rather than being quietly skipped.
            raise pytest.Collector.CollectError(
                f"could not import {self.path}: {type(e).__name__}: {e}"
            ) from e

        for name, fn in discover_in_module(module):
            yield KernelItem.from_parent(self, name=name, kernel=fn)


class KernelItem(pytest.Item):
    def __init__(self, *, kernel, **kw):
        super().__init__(**kw)
        self.kernel = kernel
        self.report_obj = None

    def runtest(self):
        from shapesandstrides.shapes import ShapeTier
        from shapesandstrides.verify import verify_kernel

        device = self.config.getoption("--sas-device")
        tier_opt = self.config.getoption("--sas-tier")
        tier = ShapeTier(tier_opt) if tier_opt else None

        report = verify_kernel(self.kernel, device=device, tier=tier)
        self.report_obj = report
        if not report.passed:
            raise KernelVerificationError(report)

    def repr_failure(self, excinfo, style=None):
        if not isinstance(excinfo.value, KernelVerificationError):
            return super().repr_failure(excinfo, style=style)
        try:
            return self._describe(excinfo.value.args[0])
        except Exception as e:  # pragma: no cover - defensive
            # A bug in formatting must never take down the run. Losing detail
            # is survivable; losing the whole gate to a traceback is not.
            return (
                f"kernel {self.name!r} is INCORRECT "
                f"(could not render detail: {type(e).__name__}: {e})"
            )

    def _describe(self, r) -> str:
        lines = [
            f"kernel {self.name!r} is INCORRECT",
            # Tier first: it is what says how much a verdict means.
            f"  oracle       {r.oracle_tier.value}:{r.oracle_kind.value}:{r.oracle_label}",
            f"  shapes       {r.total - r.failed_count}/{r.total} passed",
        ]
        m = r.minimal_failure
        if m is not None:
            lines.append(f"  minimal case {m.spec.label}  seed={m.seed}")
            if m.error:
                lines.append(f"  error        {m.error}")
            elif m.oracle is not None:
                o = m.oracle
                if o.shape_mismatch:
                    lines.append("  cause        output shape did not match the reference")
                else:
                    lines.append(
                        f"  mismatched   {o.mismatch_count}/{o.total_elements} elements"
                    )
                    lines.append(
                        f"  worst        max_abs_error={_fmt(o.max_abs_error)}"
                        f"  max_rel_error={_fmt(o.max_rel_error)}"
                    )
                if o.has_nan or o.has_inf:
                    flags = ", ".join(
                        n for n, on in (("NaN", o.has_nan), ("Inf", o.has_inf)) if on
                    )
                    lines.append(f"  contains     {flags}")
            lines.append(f"  replay       {r.replay_command}")
        return "\n".join(lines)

    def reportinfo(self):
        return self.path, 0, f"kernel: {self.name}"
