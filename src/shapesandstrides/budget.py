"""Grade a kernel by the error it is allowed, not against a fixed tolerance.

A fused kernel deliberately disagrees with the unfused chain it replaces. It
keeps intermediates in registers at higher precision instead of round-tripping
them through low-precision memory, and it accumulates in a different order. The
result is frequently *closer to the truth* than the chain it replaced.

`torch.allclose` cannot tell that apart from a bug, and
`tolerance.tolerance_for` handles it by widening ``atol``/``rtol`` when
``fused_ops`` is supplied -- an informed guess, but a guess.

This module measures instead. Given three tensors it asks the question that
actually matters:

    is the kernel at least as accurate as the code it replaces?

======  ====================================================================
golden  the computation in float64 -- as close to truth as is cheap to get
ref     the unfused chain at production precision, rounding error and all
kernel  the fused kernel under test
======  ====================================================================

Both distances are measured against ``golden``, and the kernel passes when its
error is no larger than the reference's by more than a stated margin.
"""

from __future__ import annotations

from pydantic import BaseModel


class ErrorBudget(BaseModel):
    """The outcome of one error-budget comparison.

    Six error figures rather than one, per rule 5: a kernel that is right
    almost everywhere and catastrophically wrong in one corner is the
    interesting failure, and a mean hides it.
    """

    passed: bool

    kernel_p50_ulp: float
    kernel_p99_ulp: float
    kernel_max_ulp: float

    reference_p50_ulp: float
    reference_p99_ulp: float
    reference_max_ulp: float

    # kernel_p99 / reference_p99. None when the reference is bit-exact: there
    # is no ratio to report, and inf does not survive a JSON round trip --
    # pydantic writes it as null and then rejects null for a required float.
    # See the same trap documented on OracleResult.max_abs_error.
    ratio_p99: float | None

    # Surfaced rather than hidden in the tool: "twice as noisy is still fine"
    # is a judgement a reader is entitled to disagree with.
    margin: float

    total_elements: int
    shape_mismatch: bool = False
    # Set when the kernel emitted NaN or Inf. A separate signal from a large
    # error: one is a broken kernel, the other is an inaccurate one.
    non_finite_output: bool = False


def ulp_error(actual, golden):
    """Elementwise error in units of the last place of ``actual``'s dtype.

    ULP rather than absolute difference because a fixed tolerance means
    entirely different things at different magnitudes: ``1e-3`` is absurdly
    tight around ``1e8`` and uselessly loose around ``1e-8``, and one kernel's
    inputs routinely span both. One ULP is "the smallest difference this dtype
    can represent here", which is the same statement everywhere.
    """
    import torch

    a = actual.detach().to("cpu", dtype=torch.float64).flatten()
    g = golden.detach().to("cpu", dtype=torch.float64).flatten()

    finfo = torch.finfo(actual.dtype)
    # Width of one ULP at each golden magnitude, expressed in the *output*
    # dtype -- the kernel cannot be blamed for precision it cannot represent.
    # Clamped at `tiny` so denormals and exact zeros do not divide by zero.
    magnitude = g.abs().clamp_min(finfo.tiny)
    ulp_width = torch.pow(2.0, torch.floor(torch.log2(magnitude))) * finfo.eps
    ulp_width = ulp_width.clamp_min(finfo.tiny)

    return (a - g).abs() / ulp_width


def _distribution(err) -> tuple[float, float, float]:
    """(p50, p99, max) over the finite entries. Empty input reports zeros.

    Sorts and indexes rather than calling `torch.quantile`, which raises
    ``quantile() input tensor is too large`` beyond roughly 16.7M elements
    (2**24). A 4097x4096 output is 16.78M, so the limit is reached by ordinary
    shapes in the sweep -- and it would bite hardest on exactly the large
    kernels most worth grading. Subsampling would dodge it at the cost of
    making p99 an estimate reported as if it were exact.
    """
    import torch

    finite = err[torch.isfinite(err)]
    n = int(finite.numel())
    if n == 0:
        return 0.0, 0.0, 0.0

    ordered, _ = torch.sort(finite)
    # Nearest-rank on a 0-based index, matching quantile's endpoints at the
    # extremes without pulling in its interpolation or its size ceiling.
    p50 = float(ordered[min(n - 1, int(0.50 * (n - 1) + 0.5))].item())
    p99 = float(ordered[min(n - 1, int(0.99 * (n - 1) + 0.5))].item())
    return p50, p99, float(ordered[-1].item())


def compare_error_budget(actual, reference, golden, margin: float = 2.0) -> ErrorBudget:
    """Pass if the kernel's p99 ULP error is within ``margin`` of the reference's.

    ``actual`` is the kernel output, ``reference`` the unfused chain at
    production precision (see `oracle.reference_lowp`), ``golden`` the same
    computation in float64 (see `oracle.reference_fp64`).

    ``margin`` defaults to 2.0: the kernel may be up to twice as noisy as the
    chain it replaced before it is called wrong. p99 rather than max, because a
    single outlier element should not condemn a kernel -- the max is reported
    alongside so a reader can apply a stricter rule themselves.
    """
    import torch

    total = int(golden.numel())

    if tuple(actual.shape) != tuple(golden.shape) or tuple(reference.shape) != tuple(
        golden.shape
    ):
        return ErrorBudget(
            passed=False,
            kernel_p50_ulp=0.0,
            kernel_p99_ulp=0.0,
            kernel_max_ulp=0.0,
            reference_p50_ulp=0.0,
            reference_p99_ulp=0.0,
            reference_max_ulp=0.0,
            ratio_p99=None,
            margin=margin,
            total_elements=total,
            shape_mismatch=True,
        )

    k50, k99, kmax = _distribution(ulp_error(actual, golden))
    r50, r99, rmax = _distribution(ulp_error(reference, golden))

    non_finite = not bool(
        torch.isfinite(actual.detach().to("cpu", dtype=torch.float64)).all().item()
    )

    if r99 > 0.0:
        ratio: float | None = k99 / r99
        within = ratio <= margin
    else:
        # A bit-exact reference leaves no ratio to take. Fall back to an
        # absolute ULP floor: demanding bit-equality from a differently
        # ordered accumulation is not a real requirement.
        ratio = None
        within = k99 <= margin

    return ErrorBudget(
        passed=bool(within and not non_finite),
        kernel_p50_ulp=k50,
        kernel_p99_ulp=k99,
        kernel_max_ulp=kmax,
        reference_p50_ulp=r50,
        reference_p99_ulp=r99,
        reference_max_ulp=rmax,
        ratio_p99=ratio,
        margin=margin,
        total_elements=total,
        non_finite_output=non_finite,
    )
