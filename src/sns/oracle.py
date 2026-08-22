"""The correctness oracle.

The reference is computed in float64 on CPU and cast to the target dtype.
It is never the PyTorch GPU op: that shares numerics, kernels and bugs with
the thing under test, so agreeing with it proves much less than it appears
to.
"""

from typing import Callable

from pydantic import BaseModel

from sns.shapes import ShapeSpec

_DTYPES = {
    "float64": "float64",
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
}


class OracleResult(BaseModel):
    passed: bool
    # None for a shape mismatch or when every element is non-finite: no
    # error magnitude exists to report. float("inf") was used previously,
    # but pydantic serialises inf to JSON null, and validation then rejects
    # null for a required float — the run saved cleanly and then vanished
    # from the catalog on load. None round-trips honestly instead.
    max_abs_error: float | None
    max_rel_error: float | None
    mismatch_count: int
    total_elements: int
    first_mismatch_index: int | None = None
    has_nan: bool = False
    has_inf: bool = False
    shape_mismatch: bool = False


def _torch_dtype(name: str):
    import torch

    if name not in _DTYPES:
        raise KeyError(f"unsupported dtype {name!r}")
    return getattr(torch, name)


def make_inputs(spec: ShapeSpec, seed: int, n_inputs: int = 2, device: str = "cpu"):
    """Deterministic inputs for a shape. Same seed, same tensors, always."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    dtype = _torch_dtype(spec.dtype)
    out = []
    for i in range(n_inputs):
        if spec.layout == "noncontiguous":
            # Allocate wider than needed and slice, so strides are real
            # rather than simulated by a transpose of a square. Cast to the
            # target dtype *before* slicing: `.to(dtype)` materializes a
            # fresh contiguous copy, which would silently undo the stride
            # trick if applied after the slice. Move to the target device
            # before slicing too: `.to(device)` on an already-strided tensor
            # copies it into a fresh contiguous allocation on the new
            # device, which would just as silently undo the stride trick at
            # the device boundary instead.
            padded = (
                torch.randn(
                    (*spec.dims[:-1], spec.dims[-1] * 2),
                    generator=gen,
                    dtype=torch.float32,
                )
                .to(dtype)
                .to(device)
            )
            t = padded[..., ::2]
        else:
            t = torch.randn(spec.dims, generator=gen, dtype=torch.float32).to(dtype).to(device)
        out.append(t)
    return out


def reference_fp64(fn: Callable, inputs: list, out_dtype) -> "object":
    """Compute fn in float64 on CPU, then cast to the target dtype.

    Fused kernels — the stated target audience — commonly return several
    tensors (a primary output plus a saved mean, rstd, index buffer, and so
    on). A tuple or list result is cast element-wise so that path is
    actually reachable through check(); a bare `.to()` on a tuple would
    raise AttributeError before compare_outputs' multi-output logic ever ran.
    """
    import torch

    fp64_inputs = [t.detach().to("cpu", dtype=torch.float64) for t in inputs]
    result = fn(*fp64_inputs)
    if isinstance(result, (tuple, list)):
        return type(result)(r.to(out_dtype) for r in result)
    return result.to(out_dtype)


def compare_against_oracle(actual, expected, atol: float, rtol: float) -> OracleResult:
    """Adjudicate one output against the oracle. We compute the verdict."""
    import torch

    if tuple(actual.shape) != tuple(expected.shape):
        return OracleResult(
            passed=False,
            max_abs_error=None,
            max_rel_error=None,
            mismatch_count=-1,
            total_elements=expected.numel(),
            shape_mismatch=True,
        )

    a = actual.detach().to("cpu", dtype=torch.float64).flatten()
    e = expected.detach().to("cpu", dtype=torch.float64).flatten()

    has_nan = bool(torch.isnan(a).any().item())
    has_inf = bool(torch.isinf(a).any().item())

    abs_err = (a - e).abs()
    # Guard the denominator so an expected value of zero doesn't produce inf.
    rel_err = abs_err / e.abs().clamp_min(1e-30)

    within = abs_err <= (atol + rtol * e.abs())
    if has_nan or has_inf:
        within = within & torch.isfinite(a)

    mismatches = (~within).nonzero().flatten()
    return OracleResult(
        passed=bool(within.all().item()),
        max_abs_error=float(abs_err[torch.isfinite(abs_err)].max().item())
        if torch.isfinite(abs_err).any()
        else None,
        max_rel_error=float(rel_err[torch.isfinite(rel_err)].max().item())
        if torch.isfinite(rel_err).any()
        else None,
        mismatch_count=int(mismatches.numel()),
        total_elements=int(e.numel()),
        first_mismatch_index=int(mismatches[0].item()) if mismatches.numel() else None,
        has_nan=has_nan,
        has_inf=has_inf,
    )


def _as_tuple(x):
    return x if isinstance(x, (tuple, list)) else (x,)


def compare_outputs(actual, expected, atol: float, rtol: float) -> list[OracleResult]:
    """Adjudicate every output a kernel returns, not just the first.

    Fused kernels commonly return extra tensors — a saved mean and rstd for
    the backward pass, an index buffer, a scale factor. Checking only the
    primary output lets a bug in any of them reach production.
    """
    a, e = _as_tuple(actual), _as_tuple(expected)
    if len(a) != len(e):
        return [
            OracleResult(
                passed=False,
                max_abs_error=None,
                max_rel_error=None,
                mismatch_count=-1,
                total_elements=0,
                shape_mismatch=True,
            )
        ]
    return [compare_against_oracle(x, y, atol=atol, rtol=rtol) for x, y in zip(a, e)]
