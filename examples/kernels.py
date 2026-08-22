"""Triton kernels used to exercise crucible end to end.

Half of these are deliberately broken, in the specific ways Triton kernels
break in the wild. They are the fixtures that prove the correctness engine
actually catches something — a test suite that only ever sees correct code
proves nothing.

Triton's @triton.jit refuses to compile from a REPL or `python -c`, so these
must live in a real file.
"""

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------
# Correct kernels
# --------------------------------------------------------------------------


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add(x, y):
    """Correct elementwise add: masked load with an explicit `other`."""
    x, y = x.contiguous(), y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    _add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return out


@triton.jit
def _mul_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * y, mask=mask)


def triton_mul(x, y):
    """Correct elementwise multiply."""
    x, y = x.contiguous(), y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    _mul_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def _autotuned_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add_autotuned(x, y):
    """Correct, and autotuned — so `discover_tiles` finds multiple configs."""
    x, y = x.contiguous(), y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    _autotuned_add_kernel[lambda m: (triton.cdiv(n, m["BLOCK"]),)](x, y, out, n)
    return out


# --------------------------------------------------------------------------
# Deliberately broken kernels
# --------------------------------------------------------------------------


@triton.jit
def _add_flat_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add_assumes_contiguous(x, y):
    """BROKEN: indexes flat storage without making the input contiguous.

    Every correct kernel here calls .contiguous() first. This one does not,
    so for a strided tensor the flat offsets walk the underlying storage in
    the wrong order and it reads the wrong elements entirely. Silent, and
    only visible if something actually tests a non-contiguous input — which
    is why the shape generator produces them.
    """
    out = torch.empty_like(x)
    n = x.numel()
    _add_flat_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return out


@triton.jit
def _add_drops_tail_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # BUG: `n - 1` instead of `n`. The final element is never written, so the
    # output keeps whatever was in the uninitialised buffer. A classic
    # forgot-the-tail bound error, and invisible at any size where the caller
    # happens not to look at the last element.
    mask = offs < n - 1
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add_drops_tail(x, y):
    """BROKEN: off-by-one bound leaves the last element unwritten."""
    x, y = x.contiguous(), y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    _add_drops_tail_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return out


@triton.jit
def _sum_fp16_accum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # BUG: accumulate in fp16 rather than fp32. Correct for small inputs and
    # progressively wrong as the reduction grows — the kind of bug that
    # passes every small test and fails in production.
    acc = tl.sum(x.to(tl.float16), axis=0)
    tl.store(out_ptr + pid, acc)


def triton_rowsum_bad_accum(x):
    """BROKEN: reduces in fp16 instead of fp32."""
    x = x.contiguous().flatten()
    n = x.numel()
    blocks = triton.cdiv(n, 1024)
    partial = torch.empty(blocks, device=x.device, dtype=x.dtype)
    _sum_fp16_accum_kernel[(blocks,)](x, partial, n, BLOCK=1024)
    return partial.sum().reshape(())


# --------------------------------------------------------------------------
# References. Plain PyTorch, used two ways: run on CPU in float64 as the
# correctness oracle, and on GPU as the timing baseline.
# --------------------------------------------------------------------------


def ref_add(x, y):
    return x + y


def ref_mul(x, y):
    return x * y


def ref_rowsum(x):
    return x.flatten().sum().reshape(())
