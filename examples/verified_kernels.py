"""What adopting this looks like: one decorator per kernel.

    shapesandstrides verify examples/verified_kernels.py

No reference implementations are written here. `against="torch.add"` means
PyTorch's own operator is the answer key, and the reduction below is satisfied
by a one-line expression because no single torch op matches it.

Run it and check `$?`. Three of these kernels are broken on purpose, so the
command exits 1 — which is the whole point: this can block a merge.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import kernels as K  # noqa: E402

from shapesandstrides.verify import verify  # noqa: E402

F32 = ["float32"]


@verify(against="torch.add", dtypes=F32)
def fused_add(x, y):
    return K.triton_add(x, y)


@verify(against="torch.mul", dtypes=F32)
def fused_mul(x, y):
    return K.triton_mul(x, y)


@verify(against="torch.add", dtypes=F32)
def fused_add_autotuned(x, y):
    return K.triton_add_autotuned(x, y)


@verify(against="torch.add", dtypes=F32)
def fused_add_drops_tail(x, y):
    """Broken on purpose: the last partial tile is never written."""
    return K.triton_add_drops_tail(x, y)


@verify(against="torch.add", dtypes=F32)
def fused_add_assumes_contiguous(x, y):
    """Broken on purpose: indexes as if the input were contiguous."""
    return K.triton_add_assumes_contiguous(x, y)


# No single torch operator matches a row-wise reduction, so a one-line torch
# expression stands in as the oracle. Arity comes from the lambda.
@verify(against=lambda x: x.sum(dim=-1), dtypes=F32, op_name="sum")
def rowsum(x):
    """Broken on purpose: accumulates in the wrong precision."""
    return K.triton_rowsum_bad_accum(x)
