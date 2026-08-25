"""reference_lowp: the unfused chain at the precision it actually runs at.

`reference_fp64` gives the correctly rounded ideal. That is not what torch
returns at float16 -- real torch carries rounding error of its own, and that
error is the budget a fused kernel is allowed.
"""

import torch

from shapesandstrides.oracle import reference_fp64, reference_lowp


def _chain(x, y):
    """Short enough to reason about, long enough that ordering matters."""
    return torch.tanh(x @ y)


def test_runs_at_the_inputs_own_dtype():
    x = torch.randn(64, 64, dtype=torch.float16)
    y = torch.randn(64, 64, dtype=torch.float16)
    assert reference_lowp(_chain, [x, y]).dtype is torch.float16


def test_carries_real_error_against_the_fp64_golden():
    """The premise of the whole error budget.

    If the low-precision path were exact, the budget's denominator would be
    zero and there would be nothing to grade against.
    """
    torch.manual_seed(0)
    x = torch.randn(128, 128, dtype=torch.float16)
    y = torch.randn(128, 128, dtype=torch.float16)

    golden = reference_fp64(_chain, [x, y], torch.float64)
    lowp = reference_lowp(_chain, [x, y]).to(torch.float64)

    assert (lowp - golden).abs().max().item() > 0.0


def test_preserves_tuple_outputs():
    """Fused kernels commonly return a primary output plus saved statistics.

    `reference_fp64` handles that; this must match, or the two references
    disagree in shape and the budget compares the wrong tensors.
    """

    def two_out(x):
        return x * 2, x + 1

    out = reference_lowp(two_out, [torch.randn(8, dtype=torch.float16)])
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert all(t.dtype is torch.float16 for t in out)


def test_does_not_mutate_or_move_the_callers_tensors():
    """A reference that moved its inputs off the GPU would silently change
    what the caller measures next."""
    x = torch.randn(16, 16, dtype=torch.float32)
    before = x.clone()
    reference_lowp(lambda t: t * 3, [x])
    assert torch.equal(x, before)


@torch.no_grad()
def test_accepts_cuda_inputs_and_computes_on_cpu():
    """Runs on CPU on purpose: a GPU reference would bake that GPU's
    accumulation order into the error budget."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("no CUDA device")

    x = torch.randn(32, 32, dtype=torch.float16, device="cuda")
    y = torch.randn(32, 32, dtype=torch.float16, device="cuda")
    out = reference_lowp(_chain, [x, y])
    assert out.device.type == "cpu"
    assert out.dtype is torch.float16
