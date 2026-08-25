"""Poison fill and canary padding: two checks that need no answer key."""

import pytest
import torch

from shapesandstrides.buffers import (
    CANARY_ELEMENTS,
    BufferReport,
    inspect_buffer,
    poisoned_output,
)

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def test_a_fresh_buffer_is_entirely_unwritten():
    r = inspect_buffer(poisoned_output((16,), torch.float32, "cpu"))
    assert r.unwritten_count == 16
    assert r.total_elements == 16
    assert r.passed is False


def test_a_fully_written_buffer_passes():
    buf = poisoned_output((16,), torch.float32, "cpu")
    buf[:] = 1.0
    r = inspect_buffer(buf)
    assert r.unwritten_count == 0
    assert r.canary_intact is True
    assert r.passed is True


def test_the_masked_tail_bug_is_detected():
    """1000 elements at BLOCK=128 leaves a tail of 104. An off-by-one in the
    mask skips it, and ordinary memory would hold plausible floats there."""
    buf = poisoned_output((1000,), torch.float32, "cpu")
    buf[:896] = 1.0  # seven full blocks, tail skipped
    r = inspect_buffer(buf)
    assert r.unwritten_count == 104
    assert r.passed is False


def test_shape_is_preserved_and_writable_as_normal():
    buf = poisoned_output((4, 8), torch.float32, "cpu")
    assert tuple(buf.shape) == (4, 8)
    buf[:] = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    assert inspect_buffer(buf).passed is True
    assert buf[1, 1].item() == 9.0


def test_an_out_of_bounds_write_trips_the_canary():
    buf = poisoned_output((16,), torch.float32, "cpu")
    buf[:] = 1.0

    # Reach past the logical end the way a kernel with a bad mask would: via
    # the shared storage, not via the view's own bounds.
    beyond = torch.empty(0, dtype=buf.dtype)
    beyond.set_(buf.untyped_storage(), storage_offset=16, size=(CANARY_ELEMENTS,))
    beyond[0] = 7.0

    r = inspect_buffer(buf)
    assert r.canary_intact is False
    assert r.passed is False


def test_a_plain_tensor_reports_no_canary_rather_than_a_passing_one():
    """A tensor we did not allocate has no canary. Saying "intact" would claim
    a check that never ran -- rule 7."""
    r = inspect_buffer(torch.ones(16, dtype=torch.float32))
    assert r.canary_present is False
    assert r.unwritten_count == 0
    assert r.passed is True


def test_report_round_trips_as_json():
    r = inspect_buffer(poisoned_output((8,), torch.float32, "cpu"))
    assert BufferReport.model_validate(r.model_dump(mode="json")) == r


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_works_across_float_dtypes(dtype):
    buf = poisoned_output((32,), dtype, "cpu")
    assert inspect_buffer(buf).unwritten_count == 32
    buf[:] = 1.0
    assert inspect_buffer(buf).passed is True


@needs_cuda
def test_detects_a_real_triton_kernel_skipping_its_tail():
    """The end-to-end case, on the GPU, with a deliberately wrong mask."""
    import triton
    import triton.language as tl

    @triton.jit
    def broken_copy(src, dst, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        # The bug: guards against the padded length rather than n, so the
        # final partial block is never stored.
        mask = offs < (n // BLOCK) * BLOCK
        tl.store(dst + offs, tl.load(src + offs, mask=mask, other=0.0), mask=mask)

    n, block = 1000, 128
    src = torch.arange(n, dtype=torch.float32, device="cuda")
    dst = poisoned_output((n,), torch.float32, "cuda")

    broken_copy[(triton.cdiv(n, block),)](src, dst, n, BLOCK=block)

    r = inspect_buffer(dst)
    assert r.unwritten_count == n - (n // block) * block == 104
    assert r.passed is False


@needs_cuda
def test_a_correct_triton_kernel_passes():
    import triton
    import triton.language as tl

    @triton.jit
    def good_copy(src, dst, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask, other=0.0), mask=mask)

    n, block = 1000, 128
    src = torch.arange(n, dtype=torch.float32, device="cuda")
    dst = poisoned_output((n,), torch.float32, "cuda")

    good_copy[(triton.cdiv(n, block),)](src, dst, n, BLOCK=block)

    r = inspect_buffer(dst)
    assert r.unwritten_count == 0
    assert r.canary_intact is True
    assert r.passed is True
