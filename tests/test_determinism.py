"""Same input, repeated launches. Any variation is a race."""

import pytest
import torch

from shapesandstrides.buffers import DeterminismReport, check_determinism

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def test_a_deterministic_launch_passes():
    x = torch.arange(64, dtype=torch.float32)
    r = check_determinism(lambda: x * 2, runs=5)
    assert r.passed is True
    assert r.varying_runs == 0
    assert r.max_deviation == 0.0
    assert r.runs == 5


def test_a_varying_launch_fails_and_says_by_how_much():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        out = torch.zeros(8)
        out[0] = state["n"]
        return out

    r = check_determinism(flaky, runs=5)
    assert r.passed is False
    assert r.varying_runs == 4  # every run after the first differs
    assert r.max_deviation == 4.0


def test_a_single_varying_run_among_many_is_still_a_failure():
    state = {"n": 0}

    def occasionally():
        state["n"] += 1
        out = torch.zeros(8)
        if state["n"] == 7:
            out[3] = 0.5
        return out

    r = check_determinism(occasionally, runs=20)
    assert r.passed is False
    assert r.varying_runs == 1
    assert r.max_deviation == 0.5


def test_a_changing_shape_is_reported_not_crashed_on():
    state = {"n": 0}

    def resizing():
        state["n"] += 1
        return torch.zeros(8 if state["n"] < 3 else 9)

    r = check_determinism(resizing, runs=5)
    assert r.passed is False
    assert r.shape_varied is True


def test_nan_output_is_not_mistaken_for_variation():
    """NaN != NaN, so a naive comparison reports a race on every run of a
    kernel that legitimately produces NaN."""

    def stable_nan():
        out = torch.ones(8)
        out[2] = float("nan")
        return out

    r = check_determinism(stable_nan, runs=5)
    assert r.passed is True
    assert r.varying_runs == 0


def test_runs_below_two_is_rejected_with_a_remedy():
    with pytest.raises(ValueError) as e:
        check_determinism(lambda: torch.zeros(4), runs=1)
    assert "at least 2" in str(e.value)


def test_report_round_trips_as_json():
    r = check_determinism(lambda: torch.zeros(4), runs=3)
    assert DeterminismReport.model_validate(r.model_dump(mode="json")) == r


@needs_cuda
def test_a_real_race_may_still_look_deterministic():
    """The documented blind spot, pinned down as a test.

    32 blocks read-modify-write the same address with no atomic. The kernel is
    badly wrong -- it returns one block's partial sum instead of the total --
    but measured on an RTX 3060 it returns the *same* wrong answer 50 times,
    because block scheduling is stable. Determinism checking cannot see it.

    This asserts the check reports honestly rather than asserting the race
    manifests, because on this hardware it does not.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def racing_sum(src, dst, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        partial = tl.sum(tl.load(src + offs))
        # Read-modify-write from every block, unsynchronised.
        tl.store(dst, tl.load(dst) + partial)

    src = torch.rand(4096, dtype=torch.float32, device="cuda")

    def launch():
        dst = torch.zeros(1, dtype=torch.float32, device="cuda")
        racing_sum[(32,)](src, dst, BLOCK=128)
        torch.cuda.synchronize()
        return dst

    r = check_determinism(launch, runs=30)
    assert r.runs == 30
    if not r.passed:
        assert r.max_deviation > 0.0

    # Whatever determinism says, the kernel is wrong -- which is the point.
    assert abs(launch().item() - src.sum().item()) > 1.0


@needs_cuda
def test_a_correct_triton_kernel_is_deterministic():
    import triton
    import triton.language as tl

    @triton.jit
    def scale(src, dst, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask, other=0.0) * 2.0, mask=mask)

    n = 4096
    src = torch.rand(n, dtype=torch.float32, device="cuda")

    def launch():
        dst = torch.empty(n, dtype=torch.float32, device="cuda")
        scale[(triton.cdiv(n, 128),)](src, dst, n, BLOCK=128)
        torch.cuda.synchronize()
        return dst

    r = check_determinism(launch, runs=20)
    assert r.passed is True
    assert r.max_deviation == 0.0
