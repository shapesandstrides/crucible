import pytest

torch = pytest.importorskip("torch")

from sns.timing import clock_sample_stride, measure, resolve_inner_reps
from sns.types import MeasurementTier

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def test_resolve_inner_reps_leaves_long_kernels_alone():
    # 0.5 ms is far above the 10 us floor.
    assert resolve_inner_reps(0.5, min_duration_us=10.0) == 1


def test_resolve_inner_reps_loops_short_kernels():
    # 2 us per iteration needs 5 reps to clear a 10 us floor.
    assert resolve_inner_reps(0.002, min_duration_us=10.0) == 5


def test_resolve_inner_reps_rounds_up():
    assert resolve_inner_reps(0.003, min_duration_us=10.0) == 4


def test_resolve_inner_reps_handles_zero_measurement():
    assert resolve_inner_reps(0.0, min_duration_us=10.0) == 1000


@pytest.mark.parametrize("iters", [30, 31, 40, 64, 100, 1000])
def test_clock_sample_stride_never_exceeds_the_cap(iters):
    stride = clock_sample_stride(iters)
    assert len(range(0, iters, stride)) <= 8


def test_clock_sample_stride_is_exactly_eight_at_the_default_iters():
    """30 is the default and the case floor division got wrong."""
    assert clock_sample_stride(30) == 4
    assert len(range(0, 30, 4)) == 8


def test_clock_sample_stride_is_never_zero():
    assert clock_sample_stride(1) >= 1
    assert clock_sample_stride(0) >= 1


@requires_gpu
def test_measure_returns_a_populated_result():
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    r = measure(lambda: a @ a, warmup=20, iters=30)

    assert r.n == 30
    assert len(r.samples_ms) == 30
    assert r.median_ms > 0
    assert r.ci95_lo_ms <= r.median_ms <= r.ci95_hi_ms
    assert r.p10_ms <= r.median_ms <= r.p90_ms
    assert r.tier in (MeasurementTier.A, MeasurementTier.B, MeasurementTier.C)
    assert r.warmup == 20


@requires_gpu
def test_measure_rejects_too_few_iterations():
    a = torch.randn(64, 64, device="cuda")
    with pytest.raises(ValueError, match="30"):
        measure(lambda: a @ a, warmup=5, iters=5)


@requires_gpu
def test_measure_result_cannot_be_floated():
    a = torch.randn(64, 64, device="cuda")
    r = measure(lambda: a @ a, warmup=10, iters=30)
    with pytest.raises(TypeError):
        float(r)


@requires_gpu
def test_measure_is_internally_consistent_for_a_tiny_kernel():
    """A tiny kernel must still produce a coherent result, whether or not the
    duration guard ends up looping it. resolve_inner_reps is pinned separately."""
    a = torch.randn(8, 8, device="cuda")
    r = measure(lambda: a + a, warmup=10, iters=30)

    assert r.median_ms > 0
    assert r.ci95_lo_ms <= r.median_ms <= r.ci95_hi_ms
    assert len(r.samples_ms) == r.n == 30
