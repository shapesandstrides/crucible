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


def test_persistent_hw_thermal_throttling_is_detected_not_just_transitions():
    """before == after == Active means throttled the whole window, not clean."""
    from sns.env import hw_thermal_throttle_active

    snap = {"sw_power_cap": "Not Active", "hw_thermal_slowdown": "Active"}
    fired = hw_thermal_throttle_active(snap) or hw_thermal_throttle_active(snap)
    assert fired is True


def test_power_cap_alone_is_not_hw_thermal_throttling():
    """sw_power_cap Active on an idle GPU must not read as thermal distress."""
    from sns.env import hw_thermal_throttle_active

    snap = {"sw_power_cap": "Active", "hw_thermal_slowdown": "Not Active"}
    assert hw_thermal_throttle_active(snap) is False


def test_sw_thermal_slowdown_alone_is_not_hw_thermal_throttling():
    """Measured Active on real laptop hardware at idle, 55C, 18W — a
    software flag stuck on is not the hardware safety-circuit assertion."""
    from sns.env import hw_thermal_throttle_active

    snap = {"sw_thermal_slowdown": "Active", "hw_thermal_slowdown": "Not Active"}
    assert hw_thermal_throttle_active(snap) is False


def test_persistent_hw_power_brake_is_detected_not_just_transitions():
    """before == after == Active means throttled the whole window, not
    clean. hw_power_brake_slowdown is a hardware assertion by the same
    naming and mechanism as hw_thermal_slowdown, so it must gate the same
    way — measured Not Active throughout on real hardware, including
    under sustained load."""
    from sns.env import hw_throttle_active

    snap = {"sw_power_cap": "Not Active", "hw_power_brake_slowdown": "Active"}
    assert hw_throttle_active(snap, snap) is True


def test_hw_power_brake_gates_even_when_hw_thermal_is_clean():
    from sns.env import hw_throttle_active

    before = {"hw_thermal_slowdown": "Not Active", "hw_power_brake_slowdown": "Not Active"}
    after = {"hw_thermal_slowdown": "Not Active", "hw_power_brake_slowdown": "Active"}
    assert hw_throttle_active(before, after) is True


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


@requires_gpu
def test_measure_uses_the_lock_state_observed_during_measurement(monkeypatch):
    """policy.restore() clears policy.locked in a finally block. The tier must
    come from the lock state captured before that, or Tier A can never occur."""
    from sns import timing
    from sns.types import MeasurementTier

    class FakeSampler:
        def __init__(self, device=0):
            pass

        def sample_clock_mhz(self):
            return 1500.0

        def throttled_now(self):
            return False

        def shutdown(self):
            pass

    monkeypatch.setattr(timing, "ClockSampler", FakeSampler)
    monkeypatch.setattr(
        timing, "throttle_snapshot", lambda *a, **k: {"sw_power_cap": "Not Active"}
    )

    class FakeLockedPolicy:
        def __init__(self):
            self.locked = False

        def apply(self):
            self.locked = True

        def restore(self):
            self.locked = False

    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    r = timing.measure(lambda: a @ a, warmup=10, iters=30, policy=FakeLockedPolicy())

    assert r.tier is MeasurementTier.A
