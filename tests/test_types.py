import pytest
from sns.types import MeasurementTier, TimingResult, EnvironmentFingerprint


def _result(**overrides):
    base = dict(
        samples_ms=[1.0, 1.1, 0.9],
        median_ms=1.0,
        p10_ms=0.9,
        p90_ms=1.1,
        ci95_lo_ms=0.95,
        ci95_hi_ms=1.05,
        n=3,
        tier=MeasurementTier.B,
        warmup=200,
        inner_reps=1,
        throttle_fired=False,
        clock_cv_pct=1.2,
        clock_range_mhz=15.0,
    )
    base.update(overrides)
    return TimingResult(**base)


def test_timing_result_holds_interval_and_count():
    r = _result()
    assert r.median_ms == 1.0
    assert (r.ci95_lo_ms, r.ci95_hi_ms) == (0.95, 1.05)
    assert r.n == 3
    assert r.tier is MeasurementTier.B


def test_timing_result_cannot_become_a_bare_float():
    """Rule 2 enforced structurally: a caller must not be able to drop the interval."""
    r = _result()
    with pytest.raises(TypeError):
        float(r)
    with pytest.raises(TypeError):
        int(r)


def test_timing_result_rejects_fewer_than_two_samples():
    with pytest.raises(ValueError):
        _result(samples_ms=[1.0], n=1)


def test_tier_c_is_not_a_performance_result():
    r = _result(tier=MeasurementTier.C)
    assert r.is_performance_valid is False
    assert _result(tier=MeasurementTier.A).is_performance_valid is True
    assert _result(tier=MeasurementTier.B).is_performance_valid is True


def test_environment_fingerprint_equality_ignores_nothing_relevant():
    a = EnvironmentFingerprint(
        torch_version="2.7.1", triton_version="3.7.0", cuda_version="12.8",
        driver_version="610.88", gpu_name="A100", compute_cap="8.0",
        arch_family="Ampere", sm_count=108,
    )
    b = a.model_copy()
    assert a.matches(b)
    assert not a.matches(b.model_copy(update={"triton_version": "3.6.0"}))
