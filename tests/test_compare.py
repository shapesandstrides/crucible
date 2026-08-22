import pytest

torch = pytest.importorskip("torch")

from sns.types import ComparisonResult, MeasurementTier, TimingResult

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _timing(median, tier=MeasurementTier.B):
    return TimingResult(
        samples_ms=[median] * 30, median_ms=median, p10_ms=median, p90_ms=median,
        ci95_lo_ms=median, ci95_hi_ms=median, n=30, tier=tier, warmup=200,
    )


def test_comparison_result_cannot_be_floated():
    c = ComparisonResult(
        candidate=_timing(1.0), baseline=_timing(2.0),
        speedup=2.0, speedup_ci_lo=1.9, speedup_ci_hi=2.1,
    )
    with pytest.raises(TypeError):
        float(c)


def test_comparison_tier_is_the_worse_of_the_two():
    c = ComparisonResult(
        candidate=_timing(1.0, MeasurementTier.A),
        baseline=_timing(2.0, MeasurementTier.C),
        speedup=2.0, speedup_ci_lo=1.9, speedup_ci_hi=2.1,
    )
    assert c.tier is MeasurementTier.C
    assert c.is_performance_valid is False


def test_comparison_tier_a_requires_both_sides_locked():
    c = ComparisonResult(
        candidate=_timing(1.0, MeasurementTier.A),
        baseline=_timing(2.0, MeasurementTier.A),
        speedup=2.0, speedup_ci_lo=1.9, speedup_ci_hi=2.1,
    )
    assert c.tier is MeasurementTier.A


@requires_gpu
def test_compare_measures_both_sides_in_one_call():
    from sns.timing import compare

    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    r = compare(lambda: a @ a, lambda: a @ a, warmup=20, iters=30)

    assert r.candidate.n == 30
    assert r.baseline.n == 30
    # Identical work on both sides: the interval must admit parity.
    assert r.speedup_ci_lo <= 1.0 <= r.speedup_ci_hi
