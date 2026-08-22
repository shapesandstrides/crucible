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
def test_compare_rejects_a_zero_median_candidate(monkeypatch):
    """A 0 ms candidate must fail loudly, not raise ZeroDivisionError.

    compare() now runs its own interleaved measurement loop rather than
    delegating to measure(), so the seam to mock is _interleaved_samples,
    the helper that does the actual GPU timing. This still exercises the
    zero-median guard in compare() in isolation from real timing noise.
    """
    from sns import timing

    monkeypatch.setattr(
        timing,
        "_interleaved_samples",
        lambda *a, **k: ([0.0] * 30, [1.0] * 30, 1, MeasurementTier.B, False, []),
    )

    with pytest.raises(ValueError, match="0 ms"):
        timing.compare(lambda: None, lambda: None)


@requires_gpu
def test_compare_measures_both_sides_in_one_call():
    from sns.timing import compare

    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    r = compare(lambda: a @ a, lambda: a @ a, warmup=20, iters=30)

    assert r.candidate.n == 30
    assert r.baseline.n == 30
    assert r.speedup > 0 and r.speedup_ci_lo <= r.speedup <= r.speedup_ci_hi
    # Identical work on both sides, interleaved. Interleaving cancels
    # between-window drift, so this envelope is tight rather than the loose
    # 0.8-1.25 range the old sequential implementation needed.
    assert 0.9 <= r.speedup <= 1.11, f"identical work gave {r.speedup:.3f}x"


@requires_gpu
def test_interleaved_comparison_of_identical_work_is_tight():
    """The whole point. Sequential measurement gave a p90 error above 100%
    on this hardware; interleaved must be an order of magnitude better."""
    from sns.timing import compare

    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    errors = []
    for _ in range(5):
        r = compare(lambda: a @ a, lambda: a @ a, warmup=50, iters=30)
        errors.append(abs(r.speedup - 1.0))

    assert max(errors) < 0.15, f"identical work drifted {max(errors):.1%}"


@requires_gpu
def test_both_sides_share_one_inner_rep_count():
    """Different rep counts would reintroduce a systematic difference
    through launch-overhead amortisation."""
    from sns.timing import compare

    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    r = compare(lambda: a @ a, lambda: a @ a, warmup=20, iters=30)
    assert r.candidate.inner_reps == r.baseline.inner_reps


@requires_gpu
def test_a_genuinely_faster_candidate_is_still_detected():
    """Cancelling drift must not cancel real differences too."""
    from sns.timing import compare

    small = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    big = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

    r = compare(lambda: small @ small, lambda: big @ big, warmup=20, iters=30)
    assert r.speedup > 1.5, f"a much cheaper candidate should win clearly, got {r.speedup:.2f}x"
    assert r.speedup_ci_lo > 1.0
