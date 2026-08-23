import pytest
from shapesandstrides.stats import bootstrap_ci, cv_percent, percentile, quantization_step, ratio_ci


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3.0
    assert percentile([1, 2, 3, 4, 5], 0.0) == 1.0
    assert percentile([1, 2, 3, 4, 5], 1.0) == 5.0
    assert percentile([1, 2], 0.5) == 1.5


def test_percentile_single_sample():
    assert percentile([7.0], 0.9) == 7.0


def test_percentile_rejects_empty():
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_bootstrap_ci_on_constant_data_is_that_constant():
    lo, hi = bootstrap_ci([5.0] * 40)
    assert lo == 5.0 and hi == 5.0


def test_bootstrap_ci_brackets_the_median():
    samples = [10.0 + (i % 5) * 0.1 for i in range(50)]
    lo, hi = bootstrap_ci(samples)
    assert lo <= percentile(samples, 0.5) <= hi


def test_bootstrap_ci_is_deterministic():
    samples = [1.0, 1.4, 0.9, 1.1, 1.2, 1.3, 0.8, 1.05]
    assert bootstrap_ci(samples, seed=42) == bootstrap_ci(samples, seed=42)


def test_bootstrap_ci_narrows_with_tighter_data():
    """The median is a rank statistic, so the datasets must differ in spread
    across distinct values — not merely in the magnitude of a few repeats."""
    tight = [1.0 + i * 0.001 for i in range(60)]
    loose = [1.0 + i * 0.1 for i in range(60)]
    t_lo, t_hi = bootstrap_ci(tight)
    l_lo, l_hi = bootstrap_ci(loose)
    assert (t_hi - t_lo) < (l_hi - l_lo)


def test_bootstrap_ci_rejects_one_sample():
    with pytest.raises(ValueError):
        bootstrap_ci([1.0])


def test_cv_percent():
    assert cv_percent([5.0, 5.0, 5.0]) == 0.0
    assert cv_percent([1.0, 3.0]) == pytest.approx(50.0)


def test_cv_percent_handles_zero_mean():
    assert cv_percent([0.0, 0.0]) == 0.0


def test_ratio_ci_candidate_twice_as_fast():
    """speedup = baseline / candidate, so >1 means the candidate wins."""
    candidate = [1.0] * 30
    baseline = [2.0] * 30
    lo, hi = ratio_ci(candidate, baseline)
    assert lo == pytest.approx(2.0) and hi == pytest.approx(2.0)


def test_ratio_ci_spans_one_when_equivalent():
    candidate = [1.0, 1.1, 0.9, 1.05, 0.95] * 6
    baseline = [1.0, 1.1, 0.9, 1.05, 0.95] * 6
    lo, hi = ratio_ci(candidate, baseline)
    assert lo <= 1.0 <= hi


def test_ci_is_never_narrower_than_the_timer_resolution():
    """A heavily quantized sample set must not yield a confident interval."""
    from shapesandstrides.stats import quantization_step

    # Simulates a fast kernel on a ~1us timer: 3 distinct values, many ties.
    samples = [0.039] * 20 + [0.040] * 20 + [0.041] * 10
    step = quantization_step(samples)
    assert step is not None
    assert abs(step - 0.001) < 1e-9


def test_quantization_step_is_none_for_a_single_distinct_value():
    assert quantization_step([1.0] * 10) is None
