"""Sweeping a format parameter.

The bias is free in hardware -- subtracting 31 and subtracting 40 are the same
adder -- and nobody tunes it. Meanwhile loss scaling is a runtime workaround for
a badly positioned exponent window. This is the machinery for asking whether
moving the window once, for free, would do the same job.
"""

import pytest

from shapesandstrides.formats import FLOAT16, FormatSpec, ieee_bias
from shapesandstrides.formats.error import gradient_like
from shapesandstrides.formats.ops import softmax, total
from shapesandstrides.formats.sweep import sweep


def _cb() -> FormatSpec:
    return FormatSpec(
        name="cb-6-9", exponent_bits=6, mantissa_bits=9, bias=ieee_bias(6)
    )


def test_raising_the_bias_lowers_the_window_and_saves_gradients():
    """The headline experiment. A larger bias subtracts more, so the window
    slides down, so fewer small values fall off the bottom.

    Measured on a 6-exponent-bit format over 300 gradient-magnitude values:
    174 lost at bias 12, 131 at 16, 87 at 20, 38 at 24, none from 28 up.
    """
    xs = gradient_like(300, seed=1)
    r = sweep(_cb(), "bias", [12, 16, 20, 24], total, [xs])

    lost = {p.parameter_value: p.underflowed for p in r.points}
    assert lost[12] > lost[16] > lost[20] > lost[24] > 0, (
        f"expected monotonic improvement, got {lost}"
    )


def test_the_underflow_cliff_is_at_half_the_smallest_subnormal():
    """Two steps further down than intuition suggests, and both matter.

    First, subnormals extend the usable range below the normal floor -- by a
    factor of 2^9 for a 9-mantissa-bit format. Second, round-to-nearest pulls
    anything above *half* the smallest subnormal up to it rather than down to
    zero, so the real cliff is another factor of two lower again.

    Measured at bias 28: the normal floor is 7.45e-09 and the smallest
    subnormal 1.46e-11, yet an input of 1.07e-11 -- below both -- survives,
    because it clears the 7.28e-12 halfway point. Reasoning from the normal
    floor would have predicted total loss here.
    """
    xs = gradient_like(300, seed=1)
    smallest_input = min(abs(x) for x in xs)
    point = sweep(_cb(), "bias", [28], total, [xs]).points[0]

    subnormal_floor = point.smallest_normal * 2.0**-9
    assert point.smallest_normal > smallest_input, (
        "the input must sit below the normal floor for this to be interesting"
    )
    assert subnormal_floor > smallest_input, (
        "and below the smallest subnormal, which is the surprising part"
    )
    assert subnormal_floor / 2 < smallest_input, "but above half of it"
    assert point.underflowed == 0, "so nothing was lost"


def test_too_low_a_bias_degrades_precision_as_well_as_losing_values():
    """A badly placed window costs twice. Values fall off the bottom, and the
    survivors get pushed into the subnormal range where there are fewer
    significant bits left."""
    xs = gradient_like(300, seed=1)
    r = sweep(_cb(), "bias", [12, 16], total, [xs])
    err = {p.parameter_value: p.error.p50_rel_error for p in r.points}
    assert err[12] > err[16] * 2, (
        f"crowding values into subnormals should visibly hurt precision, got {err}"
    )


def test_each_point_reports_the_window_floor_that_explains_it():
    """A count with no explanation is not a finding. The floor is why."""
    r = sweep(_cb(), "bias", [24, 40], total, [gradient_like(50, seed=2)])
    floors = {p.parameter_value: p.smallest_normal for p in r.points}
    assert floors[40] < floors[24], "a larger bias must give a lower floor"


def test_the_best_bias_is_the_lowest_one_that_loses_nothing():
    """Raising a bias buys room at the bottom by giving up the ceiling. Once a
    setting loses nothing, raising it further buys nothing and costs headroom --
    so among the lossless settings the lowest wins. Without that rule "best"
    would be decided by list order, since every lossless setting here reports
    identical error."""
    xs = gradient_like(300, seed=1)
    r = sweep(_cb(), "bias", [20, 28, 31, 42], total, [xs])

    lossless = [p.parameter_value for p in r.points if p.underflowed == 0]
    assert lossless == [28, 31, 42], f"expected 20 to lose values, got {lossless}"
    assert r.best_by_silent_loss == 28


def test_every_point_carries_a_full_error_distribution():
    """Never a bare number, even inside a sweep."""
    r = sweep(_cb(), "bias", [31, 35], softmax, [gradient_like(100, seed=4)])
    for p in r.points:
        assert p.error.p50_rel_error <= p.error.p90_rel_error
        assert p.error.ci95_lo <= p.error.ci95_hi


def test_a_bias_outside_the_exponent_range_is_refused_with_the_valid_range():
    """Silently skipping impossible points would make the curve lie about what
    was actually tested."""
    with pytest.raises(ValueError, match="0..63"):
        sweep(_cb(), "bias", [31, 999], total, [[1.0, 2.0]])


def test_an_unsweepable_parameter_is_refused_with_the_valid_list():
    with pytest.raises(ValueError, match="cannot be swept"):
        sweep(_cb(), "notes", ["a", "b"], total, [[1.0, 2.0]])


def test_an_unknown_parameter_is_refused():
    with pytest.raises(ValueError, match="not a field"):
        sweep(_cb(), "biass", [31], total, [[1.0, 2.0]])


def test_mantissa_bits_can_be_swept_too():
    """The bias is the interesting axis but not the only one."""
    r = sweep(_cb(), "mantissa_bits", [5, 9], total, [[0.1, 0.3, 0.7]])
    errs = {p.parameter_value: p.error.max_rel_error for p in r.points}
    assert errs[5] > errs[9], "fewer mantissa bits must round more coarsely"


def test_the_sweep_records_what_it_swept_and_its_seed():
    r = sweep(_cb(), "bias", [31], total, [[1.0, 2.0]], seed=77)
    assert r.parameter == "bias"
    assert r.base_format_name == "cb-6-9"
    assert r.seed == 77


def test_the_report_serialises_for_agents():
    r = sweep(_cb(), "bias", [31, 40], total, [gradient_like(20, seed=5)])
    d = r.model_dump(mode="json")
    assert d["parameter"] == "bias"
    assert len(d["points"]) == 2
    assert {frozenset(p) for p in d["points"]}.__len__() == 1
    assert d["points"][0]["error"]["quantization_model"] == "storage_only"


def test_a_swept_point_is_never_grade_a():
    """A swept variant is a format nobody validated -- by construction it has
    no native counterpart, so it cannot claim the stronger grade."""
    r = sweep(FLOAT16, "bias", [15, 16], total, [[1.0, 2.0]])
    assert all(p.error.format_tier.value == "B" for p in r.points)


def test_an_empty_value_list_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        sweep(_cb(), "bias", [], total, [[1.0, 2.0]])
