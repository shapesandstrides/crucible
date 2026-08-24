"""Is shifting the exponent bias the same thing as loss scaling?

Loss scaling multiplies values up into the representable window at runtime, then
divides back out. Shifting the bias moves the window down instead, once, for
free. If those are the same operation, then wherever you control the format the
bias shift is strictly better -- same numerics, no runtime machinery, no scale
factor to tune, no steps skipped on overflow.
"""

import pytest

from shapesandstrides.formats import FLOAT16, FormatSpec
from shapesandstrides.formats.equivalence import loss_scaling_equivalence
from shapesandstrides.formats.error import recorded_gradients


def _cb(bias: int = 31) -> FormatSpec:
    return FormatSpec(
        name=f"cb-6-9-b{bias}", exponent_bits=6, mantissa_bits=9, bias=bias
    )


def test_a_bias_shift_is_numerically_identical_to_loss_scaling():
    """The headline result, on real recorded gradients plus the boundaries."""
    r = loss_scaling_equivalence(_cb(), bias_shift=5, values=recorded_gradients())
    assert r.differing == 0
    assert r.equivalent is True
    assert r.loss_scale == 32.0


def test_it_holds_on_the_extreme_tail_too():
    """The tail is where a difference would show up if there were one."""
    r = loss_scaling_equivalence(
        _cb(), bias_shift=5, values=recorded_gradients(tail=True)
    )
    assert r.differing == 0


def test_it_holds_at_the_ceiling_where_both_overflow():
    """Loss scaling can push a large value past the ceiling; a bias shift lowers
    the ceiling by the same factor. Those must coincide, or the equivalence
    would break exactly where it matters most."""
    base = _cb()
    r = loss_scaling_equivalence(
        base,
        bias_shift=5,
        values=[base.max_value, base.max_value * 0.9, base.max_value / 32],
    )
    assert r.differing == 0


def test_the_shifted_ceiling_equals_the_base_ceiling_divided_by_the_scale():
    """Why the equivalence holds: the two describe the same grid."""
    base = _cb()
    r = loss_scaling_equivalence(base, bias_shift=5, values=[1.0])
    assert r.shifted_max == pytest.approx(base.max_value / 32.0)


def test_several_shift_sizes_all_hold():
    xs = recorded_gradients(tail=True)
    for shift in (1, 3, 8, 12):
        r = loss_scaling_equivalence(_cb(), bias_shift=shift, values=xs)
        assert r.differing == 0, f"shift {shift} diverged"
        assert r.loss_scale == 2.0**shift


def test_a_negative_shift_is_allowed_and_still_holds():
    """Raising the ceiling instead of lowering the floor is the same trade in
    reverse."""
    r = loss_scaling_equivalence(_cb(), bias_shift=-4, values=recorded_gradients())
    assert r.differing == 0
    assert r.loss_scale == 2.0**-4


def test_a_shift_off_the_end_of_the_exponent_is_refused():
    with pytest.raises(ValueError, match="outside the range"):
        loss_scaling_equivalence(_cb(), bias_shift=100, values=[1.0])


def test_a_zero_shift_is_refused_as_meaningless():
    with pytest.raises(ValueError, match="non-zero"):
        loss_scaling_equivalence(_cb(), bias_shift=0, values=[1.0])


def test_it_reports_a_difference_when_one_genuinely_exists():
    """Proof the check can fail. Comparing a bias shift against the WRONG scale
    must be caught -- otherwise this test suite proves nothing."""
    r = loss_scaling_equivalence(
        _cb(), bias_shift=5, values=recorded_gradients(tail=True), loss_scale=8.0
    )
    assert r.differing > 0
    assert r.equivalent is False
    assert r.first_difference is not None


def test_it_serialises_for_agents():
    d = loss_scaling_equivalence(FLOAT16, bias_shift=3, values=[1.0, 1e-6]).model_dump(
        mode="json"
    )
    assert d["equivalent"] is True
    assert d["bias_shift"] == 3
    assert d["first_difference"] is None
