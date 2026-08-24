"""The values that actually break formats.

Known-answer tests throughout: if the generator cannot get fp16's smallest
subnormal exactly right, nothing built on top of it means anything.
"""

import math

import pytest

from shapesandstrides.formats import BFLOAT16, FLOAT16, FormatSpec
from shapesandstrides.formats.values import ValueClass, values_for


def test_fp16_smallest_subnormal_is_exactly_two_to_the_minus_24():
    vs = values_for(FLOAT16, classes=[ValueClass.SMALLEST_SUBNORMAL])
    assert vs.of_class(ValueClass.SMALLEST_SUBNORMAL) == [2.0**-24]


def test_fp16_smallest_normal_is_exactly_two_to_the_minus_14():
    vs = values_for(FLOAT16, classes=[ValueClass.SMALLEST_NORMAL])
    assert vs.of_class(ValueClass.SMALLEST_NORMAL) == [2.0**-14]


def test_fp16_largest_normal_is_65504():
    vs = values_for(FLOAT16, classes=[ValueClass.LARGEST_NORMAL])
    assert vs.of_class(ValueClass.LARGEST_NORMAL) == [65504.0]


def test_a_tie_sits_exactly_halfway_between_two_representable_numbers():
    """Ties are the only place two rounding modes can disagree, so a suite
    without them cannot tell one mode from another."""
    vs = values_for(BFLOAT16, classes=[ValueClass.TIE])
    step = 2.0**-7  # bf16 has 7 explicit mantissa bits: the spacing above 1.0
    assert 1.0 + step / 2 in vs.of_class(ValueClass.TIE)


def test_just_over_max_is_above_the_ceiling():
    vs = values_for(FLOAT16, classes=[ValueClass.JUST_OVER_MAX])
    got = vs.of_class(ValueClass.JUST_OVER_MAX)
    assert got and all(v > 65504.0 for v in got)


def test_just_under_min_is_below_the_floor():
    vs = values_for(FLOAT16, classes=[ValueClass.JUST_UNDER_MIN])
    got = vs.of_class(ValueClass.JUST_UNDER_MIN)
    assert got and all(0 < v < 2.0**-24 for v in got)


def test_specials_are_present_and_are_what_they_claim():
    vs = values_for(
        FLOAT16,
        classes=[
            ValueClass.NAN,
            ValueClass.POSITIVE_INFINITY,
            ValueClass.NEGATIVE_INFINITY,
            ValueClass.NEGATIVE_ZERO,
        ],
    )
    assert all(math.isnan(v) for v in vs.of_class(ValueClass.NAN))
    assert vs.of_class(ValueClass.POSITIVE_INFINITY) == [math.inf]
    assert vs.of_class(ValueClass.NEGATIVE_INFINITY) == [-math.inf]
    nz = vs.of_class(ValueClass.NEGATIVE_ZERO)[0]
    assert nz == 0.0 and math.copysign(1.0, nz) < 0


def test_the_default_is_every_class():
    """A class silently absent is a test silently not run."""
    vs = values_for(FLOAT16)
    produced = {lv.value_class for lv in vs.values}
    assert produced == set(ValueClass), f"missing: {set(ValueClass) - produced}"


def test_the_same_seed_gives_the_same_values():
    a = values_for(FLOAT16, seed=7)
    b = values_for(FLOAT16, seed=7)
    c = values_for(FLOAT16, seed=8)
    assert a.as_floats == b.as_floats
    assert a.as_floats != c.as_floats


def test_the_seed_is_recorded_in_the_output():
    """Whoever reads this has to be able to reproduce it."""
    assert values_for(FLOAT16, seed=99).seed == 99


def test_an_unknown_class_raises_rather_than_being_skipped():
    with pytest.raises(ValueError, match="not a value class"):
        values_for(FLOAT16, classes=["denormalish"])


def test_it_works_for_a_reconstructed_format_too():
    f = FormatSpec(name="cb-6-9", exponent_bits=6, mantissa_bits=9, bias=31)
    vs = values_for(f, classes=[ValueClass.SMALLEST_NORMAL])
    assert math.isclose(
        vs.of_class(ValueClass.SMALLEST_NORMAL)[0], 2.0**-30, rel_tol=1e-12
    )


def test_every_value_is_labelled_with_the_class_that_made_it():
    """A failure should name the kind of number that caused it, not just a
    magnitude."""
    vs = values_for(FLOAT16)
    assert all(isinstance(lv.value_class, ValueClass) for lv in vs.values)
    assert vs.format_name == FLOAT16.name


def test_the_value_set_serialises_for_agents():
    d = values_for(FLOAT16, classes=[ValueClass.SMALLEST_NORMAL], seed=3).model_dump(
        mode="json"
    )
    assert d["seed"] == 3
    assert d["values"][0]["value_class"] == "smallest_normal"
