"""What happens to a number when you store it in a format.

The headline case is a vanishing gradient: 1e-8 is an utterly ordinary
gradient, and fp16 turns it into zero without raising anything at all.
"""

import math

from shapesandstrides.formats import BFLOAT16, FLOAT16, FormatSpec, ieee_bias
from shapesandstrides.formats.grade import FormatTier
from shapesandstrides.formats.roundtrip import Outcome, round_trip


def test_a_gradient_that_vanishes_is_reported_as_underflow_not_as_error():
    """Becoming zero and becoming imprecise are different events, so underflow
    gets its own outcome rather than showing up as a large relative error."""
    o = round_trip(1e-8, into=[FLOAT16]).outcomes[0]
    assert o.result == 0.0
    assert o.outcome is Outcome.UNDERFLOW


def test_the_same_gradient_survives_a_six_exponent_bit_format():
    """The whole reason a format like cbfloat16 exists."""
    cb = FormatSpec(
        name="cb-6-9", exponent_bits=6, mantissa_bits=9, bias=ieee_bias(6)
    )
    o = round_trip(1e-8, into=[cb]).outcomes[0]
    assert o.outcome is Outcome.ROUNDED
    assert o.rel_error is not None and o.rel_error < 0.001


def test_an_exactly_representable_value_round_trips_exactly():
    o = round_trip(1.0, into=[FLOAT16]).outcomes[0]
    assert o.result == 1.0
    assert o.outcome is Outcome.EXACT
    assert o.abs_error == 0.0


def test_a_value_above_the_ceiling_overflows():
    o = round_trip(1e39, into=[FLOAT16]).outcomes[0]
    assert math.isinf(o.result)
    assert o.outcome is Outcome.OVERFLOW


def test_nan_in_stays_nan_out():
    o = round_trip(math.nan, into=[FLOAT16]).outcomes[0]
    assert o.outcome is Outcome.BECAME_NAN
    assert math.isnan(o.result)


def test_an_infinity_that_was_already_infinite_is_not_an_overflow():
    """Overflow means a finite number stopped being finite. An infinity that
    arrived as an infinity did not overflow."""
    o = round_trip(math.inf, into=[FLOAT16]).outcomes[0]
    assert o.outcome is Outcome.EXACT


def test_several_formats_at_once_is_one_call():
    """Convenience: the question people actually have is a comparison."""
    r = round_trip(1e-8, into=[FLOAT16, BFLOAT16])
    assert [o.format_name for o in r.outcomes] == [FLOAT16.name, BFLOAT16.name]


def test_a_list_of_values_is_accepted():
    r = round_trip([1.0, 2.0, 1e-8], into=[FLOAT16])
    assert len(r.outcomes) == 3


def test_every_outcome_carries_a_grade_and_defaults_to_the_weaker_claim():
    o = round_trip(1.0, into=[FLOAT16]).outcomes[0]
    assert o.format_tier is FormatTier.B
    assert o.is_format_valid is True


def test_a_validated_format_can_be_reported_as_grade_a():
    o = round_trip(
        1.0, into=[FLOAT16], tiers={FLOAT16.name: FormatTier.A}
    ).outcomes[0]
    assert o.format_tier is FormatTier.A


def test_non_finite_errors_are_none_not_infinity():
    """pydantic serialises inf to JSON null and the record then fails to load.
    The parent project already carries this scar on OracleResult."""
    o = round_trip(1e39, into=[FLOAT16]).outcomes[0]
    assert o.abs_error is None
    assert o.rel_error is None


def test_the_report_serialises_with_a_stable_key_set():
    """Agents must not need defensive lookups: every entry carries the same
    keys whatever the outcome."""
    r = round_trip([1.0, 1e-8, 1e39, math.nan], into=[FLOAT16])
    dumped = r.model_dump(mode="json")["outcomes"]
    assert len({frozenset(d) for d in dumped}) == 1
    assert {d["outcome"] for d in dumped} == {
        "exact",
        "underflow",
        "overflow",
        "became_nan",
    }


def test_the_report_can_be_filtered_by_format():
    r = round_trip([1.0, 1e-8], into=[FLOAT16, BFLOAT16])
    assert len(r.for_format(FLOAT16.name)) == 2
