"""How much a format result can be trusted.

Parallel to MeasurementTier and OracleTier, and gating the same way: C is the
absence of a verdict, not a failure.
"""

import pytest
from pydantic import ValidationError

from shapesandstrides.formats.grade import FormatTier, Graded


class _Result(Graded):
    value: float


def test_a_graded_result_cannot_omit_its_tier():
    """The same hole closed in 39e58a5: a default lets a result present itself
    as a verdict without saying what backs it."""
    with pytest.raises(ValidationError):
        _Result(value=1.0)


def test_the_tier_field_is_required_structurally():
    """Re-adding a default is the precise regression that reopens the hole, so
    the requirement is asserted on the model rather than only on behaviour."""
    assert _Result.model_fields["format_tier"].is_required()


def test_tier_c_is_the_absence_of_a_verdict():
    r = _Result(value=1.0, format_tier=FormatTier.C)
    assert r.is_format_valid is False


def test_tiers_a_and_b_are_usable_verdicts():
    for t in (FormatTier.A, FormatTier.B):
        assert _Result(value=1.0, format_tier=t).is_format_valid is True


def test_the_tier_survives_json():
    d = _Result(value=1.0, format_tier=FormatTier.B).model_dump(mode="json")
    assert d["format_tier"] == "B"
