"""What happens to a number when you store it in a format.

The whole subpackage in one function, and the question people actually have.

``UNDERFLOW`` is its own outcome rather than a large relative error, because a
gradient becoming zero and a gradient becoming imprecise are different events
with different consequences. Overflow announces itself -- an infinity is hard
to miss. Underflow does not: the value simply becomes zero, nothing is raised,
and the weight stops learning. Silent loss is the failure this tool exists to
surface, so it is named.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Iterable, Sequence

import gfloat
from pydantic import BaseModel

from shapesandstrides.formats.grade import FormatTier, Graded
from shapesandstrides.formats.spec import FormatSpec


class Outcome(str, Enum):
    """What became of one value.

    EXACT       stored with no loss at all
    ROUNDED     stored with some error, but still a number
    OVERFLOW    was finite, is now infinite
    UNDERFLOW   was non-zero, is now zero -- the silent one
    BECAME_NAN  is not a number any more
    """

    EXACT = "exact"
    ROUNDED = "rounded"
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"
    BECAME_NAN = "became_nan"


class RoundTripOutcome(Graded):
    format_name: str
    original: float
    result: float
    # None rather than inf when the error is not finite: pydantic serialises
    # inf to JSON null, and a record written that way fails to load again. The
    # parent project carries the same scar on oracle.OracleResult.
    abs_error: float | None = None
    rel_error: float | None = None
    outcome: Outcome


class RoundTripReport(BaseModel):
    outcomes: list[RoundTripOutcome] = []

    def for_format(self, name: str) -> list[RoundTripOutcome]:
        return [o for o in self.outcomes if o.format_name == name]


def _classify(original: float, result: float) -> Outcome:
    if math.isnan(result):
        # A NaN that arrived as a NaN was not damaged by the format, but it is
        # still not a number, and callers care about that either way.
        return Outcome.BECAME_NAN
    if math.isinf(result) and not math.isinf(original):
        return Outcome.OVERFLOW
    if result == 0.0 and original != 0.0:
        return Outcome.UNDERFLOW
    if result == original:
        return Outcome.EXACT
    return Outcome.ROUNDED


def _errors(original: float, result: float) -> tuple[float | None, float | None]:
    if not (math.isfinite(original) and math.isfinite(result)):
        return None, None
    abs_err = abs(result - original)
    # Guard the denominator rather than dividing by zero. Same approach as
    # oracle.compare_against_oracle.
    denom = max(abs(original), 1e-300)
    return abs_err, abs_err / denom


def round_trip(
    values: float | Sequence[float],
    into: Iterable[FormatSpec],
    *,
    rounding: gfloat.RoundMode = gfloat.RoundMode.TiesToEven,
    tiers: dict[str, FormatTier] | None = None,
) -> RoundTripReport:
    """Push ``values`` through each format in ``into`` and report what happened.

    Accepts a bare float as well as a sequence, and several formats at once,
    because the question people actually have is a comparison and it should
    cost one line.

    ``tiers`` maps a format name to the grade a validation run earned it. Any
    format not named there is grade B -- unvalidated. The default is therefore
    the weaker claim, so forgetting to validate can only under-claim, never
    over-claim.
    """
    vs = [values] if isinstance(values, (int, float)) else list(values)
    tiers = tiers or {}
    out: list[RoundTripOutcome] = []
    for spec in into:
        fi = spec.to_gfloat()
        tier = tiers.get(spec.name, FormatTier.B)
        for raw in vs:
            v = float(raw)
            r = gfloat.round_float(fi, v, rounding, sat=False)
            abs_err, rel_err = _errors(v, r)
            out.append(
                RoundTripOutcome(
                    format_name=spec.name,
                    original=v,
                    result=r,
                    abs_error=abs_err,
                    rel_error=rel_err,
                    outcome=_classify(v, r),
                    format_tier=tier,
                )
            )
    return RoundTripReport(outcomes=out)
