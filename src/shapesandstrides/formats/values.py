"""The values that actually break formats.

Formats do not break on 3.7 and 0.5 -- those always work, which is why a suite
built from them always passes and proves nothing. They break at the edges: the
smallest subnormal, one step past the ceiling, and above all at *ties*, values
sitting exactly halfway between two representable numbers, which are the only
place two rounding modes can disagree.

The parent package needs this too. ``oracle.make_inputs`` draws standard-normal
float32 and casts down, so it produces no subnormals, no near-overflow
magnitudes, no NaN inputs and no cancellation-prone pairs -- a gap
``docs/guide/limits.md`` already admits to. Wiring this in there is future work,
but the generator is shared rather than duplicated.

Members are added to ``ValueClass`` only when this module can actually emit
them. An enum value nothing produces is a false promise to whoever reads the
JSON.
"""

from __future__ import annotations

import math
import random
from enum import Enum

from pydantic import BaseModel

from shapesandstrides.formats.spec import FormatSpec


class ValueClass(str, Enum):
    """A kind of awkward number, and the reason it is awkward.

    ZERO / NEGATIVE_ZERO   two distinct encodings of the same value
    SMALLEST_SUBNORMAL     the very smallest thing the format holds
    LARGEST_SUBNORMAL      the last value before normals begin
    SMALLEST_NORMAL        the floor below which precision degrades
    LARGEST_NORMAL         the ceiling
    JUST_OVER_MAX          must overflow, not round back down to the ceiling
    JUST_UNDER_MIN         must underflow to zero -- the silent failure
    POWER_OF_TWO           must be exact; a rounding bug here is unmissable
    TIE                    exactly halfway between neighbours: where rounding
                           modes disagree, and the only class that can tell
                           ties-to-even from ties-to-away
    NAN / *_INFINITY       the specials, which get their own encodings
    ORDINARY               a control group: these should simply work
    """

    ZERO = "zero"
    NEGATIVE_ZERO = "negative_zero"
    SMALLEST_SUBNORMAL = "smallest_subnormal"
    LARGEST_SUBNORMAL = "largest_subnormal"
    SMALLEST_NORMAL = "smallest_normal"
    LARGEST_NORMAL = "largest_normal"
    JUST_OVER_MAX = "just_over_max"
    JUST_UNDER_MIN = "just_under_min"
    POWER_OF_TWO = "power_of_two"
    TIE = "tie"
    NAN = "nan"
    POSITIVE_INFINITY = "positive_infinity"
    NEGATIVE_INFINITY = "negative_infinity"
    ORDINARY = "ordinary"


class LabelledValue(BaseModel):
    """A value together with the class that produced it.

    The label is the point: a failure should name the kind of number that
    caused it, not merely report a magnitude.
    """

    value: float
    value_class: ValueClass


class ValueSet(BaseModel):
    format_name: str
    seed: int
    values: list[LabelledValue] = []

    def of_class(self, c: ValueClass) -> list[float]:
        return [lv.value for lv in self.values if lv.value_class is c]

    @property
    def as_floats(self) -> list[float]:
        return [lv.value for lv in self.values]


def _for_class(
    c: ValueClass, spec: FormatSpec, rng: random.Random, n: int
) -> list[float]:
    sub = spec.smallest_subnormal
    norm = spec.smallest_normal
    top = spec.max_value
    # Spacing between representable numbers immediately above 1.0.
    step = 2.0**-spec.mantissa_bits

    if c is ValueClass.ZERO:
        return [0.0]
    if c is ValueClass.NEGATIVE_ZERO:
        return [-0.0]
    if c is ValueClass.SMALLEST_SUBNORMAL:
        return [sub]
    if c is ValueClass.LARGEST_SUBNORMAL:
        return [norm - sub]
    if c is ValueClass.SMALLEST_NORMAL:
        return [norm]
    if c is ValueClass.LARGEST_NORMAL:
        return [top]
    if c is ValueClass.JUST_OVER_MAX:
        # Comfortably past the ceiling, so it must overflow rather than round
        # back down to max.
        return [top * 1.5, math.ldexp(top, 1)]
    if c is ValueClass.JUST_UNDER_MIN:
        # Below half the smallest subnormal, so it must round to zero. This is
        # the class that catches silent underflow.
        return [sub / 4, sub / 1000]
    if c is ValueClass.POWER_OF_TWO:
        return [1.0, 2.0, 0.5, 4.0, 0.25]
    if c is ValueClass.TIE:
        # Exactly halfway between 1.0 and its successor, and between the next
        # pair up. Ties-to-even breaks these in opposite directions, which is
        # what makes them discriminating rather than merely awkward.
        return [1.0 + step / 2, 1.0 + step + step / 2]
    if c is ValueClass.NAN:
        return [math.nan]
    if c is ValueClass.POSITIVE_INFINITY:
        return [math.inf]
    if c is ValueClass.NEGATIVE_INFINITY:
        return [-math.inf]
    if c is ValueClass.ORDINARY:
        return [rng.gauss(0.0, 1.0) for _ in range(n)]
    raise AssertionError(
        f"ValueClass.{c.name} has no generator. A class that cannot be emitted "
        f"must not be a member -- see the module docstring."
    )


def values_for(
    spec: FormatSpec,
    *,
    classes: list[ValueClass] | list[str] | None = None,
    seed: int = 0xC0FFEE,
    ordinary_count: int = 64,
) -> ValueSet:
    """Build a labelled, reproducible set of awkward values for ``spec``.

    Defaults to every class, because the classes a caller would think to skip
    are the ones that find bugs. An unrecognised class is an error rather than
    a silent skip: a class that quietly did not run is a test that quietly did
    not run.
    """
    wanted: list[ValueClass] | list[str] = (
        list(ValueClass) if classes is None else classes
    )

    resolved: list[ValueClass] = []
    for c in wanted:
        if isinstance(c, ValueClass):
            resolved.append(c)
            continue
        try:
            resolved.append(ValueClass(c))
        except ValueError:
            raise ValueError(
                f"{c!r} is not a value class. Valid classes: "
                f"{[m.value for m in ValueClass]}"
            ) from None

    rng = random.Random(seed)
    out: list[LabelledValue] = []
    for c in resolved:
        for v in _for_class(c, spec, rng, ordinary_count):
            out.append(LabelledValue(value=v, value_class=c))
    return ValueSet(format_name=spec.name, seed=seed, values=out)
