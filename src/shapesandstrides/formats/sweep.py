"""Vary one format parameter and measure what it costs.

## Why this exists

The exponent bias is **free in hardware**. The chip stores the exponent as a
plain unsigned integer; the bias is only what you subtract to read it, and
subtracting 31 costs exactly the same adder as subtracting 40. It is a dial
nobody pays for and nobody tunes.

Meanwhile, the industry's remedy for gradients falling off the bottom of the
representable window is **loss scaling**: multiply the loss by a constant,
divide it back out afterwards, and manage that constant every step of training.

Look at what that is -- a runtime workaround for a badly positioned window.
Shifting the bias moves the window directly, once, for nothing.

So: could a shifted bias remove the need for loss scaling? No existing tool can
run that experiment. This is the machinery for it.

## Why a sweep rather than a guess

The bias of a vendor format is frequently unpublished -- Cerebras does not state
cbfloat16's. This subpackage refuses to guess it for you. A sweep is the honest
alternative: instead of picking one value and hoping, test the range and read
the curve. The result is meaningful for any format with that exponent width,
independent of what any vendor chose.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import gfloat
from pydantic import BaseModel

from shapesandstrides.formats.error import ErrorDistribution, error_over
from shapesandstrides.formats.grade import FormatTier
from shapesandstrides.formats.spec import FormatSpec

# Only integer-valued structural parameters make sense to sweep. Sweeping a
# name or a provenance dict is meaningless, and silently accepting it would
# produce a curve with no interpretation.
SWEEPABLE = ("bias", "exponent_bits", "mantissa_bits", "num_high_nans")


class SweepPoint(BaseModel):
    """One setting of the swept parameter, and what it cost.

    ``smallest_normal`` is carried deliberately: for a bias sweep it is the
    floor of the representable window, which is the reason the silent-loss
    count moves. A count without its explanation is not a finding.
    """

    parameter_value: int
    format_name: str
    smallest_normal: float
    max_value: float
    underflowed: int
    overflowed: int
    error: ErrorDistribution


class SweepReport(BaseModel):
    base_format_name: str
    parameter: str
    seed: int
    points: list[SweepPoint] = []
    best_by_silent_loss: int | None = None
    best_by_error: int | None = None


def _variant(base: FormatSpec, parameter: str, value: int) -> FormatSpec:
    """A copy of `base` with one parameter changed.

    Renamed, because it is no longer the format it started as, and a log that
    could not tell the two apart would be useless. Constructed through normal
    validation, so an impossible setting fails here rather than producing
    nonsense downstream.
    """
    data = base.model_dump()
    data[parameter] = value
    data["name"] = f"{base.name}-{parameter}{value}"
    data["notes"] = (
        f"swept variant of {base.name!r}: {parameter}={value}. "
        f"Not a published format."
    )
    # The swept parameter's provenance is now this sweep, not whatever the base
    # claimed, so drop the inherited claim rather than carry a stale one.
    prov = dict(data.get("provenance") or {})
    prov.pop(parameter, None)
    data["provenance"] = prov
    return FormatSpec(**data)


def sweep(
    base: FormatSpec,
    parameter: str,
    values: Iterable[int],
    fn: Callable[..., float | Sequence[float]],
    inputs: Sequence[Sequence[float]],
    *,
    rounding: gfloat.RoundMode = gfloat.RoundMode.TiesToEven,
    seed: int = 0xC0FFEE,
) -> SweepReport:
    """Vary ``parameter`` across ``values`` and measure ``fn`` at each setting.

    An impossible setting raises rather than being skipped: a curve that
    silently omitted the points that did not work would misrepresent what was
    actually tested.

    Every point is grade B by construction. A swept variant is a format nobody
    has validated -- it has no native counterpart to validate against -- so it
    cannot claim the stronger grade.
    """
    if parameter not in FormatSpec.model_fields:
        raise ValueError(
            f"{parameter!r} is not a field of FormatSpec. Sweepable fields: "
            f"{list(SWEEPABLE)}."
        )
    if parameter not in SWEEPABLE:
        raise ValueError(
            f"{parameter!r} cannot be swept: only integer structural parameters "
            f"produce an interpretable curve. Sweepable fields: {list(SWEEPABLE)}."
        )

    vals = list(values)
    if not vals:
        raise ValueError(
            "a sweep needs at least one value. An empty sweep produces an empty "
            "report, which would read as a pass."
        )

    points: list[SweepPoint] = []
    for v in vals:
        spec = _variant(base, parameter, v)
        dist = error_over(
            fn, inputs, spec, rounding=rounding, seed=seed, tier=FormatTier.B
        )
        points.append(
            SweepPoint(
                parameter_value=v,
                format_name=spec.name,
                smallest_normal=spec.smallest_normal,
                max_value=spec.max_value,
                underflowed=dist.input_loss.underflowed,
                overflowed=dist.input_loss.overflowed,
                error=dist,
            )
        )

    # Fewest values destroyed wins. Ties break by lower median error, then by
    # the *lowest* parameter value.
    #
    # That last key matters and is not arbitrary. Raising a bias slides the
    # whole window down: it buys room at the bottom by giving up the ceiling.
    # Once a setting loses nothing, raising it further buys nothing and costs
    # headroom, so among the settings that lose nothing the lowest is the best
    # one. Measured on a 6-exponent-bit format over gradient-magnitude inputs,
    # every bias from 28 upward loses nothing and reports identical error, so
    # without this key "best" would be decided by list order.
    best_loss = min(
        points,
        key=lambda p: (
            p.underflowed + p.overflowed,
            p.error.p50_rel_error,
            p.parameter_value,
        ),
    ).parameter_value
    best_err = min(points, key=lambda p: p.error.p50_rel_error).parameter_value

    return SweepReport(
        base_format_name=base.name,
        parameter=parameter,
        seed=seed,
        points=points,
        best_by_silent_loss=best_loss,
        best_by_error=best_err,
    )
