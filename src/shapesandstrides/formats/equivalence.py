"""Is shifting the exponent bias the same thing as loss scaling?

Loss scaling is how mixed-precision training keeps small gradients alive:
multiply the loss by a constant so gradients land inside the representable
window, then divide back out. It costs a multiply and a divide every step, a
scale factor that has to be tuned or adapted, and occasional skipped steps when
the scaled values overflow.

Shifting the exponent bias does the same thing by moving the window instead of
the values. It costs **nothing** -- the bias is only what you subtract when
reading the exponent, and subtracting 36 is the same adder as subtracting 31.

If those two are the same operation, then wherever you control the format the
bias shift is strictly better: identical numerics, none of the machinery.

## What this measures, and what it does not

It compares them **for storage**: round a value into a bias-shifted format,
versus scale the value, round it into the base format, and unscale. Measured on
real recorded gradients plus the boundary values, across several shift sizes,
those are bit-identical -- which stands to reason, since multiplying by a power
of two is exact and the bias-shifted format's grid *is* the base grid scaled.

What it does **not** establish is that a bias shift can replace loss scaling in
a training loop. Two reasons, both real:

- Loss scaling is usually **dynamic**. It tracks the gradient distribution as it
  moves, and it does move: in ``scripts/record_gradients.py`` the median
  gradient magnitude fell about sixfold between step 1 and step 400. A bias is
  fixed when the format is designed and cannot adapt.
- Loss scaling also changes the magnitudes flowing through **intermediate**
  accumulations and optimiser state, not only what gets stored.

So the honest claim is narrow and still useful: a bias shift reproduces the
*static* part of loss scaling exactly, and for free. It does not reproduce the
adaptive part.
"""

from __future__ import annotations

import math
from typing import Sequence

import gfloat
from pydantic import BaseModel

from shapesandstrides.formats.spec import FormatSpec


class Difference(BaseModel):
    value: float
    via_bias_shift: float
    via_loss_scale: float


class EquivalenceReport(BaseModel):
    base_format_name: str
    bias_shift: int
    loss_scale: float
    base_max: float
    shifted_max: float
    compared: int
    identical: int
    differing: int
    first_difference: Difference | None
    equivalent: bool


def loss_scaling_equivalence(
    base: FormatSpec,
    bias_shift: int,
    values: Sequence[float],
    *,
    rounding: gfloat.RoundMode = gfloat.RoundMode.TiesToEven,
    loss_scale: float | None = None,
) -> EquivalenceReport:
    """Compare a bias shift against the loss scale it should be equivalent to.

    A shift of ``+N`` moves the window down by a factor of ``2**N``, so the
    loss scale it corresponds to is ``2**N``. ``loss_scale`` overrides that,
    which is how the check is proven capable of failing: comparing a shift
    against the wrong scale must be caught.

    Returns a report rather than raising on divergence, because a divergence is
    a finding and needs to be readable.
    """
    if bias_shift == 0:
        raise ValueError(
            "bias_shift must be non-zero: comparing a format against itself "
            "establishes nothing."
        )

    new_bias = base.bias + bias_shift
    max_bias = 2**base.exponent_bits - 1
    if not 0 <= new_bias <= max_bias:
        raise ValueError(
            f"bias {base.bias} shifted by {bias_shift} gives {new_bias}, which is "
            f"outside the range an {base.exponent_bits}-bit exponent can express "
            f"(0..{max_bias})."
        )

    shifted = base.model_copy(
        update={
            "name": f"{base.name}-bias{new_bias}",
            "bias": new_bias,
            "notes": f"bias-shifted variant of {base.name!r} by {bias_shift:+d}",
        }
    )

    scale = 2.0**bias_shift if loss_scale is None else loss_scale
    fb, fs = base.to_gfloat(), shifted.to_gfloat()

    compared = identical = differing = 0
    first: Difference | None = None

    for raw in values:
        x = float(raw)
        # Route 1: store x in the format whose window already sits lower.
        via_shift = gfloat.round_float(fs, x, rounding, sat=False)
        # Route 2: scale x up, store it in the original format, scale back.
        via_scale = gfloat.round_float(fb, x * scale, rounding, sat=False) / scale

        compared += 1
        same = via_shift == via_scale or (
            math.isnan(via_shift) and math.isnan(via_scale)
        )
        if same:
            identical += 1
            continue
        differing += 1
        if first is None:
            first = Difference(
                value=x, via_bias_shift=via_shift, via_loss_scale=via_scale
            )

    return EquivalenceReport(
        base_format_name=base.name,
        bias_shift=bias_shift,
        loss_scale=scale,
        base_max=base.max_value,
        shifted_max=shifted.max_value,
        compared=compared,
        identical=identical,
        differing=differing,
        first_difference=first,
        equivalent=differing == 0,
    )
