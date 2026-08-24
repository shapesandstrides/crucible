"""Small reference calculations to measure a format against.

Deliberately plain Python floats: no numpy, no torch, matching ``stats.py``.
These are the *exact* reference when called normally, so they need to be
obviously correct rather than fast. Anything clever here would be a bug hiding
inside the thing that decides whether other things are buggy.

## The ``q`` parameter

Every op takes ``q``, a function applied after each elementary operation.

- ``q`` left at its default is the identity, so the op computes exactly in
  double precision. That is the reference.
- ``q`` set to a format's rounding function makes the op round after *every*
  step, the way real silicon does.

One implementation, two behaviours. The alternative -- a separate quantised
copy of each op -- would let the reference and the thing being measured drift
apart, which is the one failure this arrangement cannot afford.

Reductions are the interesting cases, because a reduction is where rounding
error compounds rather than merely occurring once.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

# A rounding function. The default leaves the value alone.
Quantizer = Callable[[float], float]


def exact(v: float) -> float:
    """The identity. Named, so `q=exact` reads as a deliberate choice."""
    return v


def total(xs: Sequence[float], q: Quantizer = exact) -> float:
    """Sum. Named `total` so it does not shadow the builtin.

    The accumulator is rounded after every addition, which is where a long sum
    in a coarse format loses far more than a single rounding would suggest.
    """
    acc = q(0.0)
    for x in xs:
        acc = q(acc + q(x))
    return acc


def dot(a: Sequence[float], b: Sequence[float], q: Quantizer = exact) -> float:
    if len(a) != len(b):
        raise ValueError(f"dot needs equal lengths, got {len(a)} and {len(b)}")
    acc = q(0.0)
    for x, y in zip(a, b):
        acc = q(acc + q(q(x) * q(y)))
    return acc


def softmax(xs: Sequence[float], q: Quantizer = exact) -> list[float]:
    """Max-subtracted, as every real implementation is.

    Without the shift, exp() overflows for perfectly ordinary inputs and we
    would be measuring that rather than the format.
    """
    if not xs:
        return []
    m = max(xs)
    exps = [q(math.exp(q(x - m))) for x in xs]
    s = total(exps, q)
    return [q(e / s) for e in exps]


def layernorm(xs: Sequence[float], eps: float = 1e-5, q: Quantizer = exact) -> list[float]:
    n = len(xs)
    if n == 0:
        return []
    mean = q(total(xs, q) / n)
    var = q(total([q(q(x - mean) ** 2) for x in xs], q) / n)
    denom = q(math.sqrt(q(var + eps)))
    return [q(q(x - mean) / denom) for x in xs]
