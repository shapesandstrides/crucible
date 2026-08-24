"""Small reference calculations to measure a format against.

Deliberately plain Python floats: no numpy, no torch, matching ``stats.py``.
These are the *exact* reference, computed in double precision, so they need to
be obviously correct rather than fast. Anything clever here would be a bug
hiding in the thing that decides whether other things are buggy.

Reductions are the interesting cases, because that is where error compounds.
"""

from __future__ import annotations

import math
from typing import Sequence


def total(xs: Sequence[float]) -> float:
    """Sum. Named `total` so it does not shadow the builtin."""
    acc = 0.0
    for x in xs:
        acc += x
    return acc


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dot needs equal lengths, got {len(a)} and {len(b)}")
    acc = 0.0
    for x, y in zip(a, b):
        acc += x * y
    return acc


def softmax(xs: Sequence[float]) -> list[float]:
    """Max-subtracted, as every real implementation is.

    Without the shift, exp() overflows for perfectly ordinary inputs, and we
    would be measuring that rather than the format.
    """
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = total(exps)
    return [e / s for e in exps]


def layernorm(xs: Sequence[float], eps: float = 1e-5) -> list[float]:
    n = len(xs)
    if n == 0:
        return []
    mean = total(xs) / n
    var = total([(x - mean) ** 2 for x in xs]) / n
    denom = math.sqrt(var + eps)
    return [(x - mean) / denom for x in xs]
