"""What a format costs a real calculation.

A format's static limits say nothing about what it costs in use. Error compounds
through a reduction, and a value that underflowed on the way in is gone before
the arithmetic even starts.

## The honesty problem, stated in the open

There are two models of "computing in a format", and they give very different
answers, so which one produced a number has to travel with it.

**STORAGE_ONLY** rounds the inputs and the output and computes exactly in
between. Real hardware rounds every intermediate, so numbers from this model are
a **lower bound** on real error. Worse, they do not grow with the length of a
reduction -- a 10-term and a 10,000-term sum report the same figure, which is
plainly wrong. It remains the default only because changing a default silently
restates every result anyone already has.

**EVERY_STEP** rounds after each elementary operation, the way silicon does.
This is the model that shows error compounding through a reduction. It needs an
op that accepts a ``q`` parameter; ``ops.py`` provides them.

The gap between the two is not noise -- it is precisely the error storage-only
was concealing. The silent-loss census, by contrast, is exact under both models:
values are destroyed at storage time, before any arithmetic happens.

So ``QuantizationModel`` is recorded on every result, and the caller chooses.
``EVERY_STEP`` rounds after each elementary operation, the way silicon does, and
is available for any op that accepts a ``q`` parameter (see ``ops.py``). The
difference between the two models is itself informative: it is exactly the error
that storage-only was hiding.
"""

from __future__ import annotations

import inspect
import math
import random
from enum import Enum
from typing import Callable, Sequence

import gfloat
from pydantic import BaseModel

from shapesandstrides.formats.grade import FormatTier, Graded
from shapesandstrides.formats.spec import FormatSpec
from shapesandstrides.stats import bootstrap_ci, percentile


class QuantizationModel(str, Enum):
    """How much of the calculation was actually done in the format.

    STORAGE_ONLY: inputs and outputs rounded into the format, the calculation
        itself exact in double precision. Intermediates are NOT rounded, so
        reported error is a lower bound on real hardware. The silent-loss
        census, however, is exact -- values are lost at storage time.

    EVERY_STEP: rounded after every elementary operation, the way silicon does.
        Requires an op that accepts a `q` parameter -- see `ops.py`. This is the
        honest model for error growth through a reduction; STORAGE_ONLY reports
        the same figure however long the sum, which is plainly wrong.

    STORAGE_ONLY remains the default: changing it would silently restate every
    result anyone had already recorded.
    """

    STORAGE_ONLY = "storage_only"
    EVERY_STEP = "every_step"


class SilentLoss(BaseModel):
    """A census of values that stopped being themselves.

    Underflow is the one that matters. Overflow announces itself as an infinity
    and something downstream will notice. A value that became zero raises
    nothing, and the gradient it represented simply stops existing.
    """

    total: int
    underflowed: int
    overflowed: int
    became_nan: int


class ErrorDistribution(Graded):
    """What a format cost one calculation.

    Deliberately defines no ``__float__``: this carries a distribution, and a
    caller must not be able to collapse it into a single number. Same rule, and
    same reason, as ``TimingResult``.
    """

    format_name: str
    quantization_model: QuantizationModel
    n: int
    p50_rel_error: float
    p90_rel_error: float
    p99_rel_error: float
    max_rel_error: float
    ci95_lo: float
    ci95_hi: float
    input_loss: SilentLoss
    output_loss: SilentLoss
    seed: int


def gradient_like(n: int, *, seed: int = 0xC0FFEE) -> list[float]:
    """Synthetic values with the rough magnitude spread of neural-net gradients.

    **This is an assumption about the world, not a measurement of one.** No
    model was trained to obtain it. It is a log-uniform spread of magnitudes
    from about 1e-11 to 1e-3 with random signs, chosen because that band
    straddles fp16's floor (6.1e-5) -- which is the region the interesting
    question lives in.

    Treat any conclusion resting on it exactly as you would treat a format
    parameter marked ASSUMED. Replacing this with magnitudes recorded from a
    real training run would strengthen every result that uses it, and is
    outstanding work.
    """
    if n < 1:
        raise ValueError("gradient_like needs at least 1 value")
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(n):
        exponent = rng.uniform(-11.0, -3.0)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        out.append(sign * (10.0**exponent))
    return out


def _quantize(values: Sequence[float], fi: gfloat.FormatInfo, rnd) -> tuple[list[float], SilentLoss]:
    """Round every value into the format, counting what was destroyed."""
    out: list[float] = []
    under = over = nan = 0
    for v in values:
        r = gfloat.round_float(fi, float(v), rnd, sat=False)
        if math.isnan(r) and not math.isnan(v):
            nan += 1
        elif math.isinf(r) and not math.isinf(v):
            over += 1
        elif r == 0.0 and v != 0.0:
            under += 1
        out.append(r)
    return out, SilentLoss(
        total=len(out), underflowed=under, overflowed=over, became_nan=nan
    )


def _quantizer(fi: gfloat.FormatInfo, rnd) -> Callable[[float], float]:
    """A rounding function for one format, for ops to apply after every step."""

    def q(v: float) -> float:
        return gfloat.round_float(fi, float(v), rnd, sat=False)

    return q


def _as_list(x: float | Sequence[float]) -> list[float]:
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


def error_over(
    fn: Callable[..., float | Sequence[float]],
    inputs: Sequence[Sequence[float]],
    fmt: FormatSpec,
    *,
    rounding: gfloat.RoundMode = gfloat.RoundMode.TiesToEven,
    model: QuantizationModel = QuantizationModel.STORAGE_ONLY,
    seed: int = 0xC0FFEE,
    tier: FormatTier = FormatTier.B,
) -> ErrorDistribution:
    """Measure what ``fmt`` costs ``fn`` on ``inputs``.

    The reference is ``fn`` applied to the original values in double precision:
    the question being asked is "what did using this format cost me, compared to
    not using it?".

    ``tier`` defaults to B -- unvalidated -- so forgetting to validate can only
    under-claim.
    """
    if not inputs or any(len(a) == 0 for a in inputs):
        raise ValueError(
            "error_over needs at least 1 value in every input. An empty input "
            "produces an empty distribution, which would report as a pass."
        )

    fi = fmt.to_gfloat()

    quantized_inputs: list[list[float]] = []
    losses: list[SilentLoss] = []
    for arg in inputs:
        q, loss = _quantize(arg, fi, rounding)
        quantized_inputs.append(q)
        losses.append(loss)

    input_loss = SilentLoss(
        total=sum(l.total for l in losses),
        underflowed=sum(l.underflowed for l in losses),
        overflowed=sum(l.overflowed for l in losses),
        became_nan=sum(l.became_nan for l in losses),
    )

    # The reference is always exact: the question is what the format cost,
    # compared to not using it.
    reference = _as_list(fn(*inputs))

    if model is QuantizationModel.EVERY_STEP:
        if "q" not in inspect.signature(fn).parameters:
            raise ValueError(
                f"{getattr(fn, '__name__', fn)!r} cannot round its intermediates: "
                f"it takes no 'q' parameter, so EVERY_STEP is not available for "
                f"it. Use one of the ops in shapesandstrides.formats.ops, give "
                f"your function a 'q' parameter applied after each operation, or "
                f"pass model=QuantizationModel.STORAGE_ONLY and read the result "
                f"as a lower bound."
            )
        q = _quantizer(fi, rounding)
        computed = _as_list(fn(*quantized_inputs, q=q))
    else:
        computed = _as_list(fn(*quantized_inputs))

    got, output_loss = _quantize(computed, fi, rounding)

    rel: list[float] = []
    for want, have in zip(reference, got):
        if not (math.isfinite(want) and math.isfinite(have)):
            # Non-finite pairs carry no meaningful relative error; the loss
            # census already records that something went wrong.
            continue
        # Guard the denominator rather than dividing by zero, as
        # oracle.compare_against_oracle does.
        rel.append(abs(have - want) / max(abs(want), 1e-300))

    if not rel:
        rel = [0.0, 0.0]
    if len(rel) < 2:
        # bootstrap_ci needs two samples, and a one-element output is a
        # legitimate case (a dot product). Duplicating the single observation
        # gives a degenerate interval, which is honest: one sample supports no
        # interval, and lo == hi says exactly that.
        rel = rel * 2

    lo, hi = bootstrap_ci(rel, seed=seed)
    return ErrorDistribution(
        format_name=fmt.name,
        quantization_model=model,
        n=len(reference),
        p50_rel_error=percentile(rel, 0.50),
        p90_rel_error=percentile(rel, 0.90),
        p99_rel_error=percentile(rel, 0.99),
        max_rel_error=max(rel),
        ci95_lo=lo,
        ci95_hi=hi,
        input_loss=input_loss,
        output_loss=output_loss,
        seed=seed,
        format_tier=tier,
    )
