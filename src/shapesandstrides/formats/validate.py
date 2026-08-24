"""Is the simulator telling the truth?

This subpackage simulates formats in software, including formats nobody outside
a vendor can execute. That simulation cannot be checked against real cbfloat16
-- doing so needs a Cerebras machine. It *can* be checked against fp16, bf16
and fp32, which really exist in torch.

If the simulator reproduces those bit for bit, it has earned the right to be
believed about a format nobody can run. If it does not, every number this
package prints is worthless, and we find out immediately rather than after
somebody else notices.

This is the subpackage's acceptance gate.
"""

from __future__ import annotations

import math

import gfloat
from pydantic import BaseModel

from shapesandstrides.formats.grade import FormatTier
from shapesandstrides.formats.spec import FormatSpec
from shapesandstrides.formats.values import ValueClass, values_for

# Format name -> the torch dtype implementing it natively. Only formats torch
# really provides on CPU appear here. A name absent from this map is not an
# error: it is grade B, which is the normal case for a reconstructed format.
#
# Both spellings are present because gfloat names its predefined IEEE formats
# binary16/binary32/binary64, so FLOAT16.name == "binary16".
NATIVE_EQUIVALENTS: dict[str, str] = {
    "binary16": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "binary32": "float32",
    "float32": "float32",
    "binary64": "float64",
    "float64": "float64",
}

# torch dtype name -> the signed integer dtype of the same width, used to view
# the raw bits.
_BIT_VIEW = {
    "float16": "int16",
    "bfloat16": "int16",
    "float32": "int32",
    "float64": "int64",
}


class Divergence(BaseModel):
    """One value where the simulator and the native dtype disagreed.

    Carries the value class as well as the value, so a failure says what *kind*
    of number broke it -- a subnormal, a tie, something past the ceiling --
    rather than only which number.
    """

    value: float
    value_class: ValueClass
    simulated_bits: str
    native_bits: str
    simulated: float
    native: float


class ValidationReport(BaseModel):
    format_name: str
    native_dtype: str | None
    compared: int
    matched: int
    mismatched: int
    first_divergence: Divergence | None
    passed: bool
    tier: FormatTier
    seed: int


def _native_bits(value: float, dtype_name: str) -> tuple[int, float]:
    """The raw bit pattern torch produces for `value` in `dtype_name`."""
    import torch

    tdtype = getattr(torch, dtype_name)
    idtype = getattr(torch, _BIT_VIEW[dtype_name])
    t = torch.tensor([value], dtype=torch.float64).to(tdtype)
    width = torch.iinfo(idtype).bits
    bits = t.view(idtype).item() & ((1 << width) - 1)
    return bits, t.to(torch.float64).item()


def validate(
    spec: FormatSpec,
    *,
    seed: int = 0xC0FFEE,
    native_dtype: str | None = None,
) -> ValidationReport:
    """Compare ``spec``'s simulation against torch's native dtype, if one exists.

    Returns a report rather than raising when the comparison fails, because the
    report is the useful artifact: a divergence needs to be readable, naming the
    value and the class of value that broke it, not merely fatal.

    ``native_dtype`` overrides the lookup. That is how a deliberately wrong spec
    can be checked against the dtype it claims to reproduce, which is what
    proves the gate is capable of failing.
    """
    if native_dtype is not None and native_dtype not in _BIT_VIEW:
        raise ValueError(
            f"{native_dtype!r} is not a torch dtype this can compare against. "
            f"Valid choices: {sorted(_BIT_VIEW)}."
        )

    dtype_name = native_dtype or NATIVE_EQUIVALENTS.get(spec.name.lower())

    if dtype_name is None:
        # No native counterpart. Normal for a reconstructed format: the absence
        # of stronger evidence, not a failure.
        return ValidationReport(
            format_name=spec.name,
            native_dtype=None,
            compared=0,
            matched=0,
            mismatched=0,
            first_divergence=None,
            passed=True,
            tier=FormatTier.B,
            seed=seed,
        )

    fi = spec.to_gfloat()
    vset = values_for(spec, seed=seed)

    compared = matched = mismatched = 0
    first: Divergence | None = None

    for lv in vset.values:
        v = lv.value
        native_bits, native_val = _native_bits(v, dtype_name)
        sim_val = gfloat.round_float(fi, v, gfloat.RoundMode.TiesToEven, sat=False)
        sim_bits = gfloat.encode_float(fi, sim_val)
        compared += 1

        # A format has many NaN encodings, and which one a cast lands on is not
        # a property worth pinning. Agreeing that a value is NaN is agreement.
        both_nan = math.isnan(sim_val) and math.isnan(native_val)
        if sim_bits == native_bits or both_nan:
            matched += 1
            continue

        mismatched += 1
        if first is None:
            first = Divergence(
                value=v,
                value_class=lv.value_class,
                simulated_bits=hex(sim_bits),
                native_bits=hex(native_bits),
                simulated=sim_val,
                native=native_val,
            )

    passed = mismatched == 0
    return ValidationReport(
        format_name=spec.name,
        native_dtype=dtype_name,
        compared=compared,
        matched=matched,
        mismatched=mismatched,
        first_divergence=first,
        passed=passed,
        # Passing means this exact format was checked against reality, which is
        # the only way to earn A. Failing means no claim from it is valid.
        tier=FormatTier.A if passed else FormatTier.C,
        seed=seed,
    )
