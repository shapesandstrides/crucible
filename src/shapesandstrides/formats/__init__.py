"""A lab for numeric formats.

Declare a format by its parameters -- including formats no vendor has fully
published -- and find out honestly what it does to real numbers.

``gfloat`` (MIT) is the numeric core and is never reimplemented here. This
package is the harness it lacks: the provenance of every parameter, the
corner-case values that actually break formats, validation of the simulator
against dtypes that really exist, and an evidence grade on every result.

Deliberately absent: any constant named after a format whose parameters are not
fully published. See ``docs/guide/formats-cbfloat16.md``.
"""

from shapesandstrides.formats.grade import FormatTier
from shapesandstrides.formats.roundtrip import (
    Outcome,
    RoundTripReport,
    round_trip,
)
from shapesandstrides.formats.spec import (
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT8_E5M2,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    FormatSpec,
    Provenance,
    ieee_bias,
)
from shapesandstrides.formats.values import ValueClass, ValueSet, values_for

__all__ = [
    "FormatSpec",
    "FormatTier",
    "Outcome",
    "RoundTripReport",
    "Provenance",
    "ValueClass",
    "ValueSet",
    "ieee_bias",
    "round_trip",
    "values_for",
    "FLOAT64",
    "FLOAT32",
    "FLOAT16",
    "BFLOAT16",
    "FLOAT8_E4M3",
    "FLOAT8_E5M2",
]
