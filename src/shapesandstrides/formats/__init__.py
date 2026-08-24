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

from shapesandstrides.formats.equivalence import (
    EquivalenceReport,
    loss_scaling_equivalence,
)
from shapesandstrides.formats.error import (
    ErrorDistribution,
    QuantizationModel,
    SilentLoss,
    error_over,
    gradient_like,
    recorded_gradient_provenance,
    recorded_gradients,
)
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
from shapesandstrides.formats.sweep import SweepPoint, SweepReport, sweep
from shapesandstrides.formats.validate import ValidationReport, validate
from shapesandstrides.formats.values import ValueClass, ValueSet, values_for

__all__ = [
    "EquivalenceReport",
    "ErrorDistribution",
    "FormatSpec",
    "FormatTier",
    "Outcome",
    "QuantizationModel",
    "RoundTripReport",
    "SilentLoss",
    "SweepPoint",
    "SweepReport",
    "Provenance",
    "ValidationReport",
    "ValueClass",
    "ValueSet",
    "error_over",
    "gradient_like",
    "loss_scaling_equivalence",
    "recorded_gradient_provenance",
    "recorded_gradients",
    "ieee_bias",
    "round_trip",
    "sweep",
    "validate",
    "values_for",
    "FLOAT64",
    "FLOAT32",
    "FLOAT16",
    "BFLOAT16",
    "FLOAT8_E4M3",
    "FLOAT8_E5M2",
]
