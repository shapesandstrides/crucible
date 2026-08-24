"""A format, and where each of its parameters came from.

The point of this module is that two of the four numbers defining a 16-bit
float are routinely unpublished, so a spec that cannot say where its own values
came from cannot support an honest result.
"""

import math

import pytest
from pydantic import ValidationError

from shapesandstrides.formats import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FormatSpec,
    Provenance,
    ieee_bias,
)


def test_bias_has_no_default():
    """A default here is how a user silently inherits someone else's guess
    about an undocumented format."""
    with pytest.raises(ValidationError):
        FormatSpec(name="mine", exponent_bits=6, mantissa_bits=9)


def test_bias_is_required_structurally():
    """Asserted on the model, because re-adding a default is the exact
    regression that reopens the hole."""
    assert FormatSpec.model_fields["bias"].is_required()


def test_ieee_bias_gives_the_conventional_value():
    """The standard choice must be one call, never a silent default: you still
    typed it, so it is still your stated assumption."""
    assert ieee_bias(5) == 15
    assert ieee_bias(6) == 31
    assert ieee_bias(8) == 127


def test_a_constructed_format_reports_its_derived_limits():
    f = FormatSpec(
        name="cb-6-9-reconstructed",
        exponent_bits=6,
        mantissa_bits=9,
        bias=31,
    )
    assert f.total_bits == 16
    assert f.precision == 10
    assert math.isclose(f.smallest_normal, 9.313225746154785e-10, rel_tol=1e-12)
    assert math.isclose(f.smallest_subnormal, 1.8189894035458565e-12, rel_tol=1e-12)
    assert math.isclose(f.max_value, 4290772992.0, rel_tol=1e-12)
    assert math.isclose(f.eps, 2.0**-9, rel_tol=1e-12)


def test_unstated_provenance_is_visible_not_hidden():
    """Absence of a source must show up as UNSTATED. Silence must never read
    as documentation."""
    f = FormatSpec(name="mine", exponent_bits=6, mantissa_bits=9, bias=31)
    assert f.provenance_of("bias") is Provenance.UNSTATED


def test_stated_provenance_is_recorded_per_field():
    f = FormatSpec(
        name="mine",
        exponent_bits=6,
        mantissa_bits=9,
        bias=31,
        provenance={
            "exponent_bits": Provenance.DOCUMENTED,
            "mantissa_bits": Provenance.DOCUMENTED,
            "bias": Provenance.ASSUMED,
        },
        notes="6/9 from Cerebras docs; bias is my assumption.",
    )
    assert f.provenance_of("exponent_bits") is Provenance.DOCUMENTED
    assert f.provenance_of("bias") is Provenance.ASSUMED
    assert f.provenance_of("has_subnormals") is Provenance.UNSTATED


def test_shipped_constants_are_fully_sourced():
    """A shipped constant carries this project's authority, so every parameter
    of one must cite a standard."""
    for f in (FLOAT16, BFLOAT16, FLOAT32):
        for field in ("exponent_bits", "mantissa_bits", "bias"):
            assert f.provenance_of(field) is Provenance.DOCUMENTED, (
                f"{f.name}.{field} is shipped without a source"
            )


def test_shipped_constants_match_the_standards():
    assert (FLOAT16.exponent_bits, FLOAT16.mantissa_bits, FLOAT16.bias) == (5, 10, 15)
    assert (BFLOAT16.exponent_bits, BFLOAT16.mantissa_bits, BFLOAT16.bias) == (8, 7, 127)
    assert (FLOAT32.exponent_bits, FLOAT32.mantissa_bits, FLOAT32.bias) == (8, 23, 127)


def test_round_trips_through_gfloat():
    f = FormatSpec(name="mine", exponent_bits=6, mantissa_bits=9, bias=31)
    fi = f.to_gfloat()
    assert (fi.k, fi.precision, fi.bias) == (16, 10, 31)
    back = FormatSpec.from_gfloat(fi)
    assert (back.exponent_bits, back.mantissa_bits, back.bias) == (6, 9, 31)


def test_impossible_specs_fail_at_construction_not_at_first_use():
    with pytest.raises(ValidationError):
        FormatSpec(name="zero-exp", exponent_bits=0, mantissa_bits=9, bias=31)
    with pytest.raises(ValidationError):
        FormatSpec(name="neg-mant", exponent_bits=6, mantissa_bits=-1, bias=31)
    with pytest.raises(ValidationError):
        FormatSpec(name="bias-too-big", exponent_bits=6, mantissa_bits=9, bias=999)


def test_a_user_format_may_not_squat_a_shipped_name():
    """Reusing a shipped name would make an unsourced spec indistinguishable
    from a sourced one in any log."""
    with pytest.raises(ValidationError):
        FormatSpec(name="float16", exponent_bits=6, mantissa_bits=9, bias=31)
    with pytest.raises(ValidationError):
        FormatSpec(name="binary16", exponent_bits=6, mantissa_bits=9, bias=31)


def test_the_spec_serialises_cleanly_for_agents():
    f = FormatSpec(
        name="mine",
        exponent_bits=6,
        mantissa_bits=9,
        bias=31,
        provenance={"bias": Provenance.ASSUMED},
    )
    d = f.model_dump(mode="json")
    assert d["bias"] == 31
    assert d["provenance"]["bias"] == "assumed"


def test_gfloat_is_not_a_top_level_import_for_the_kernel_half():
    """Someone using only the kernel half must not pay for a numerics
    dependency."""
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-c", "import shapesandstrides, sys; "
                              "sys.exit(1 if 'gfloat' in sys.modules else 0)"],
        capture_output=True,
    )
    assert r.returncode == 0, "importing shapesandstrides pulled in gfloat"
