"""THE ACCEPTANCE GATE.

A software simulation of a format nobody outside a vendor can execute is worth
nothing unless it reproduces the formats that *do* exist. We cannot check a
reconstructed cbfloat16 against real cbfloat16 -- that needs a Cerebras
machine. We can check the simulator against torch's own fp16, bf16 and fp32.

If these tests fail, every number this subpackage prints is worthless. They
exist so that is discovered on day one rather than by a stranger later.
"""

import pytest

from shapesandstrides.formats import BFLOAT16, FLOAT16, FLOAT32, FormatSpec, ieee_bias
from shapesandstrides.formats.grade import FormatTier
from shapesandstrides.formats.validate import validate

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("spec", [FLOAT16, BFLOAT16, FLOAT32], ids=lambda s: s.name)
def test_the_simulator_reproduces_a_real_dtype_bit_for_bit(spec):
    r = validate(spec)
    assert r.mismatched == 0, f"first divergence: {r.first_divergence}"
    assert r.passed is True
    assert r.compared > 50, "too few values compared to mean anything"


def test_validation_earns_grade_a():
    assert validate(FLOAT16).tier is FormatTier.A


def test_a_format_with_no_native_counterpart_is_grade_b_not_an_error():
    """The normal case for a reconstructed format. Having no counterpart is the
    absence of stronger evidence, not a failure."""
    cb = FormatSpec(
        name="cb-6-9", exponent_bits=6, mantissa_bits=9, bias=ieee_bias(6)
    )
    r = validate(cb)
    assert r.tier is FormatTier.B
    assert r.native_dtype is None
    assert r.compared == 0
    assert r.passed is True


def test_a_deliberately_wrong_spec_fails_validation_and_grades_c():
    """Proof the gate can actually fail. A gate that cannot fail is not a gate:
    fp16's widths with the wrong bias must not reproduce torch.float16."""
    wrong = FormatSpec(
        name="fp16-wrong-bias",
        exponent_bits=5,
        mantissa_bits=10,
        bias=14,
        notes="deliberately wrong, to prove the gate can fail",
    )
    r = validate(wrong, native_dtype="float16")
    assert r.passed is False
    assert r.mismatched > 0
    assert r.tier is FormatTier.C
    assert r.first_divergence is not None
    assert r.first_divergence.value_class is not None


def test_a_failing_report_names_the_value_class_that_diverged():
    """A divergence should say what kind of number broke it, not just which
    number."""
    wrong = FormatSpec(
        name="fp16-wrong-mantissa", exponent_bits=5, mantissa_bits=9, bias=15
    )
    r = validate(wrong, native_dtype="float16")
    assert r.passed is False
    assert r.first_divergence.simulated_bits != r.first_divergence.native_bits


def test_the_report_records_its_seed():
    assert validate(FLOAT16, seed=123).seed == 123


def test_the_report_serialises_for_agents():
    d = validate(FLOAT16).model_dump(mode="json")
    assert d["passed"] is True
    assert d["tier"] == "A"
    assert d["native_dtype"] == "float16"


def test_an_unknown_native_dtype_is_an_error_with_a_remedy():
    with pytest.raises(ValueError, match="not a torch dtype this can compare"):
        validate(FLOAT16, native_dtype="float123")
