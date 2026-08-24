"""What a format costs a real calculation.

A format's static properties say nothing about what it costs in use, because
error compounds through a reduction. These tests pin the distribution, the
silent-loss census, and above all the honesty label: this models storage
precision only, not full arithmetic.
"""

import math

import pytest

from shapesandstrides.formats import BFLOAT16, FLOAT16, FLOAT32, FormatSpec, ieee_bias
from shapesandstrides.formats.error import (
    QuantizationModel,
    error_over,
    gradient_like,
)
from shapesandstrides.formats.grade import FormatTier
from shapesandstrides.formats.ops import dot, layernorm, softmax, total


def test_the_quantization_model_is_recorded_on_every_result():
    """Storage-only understates real hardware error, which rounds every
    intermediate. A result that did not say so would be misread."""
    d = error_over(total, [[1.0, 2.0, 3.0]], FLOAT16)
    assert d.quantization_model is QuantizationModel.STORAGE_ONLY


def test_a_format_that_represents_the_inputs_exactly_has_no_error():
    """Powers of two through fp32: nothing should move."""
    d = error_over(total, [[1.0, 2.0, 4.0, 8.0]], FLOAT32)
    assert d.max_rel_error == 0.0
    assert d.p50_rel_error == 0.0


def test_a_coarser_format_has_more_error_than_a_finer_one():
    xs = [0.1, 0.3, 0.7, 1.1, 2.9, 5.3]
    fine = error_over(dot, [xs, xs], FLOAT16)
    coarse = error_over(dot, [xs, xs], BFLOAT16)
    assert coarse.max_rel_error > fine.max_rel_error, (
        "bf16 has 3 fewer significand bits than fp16, so it must be worse here"
    )


def test_the_distribution_reports_percentiles_and_an_interval_not_a_mean():
    """Never report a bare number."""
    xs = gradient_like(200, seed=1)
    d = error_over(softmax, [xs], FLOAT16)
    assert d.p50_rel_error <= d.p90_rel_error <= d.p99_rel_error
    assert d.p99_rel_error <= d.max_rel_error
    assert d.ci95_lo <= d.ci95_hi
    assert d.n == 200


def test_the_distribution_cannot_collapse_to_a_bare_float():
    """Rule 2 of the parent project, enforced structurally."""
    d = error_over(total, [[1.0, 2.0]], FLOAT16)
    with pytest.raises(TypeError):
        float(d)


def test_silent_underflow_is_counted_at_the_input():
    """The headline measurement. Gradients this small vanish in fp16, and the
    count is the signal that makes the bias question answerable."""
    tiny = [1e-8] * 50
    d = error_over(total, [tiny], FLOAT16)
    assert d.input_loss.underflowed == 50
    assert d.input_loss.total == 50


def test_nothing_underflows_in_a_format_with_room_for_it():
    cb = FormatSpec(name="cb-6-9", exponent_bits=6, mantissa_bits=9, bias=ieee_bias(6))
    d = error_over(total, [[1e-8] * 50], cb)
    assert d.input_loss.underflowed == 0


def test_overflow_is_counted_separately_from_underflow():
    d = error_over(total, [[1e30] * 4], FLOAT16)
    assert d.input_loss.overflowed == 4
    assert d.input_loss.underflowed == 0


def test_a_multi_element_output_is_measured_elementwise():
    xs = [1.0, 2.0, 3.0, 4.0]
    d = error_over(layernorm, [xs], FLOAT16)
    assert d.n == 4


def test_the_result_carries_a_grade_defaulting_to_the_weaker_claim():
    d = error_over(total, [[1.0, 2.0]], FLOAT16)
    assert d.format_tier is FormatTier.B
    d2 = error_over(total, [[1.0, 2.0]], FLOAT16, tier=FormatTier.A)
    assert d2.format_tier is FormatTier.A


def test_the_same_seed_gives_the_same_interval():
    xs = gradient_like(64, seed=5)
    a = error_over(softmax, [xs], FLOAT16, seed=3)
    b = error_over(softmax, [xs], FLOAT16, seed=3)
    assert (a.ci95_lo, a.ci95_hi) == (b.ci95_lo, b.ci95_hi)


def test_it_serialises_for_agents():
    d = error_over(total, [[1.0, 2.0]], FLOAT16).model_dump(mode="json")
    assert d["quantization_model"] == "storage_only"
    assert d["format_tier"] == "B"
    assert "p90_rel_error" in d
    assert d["input_loss"]["underflowed"] == 0


def test_gradient_like_states_its_own_assumption():
    """Synthetic inputs are an assumption about the world, so they have to be
    as labelled as an assumed format parameter."""
    xs = gradient_like(500, seed=2)
    assert len(xs) == 500
    assert all(math.isfinite(x) for x in xs)
    # Magnitudes should sit in the region where fp16 is in trouble.
    tiny = sum(1 for x in xs if abs(x) < 6.1e-5)
    assert tiny > 250, "the point of this generator is gradients near fp16's floor"


def test_gradient_like_is_reproducible():
    assert gradient_like(20, seed=9) == gradient_like(20, seed=9)
    assert gradient_like(20, seed=9) != gradient_like(20, seed=10)


def test_an_empty_input_is_an_error_with_a_remedy():
    with pytest.raises(ValueError, match="at least"):
        error_over(total, [[]], FLOAT16)
