"""Rounding after every operation, not just at the edges.

STORAGE_ONLY quantises the inputs and the output and computes exactly in
between, which understates real hardware. EVERY_STEP rounds after each
elementary operation, the way silicon does. The gap between the two is the
error that storage-only was hiding.
"""

import pytest

from shapesandstrides.formats import BFLOAT16, FLOAT16, FLOAT32
from shapesandstrides.formats.error import QuantizationModel, error_over, gradient_like
from shapesandstrides.formats.ops import dot, layernorm, softmax, total


def test_every_step_is_a_distinct_recorded_model():
    d = error_over(total, [[1.0, 2.0, 3.0]], FLOAT16, model=QuantizationModel.EVERY_STEP)
    assert d.quantization_model is QuantizationModel.EVERY_STEP


def test_rounding_every_step_finds_more_error_than_rounding_only_the_edges():
    """The whole reason this exists. A long accumulation in a coarse format
    loses precision at every addition, and storage-only cannot see it."""
    xs = [0.1] * 500
    storage = error_over(total, [xs], BFLOAT16)
    stepwise = error_over(total, [xs], BFLOAT16, model=QuantizationModel.EVERY_STEP)
    assert stepwise.max_rel_error > storage.max_rel_error * 5, (
        f"storage-only {storage.max_rel_error:.3e} should badly understate "
        f"step-wise {stepwise.max_rel_error:.3e} over a 500-term sum"
    )


def test_a_longer_accumulation_compounds_more_error():
    """Error growth through a reduction is the thing storage-only cannot model
    at all: it reports the same figure however long the sum."""
    short = error_over(
        total, [[0.1] * 10], BFLOAT16, model=QuantizationModel.EVERY_STEP
    )
    long = error_over(
        total, [[0.1] * 1000], BFLOAT16, model=QuantizationModel.EVERY_STEP
    )
    assert long.max_rel_error > short.max_rel_error


def test_an_exact_format_still_has_no_error_step_wise():
    """A sanity check on the machinery: powers of two in fp32 must survive
    every intermediate rounding too."""
    d = error_over(
        total, [[1.0, 2.0, 4.0]], FLOAT32, model=QuantizationModel.EVERY_STEP
    )
    assert d.max_rel_error == 0.0


def test_the_built_in_ops_all_support_step_wise():
    xs = gradient_like(32, seed=1)
    for fn, args in ((total, [xs]), (dot, [xs, xs]), (softmax, [xs]), (layernorm, [xs])):
        d = error_over(fn, args, FLOAT16, model=QuantizationModel.EVERY_STEP)
        assert d.quantization_model is QuantizationModel.EVERY_STEP


def test_a_callable_that_cannot_round_internally_is_refused_with_a_remedy():
    """Silently falling back to storage-only would report a stronger model than
    was actually used, which is the one thing this label exists to prevent."""

    def opaque(xs):
        return sum(xs)

    with pytest.raises(ValueError, match="cannot round its intermediates"):
        error_over(opaque, [[1.0, 2.0]], FLOAT16, model=QuantizationModel.EVERY_STEP)


def test_storage_only_remains_the_default():
    """Changing the default would silently change every existing result."""
    d = error_over(total, [[1.0, 2.0]], FLOAT16)
    assert d.quantization_model is QuantizationModel.STORAGE_ONLY


def test_the_model_appears_in_json():
    d = error_over(
        total, [[1.0, 2.0]], FLOAT16, model=QuantizationModel.EVERY_STEP
    ).model_dump(mode="json")
    assert d["quantization_model"] == "every_step"
