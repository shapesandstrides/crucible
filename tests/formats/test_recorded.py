"""Gradient magnitudes recorded from a real backward pass.

`gradient_like` is a log-uniform guess with no tail. The question it gets used
for -- whether a shifted bias removes the need for loss scaling -- turns
entirely on the tail. So these come from a real transformer's real backward
pass, and the fixture states its own limitations.
"""

import math

import pytest

from shapesandstrides.formats.error import (
    gradient_like,
    recorded_gradients,
    recorded_gradient_provenance,
)


def test_the_recorded_gradients_load():
    xs = recorded_gradients()
    assert len(xs) > 1000
    assert all(math.isfinite(x) and x > 0 for x in xs)


def test_they_reach_lower_than_the_synthetic_guess():
    """The whole reason for recording them. A log-uniform spread has no tail,
    and the tail is the entire question.

    Measured: 224 of 841,471 real gradients lie below 1e-11, which is
    gradient_like's floor -- so the synthetic generator had exactly zero values
    in the region that decides the answer.
    """
    real = min(recorded_gradients(tail=True))
    synthetic = min(abs(x) for x in gradient_like(5000, seed=1))
    assert real < synthetic / 10, (
        f"recorded {real:.3e} should reach well below synthetic {synthetic:.3e}"
    )


def test_the_tail_sample_is_separate_from_the_representative_one():
    """Uniform sampling cannot preserve a 0.03% tail, so the tail gets its own
    sample -- and must never be mistaken for a distribution."""
    bulk = recorded_gradients(step=400)
    tail = recorded_gradients(step=400, tail=True)
    assert min(tail) < min(bulk), "the tail must reach lower than a uniform sample"
    assert sorted(tail)[len(tail) // 2] < sorted(bulk)[len(bulk) // 2]


def test_exact_tail_counts_come_from_provenance_not_from_a_sample():
    """"What fraction is below X" must be answered from the full set. A sample
    would understate a thin tail."""
    late = [
        s for s in recorded_gradient_provenance()["snapshots"] if s["step"] == 400
    ][0]
    assert late["gradients_nonzero"] > 100_000
    assert late["count_below"]["1e-11"] > 0
    assert late["count_below"]["1e-13"] == 0


def test_a_specific_training_step_can_be_selected():
    """The distribution moves during training -- late gradients are smaller --
    so a caller studying convergence needs to pick."""
    early = recorded_gradients(step=1)
    late = recorded_gradients(step=400)
    assert sorted(late)[len(late) // 2] < sorted(early)[len(early) // 2], (
        "median gradient magnitude should fall as training converges"
    )


def test_an_unknown_step_is_refused_with_the_available_ones():
    with pytest.raises(ValueError, match="Recorded steps"):
        recorded_gradients(step=999)


def test_the_provenance_states_the_limitations_rather_than_hiding_them():
    """A fixture that cannot say what is wrong with it is worse than a labelled
    guess."""
    p = recorded_gradient_provenance()
    assert p["architecture"]["layers"] >= 1
    assert p["seed"]
    assert p["torch_version"]
    limitations = " ".join(p["honest_limitations"]).lower()
    assert "synthetic" in limitations, "the task was synthetic and must say so"
    assert "small" in limitations, "the model was small and must say so"


def test_the_recorded_fraction_below_fp16s_floor_is_reported():
    """The number that motivates the whole exercise."""
    p = recorded_gradient_provenance()
    late = [s for s in p["snapshots"] if s["step"] == 400][0]
    assert 0.0 < late["fraction_below_fp16_smallest_normal"] < 1.0


def test_it_is_reproducible_and_needs_no_gpu():
    assert recorded_gradients() == recorded_gradients()
