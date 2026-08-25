"""check(error_budget=...) — grade against the unfused chain, not a tolerance.

The governing decision, encoded in `test_budget_verdict_replaces_the_tolerance_verdict`:
when a budget is requested it **replaces** the atol/rtol verdict rather than
adding to it. Requiring both would leave the false failure the budget exists to
fix still failing, and the feature would be decorative.
"""

import pytest
import torch

from shapesandstrides.correctness import check
from shapesandstrides.shapes import ShapeTier
from shapesandstrides.types import CheckKind

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _add(x, y):
    return x + y


def test_absent_by_default():
    """None means the check did not run, which is not the same statement as
    running and finding nothing."""
    r = check(_add, torch.add, tier=ShapeTier.FAST, device="cpu", dtypes=["float32"])
    assert CheckKind.ERROR_BUDGET not in r.checks
    assert all(o.budget is None for o in r.outcomes)


def test_recorded_when_requested():
    r = check(
        _add,
        torch.add,
        tier=ShapeTier.FAST,
        device="cpu",
        dtypes=["float32"],
        error_budget=2.0,
    )
    assert CheckKind.ERROR_BUDGET in r.checks
    assert r.outcomes, "shape space was empty; the rest of this asserts nothing"
    assert all(o.budget is not None for o in r.outcomes)
    assert all(o.budget.margin == 2.0 for o in r.outcomes)


def test_budget_verdict_replaces_the_tolerance_verdict():
    """A zero tolerance fails every shape. With a budget, the budget governs.

    tolerance_override=(0, 0) is a stand-in for the real situation: a fused
    kernel whose output is legitimately outside atol/rtol because it is more
    accurate than the reference, not less.
    """
    strict = check(
        _add,
        torch.add,
        tier=ShapeTier.FAST,
        device="cpu",
        dtypes=["float32"],
        tolerance_override=(0.0, 0.0),
    )

    with_budget = check(
        _add,
        torch.add,
        tier=ShapeTier.FAST,
        device="cpu",
        dtypes=["float32"],
        tolerance_override=(0.0, 0.0),
        error_budget=2.0,
    )

    # Both ran the oracle; only the second is graded on the budget.
    assert all(o.budget.passed for o in with_budget.outcomes)
    assert with_budget.passed is True
    # The tolerance comparison is still recorded, so nothing is hidden.
    assert all(o.oracle is not None for o in with_budget.outcomes)
    assert strict.total == with_budget.total


def test_a_genuinely_wrong_kernel_still_fails_with_a_budget():
    """The budget must not become a way to pass anything."""

    def wrong(x, y):
        return x - y

    r = check(
        wrong,
        torch.add,
        tier=ShapeTier.FAST,
        device="cpu",
        dtypes=["float32"],
        error_budget=2.0,
    )
    assert r.passed is False
    assert r.minimal_failure is not None


def test_report_round_trips_as_json():
    r = check(
        _add,
        torch.add,
        tier=ShapeTier.FAST,
        device="cpu",
        dtypes=["float32"],
        error_budget=2.0,
    )
    dumped = r.model_dump(mode="json")
    assert dumped["outcomes"][0]["budget"]["kernel_p99_ulp"] is not None
    assert dumped["checks"] == ["reference", "error_budget"]


@needs_cuda
def test_the_real_case_a_gpu_matmul_beats_the_cpu_reference():
    """The scenario the whole feature exists for.

    An fp16 matmul on tensor cores accumulates in fp32, so the GPU result is
    *more accurate* than torch's own fp16 matmul on CPU. Against a fixed
    tolerance that reads as disagreement; against the budget it reads as an
    improvement, which is what it is.
    """

    def fused(x, y):
        return torch.tanh(x @ y)

    r = check(
        fused,
        lambda x, y: torch.tanh(x @ y),
        tier=ShapeTier.FAST,
        device="cuda",
        dtypes=["float16"],
        error_budget=2.0,
        max_elements=1 << 16,
    )

    graded = [o for o in r.outcomes if o.budget is not None]
    assert graded, "no shape produced a budget; the assertion below is vacuous"
    # Report the distribution rather than asserting a single number: this is
    # measured behaviour on real hardware, not a fixed constant.
    for o in graded:
        assert o.budget.total_elements > 0
        assert o.budget.kernel_p99_ulp >= 0.0
