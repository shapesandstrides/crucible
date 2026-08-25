"""The error budget: is the kernel any worse than the code it replaced?"""

import math

import torch

from shapesandstrides.budget import ErrorBudget, compare_error_budget, ulp_error


def test_identical_tensors_have_zero_ulp_error():
    x = torch.randn(32, dtype=torch.float32)
    assert ulp_error(x, x.to(torch.float64)).max().item() == 0.0


def test_one_representable_step_measures_as_about_one_ulp():
    x = torch.tensor([1.0], dtype=torch.float32)
    nudged = torch.nextafter(x, torch.tensor([2.0], dtype=torch.float32))
    assert 0.9 < ulp_error(nudged, x.to(torch.float64)).max().item() < 1.1


def test_ulp_is_scale_invariant():
    """The point of ULP over absolute error: the same relative wrongness at
    1e-6 and 1e6 must measure the same."""
    small = torch.tensor([1e-6], dtype=torch.float32)
    large = torch.tensor([1e6], dtype=torch.float32)

    small_off = torch.nextafter(small, torch.tensor([1.0], dtype=torch.float32))
    large_off = torch.nextafter(large, torch.tensor([1e7], dtype=torch.float32))

    e_small = ulp_error(small_off, small.to(torch.float64)).item()
    e_large = ulp_error(large_off, large.to(torch.float64)).item()
    assert math.isclose(e_small, e_large, rel_tol=0.2)


def test_kernel_more_accurate_than_reference_passes():
    """The case that motivates the whole module: a fused kernel that beats
    the unfused chain must not be reported as broken."""
    golden = torch.linspace(1.0, 2.0, 256, dtype=torch.float64)
    kernel = (golden + 1e-7).to(torch.float32)
    reference = (golden + 1e-4).to(torch.float32)

    b = compare_error_budget(kernel, reference, golden)
    assert b.passed is True
    assert b.kernel_p99_ulp < b.reference_p99_ulp
    assert b.ratio_p99 is not None and b.ratio_p99 < 1.0


def test_kernel_much_worse_than_reference_fails():
    golden = torch.linspace(1.0, 2.0, 256, dtype=torch.float64)
    kernel = (golden + 1e-2).to(torch.float32)
    reference = (golden + 1e-7).to(torch.float32)

    b = compare_error_budget(kernel, reference, golden)
    assert b.passed is False
    assert b.ratio_p99 is not None and b.ratio_p99 > 1.0


def test_margin_is_honoured_and_reported():
    golden = torch.linspace(1.0, 2.0, 256, dtype=torch.float64)
    kernel = (golden + 3e-7).to(torch.float32)
    reference = (golden + 1e-7).to(torch.float32)

    assert compare_error_budget(kernel, reference, golden, margin=1.5).passed is False
    generous = compare_error_budget(kernel, reference, golden, margin=100.0)
    assert generous.passed is True
    assert generous.margin == 100.0


def test_bit_exact_reference_does_not_divide_by_zero():
    """A zero denominator must not produce inf: pydantic serialises inf to
    JSON null and validation then rejects null for a required float.

    Small integers are exactly representable in float32, so casting the golden
    down is lossless and the reference really does have zero error. Values
    like `linspace(1, 2)` would not do -- the cast alone costs up to 0.5 ULP,
    the denominator is nonzero, and this stops testing what it claims to.
    """
    golden = torch.arange(1, 65, dtype=torch.float64)
    same = golden.to(torch.float32)
    assert torch.equal(same.to(torch.float64), golden), "cast was not lossless"

    b = compare_error_budget(same, same, golden)
    assert b.passed is True
    assert b.reference_p99_ulp == 0.0
    assert b.ratio_p99 is None
    assert b.model_dump(mode="json")["ratio_p99"] is None


def test_bit_exact_reference_still_rejects_a_bad_kernel():
    """The zero-denominator fallback must not become a free pass."""
    golden = torch.arange(1, 65, dtype=torch.float64)
    reference = golden.to(torch.float32)
    kernel = (golden + 0.5).to(torch.float32)  # wildly more than a few ULP

    b = compare_error_budget(kernel, reference, golden)
    assert b.ratio_p99 is None
    assert b.passed is False


def test_nan_in_the_kernel_output_fails_regardless_of_budget():
    golden = torch.ones(64, dtype=torch.float64)
    kernel = torch.ones(64, dtype=torch.float32)
    kernel[7] = float("nan")
    reference = torch.ones(64, dtype=torch.float32)

    b = compare_error_budget(kernel, reference, golden)
    assert b.passed is False


def test_shape_mismatch_is_a_failure_not_a_crash():
    golden = torch.zeros(8, dtype=torch.float64)
    b = compare_error_budget(
        torch.zeros(4, dtype=torch.float32),
        torch.zeros(8, dtype=torch.float32),
        golden,
    )
    assert b.passed is False
    assert b.shape_mismatch is True


def test_everything_serialises_json_clean():
    """Rule: an agent must be able to read this as JSON. inf and nan do not
    survive the round trip."""
    golden = torch.linspace(1.0, 2.0, 64, dtype=torch.float64)
    b = compare_error_budget(
        (golden + 1e-6).to(torch.float32), (golden + 1e-5).to(torch.float32), golden
    )
    dumped = b.model_dump(mode="json")
    restored = ErrorBudget.model_validate(dumped)
    assert restored == b
    for key, value in dumped.items():
        if isinstance(value, float):
            assert math.isfinite(value), f"{key} is not JSON-safe: {value}"


def test_handles_tensors_larger_than_quantiles_limit():
    """torch.quantile raises above ~16.7M elements (2**24).

    A 4097x4096 kernel output is 16.78M and appears in the standard FAST shape
    sweep, so this is an ordinary size, not a pathological one. Regression
    guard: the first implementation used torch.quantile and every large shape
    in the sweep errored out.
    """
    n = (1 << 24) + 1024
    golden = torch.ones(n, dtype=torch.float64)
    kernel = torch.ones(n, dtype=torch.float32)
    reference = torch.ones(n, dtype=torch.float32)

    b = compare_error_budget(kernel, reference, golden)
    assert b.total_elements == n
    assert b.passed is True


def test_reports_a_distribution_not_a_scalar():
    """Rule 5. A kernel right almost everywhere and wrong in one corner is the
    interesting failure, and a mean hides it."""
    golden = torch.ones(1000, dtype=torch.float64)
    kernel = torch.ones(1000, dtype=torch.float32)
    kernel[0] = 1.5  # one catastrophic element among a thousand good ones
    reference = torch.ones(1000, dtype=torch.float32)

    b = compare_error_budget(kernel, reference, golden)
    assert b.kernel_p50_ulp == 0.0
    assert b.kernel_max_ulp > b.kernel_p50_ulp
    assert b.total_elements == 1000
