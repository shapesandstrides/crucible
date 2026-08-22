import pytest

torch = pytest.importorskip("torch")

from sns.oracle import (
    OracleResult,
    compare_against_oracle,
    compare_outputs,
    make_inputs,
    reference_fp64,
)
from sns.shapes import ShapeSpec

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _spec(dims=(64, 64), dtype="float16", layout="contiguous"):
    return ShapeSpec(dims=dims, dtype=dtype, layout=layout, label="t")


def test_make_inputs_is_reproducible_from_a_seed():
    a = make_inputs(_spec(), seed=1234, n_inputs=2)
    b = make_inputs(_spec(), seed=1234, n_inputs=2)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_different_seeds_give_different_inputs():
    a = make_inputs(_spec(), seed=1, n_inputs=1)
    b = make_inputs(_spec(), seed=2, n_inputs=1)
    assert not torch.equal(a[0], b[0])


def test_make_inputs_honours_shape_and_dtype():
    x = make_inputs(_spec(dims=(8, 3), dtype="float32"), seed=0, n_inputs=1)[0]
    assert tuple(x.shape) == (8, 3)
    assert x.dtype is torch.float32


def test_noncontiguous_layout_produces_a_noncontiguous_tensor():
    x = make_inputs(_spec(layout="noncontiguous"), seed=0, n_inputs=1)[0]
    assert not x.is_contiguous()


@requires_gpu
def test_noncontiguous_survives_the_device_transfer():
    """Moving to the GPU after slicing silently re-contiguifies the tensor,
    which would make the entire non-contiguous shape class vacuous."""
    for dtype in ("float32", "float16", "bfloat16"):
        spec = _spec(dims=(256, 256), dtype=dtype, layout="noncontiguous")
        x = make_inputs(spec, seed=0, n_inputs=1, device="cuda")[0]
        assert not x.is_contiguous(), f"{dtype} lost its strides on the GPU"
        assert x.device.type == "cuda"


@requires_gpu
def test_cpu_and_gpu_inputs_hold_identical_values():
    """The oracle runs on CPU and the kernel on GPU; if the two disagree the
    comparison is meaningless."""
    spec = _spec(dims=(64, 64), dtype="float32", layout="noncontiguous")
    cpu = make_inputs(spec, seed=7, n_inputs=2, device="cpu")
    gpu = make_inputs(spec, seed=7, n_inputs=2, device="cuda")
    for c, g in zip(cpu, gpu):
        assert torch.equal(c, g.cpu())


def test_reference_is_computed_in_fp64_then_cast_down():
    """The oracle must not inherit the target dtype's rounding."""
    spec = _spec(dims=(512,), dtype="float16")
    inputs = make_inputs(spec, seed=7, n_inputs=2)

    ref = reference_fp64(lambda a, b: a + b, inputs, out_dtype=torch.float16)

    assert ref.dtype is torch.float16
    fp64 = inputs[0].double() + inputs[1].double()
    assert torch.equal(ref, fp64.to(torch.float16))


def test_reference_computes_in_fp64_on_cpu():
    """Asserting on output values cannot detect a missing upcast: a single
    IEEE add is exactly rounded, and torch already accumulates fp16 sums in
    fp32. Assert what the reference function is actually handed."""
    spec = _spec(dims=(128,), dtype="float16")
    inputs = make_inputs(spec, seed=3, n_inputs=2)
    seen = {}

    def spy(a, b):
        seen["dtype"] = a.dtype
        seen["device"] = a.device.type
        return a + b

    out = reference_fp64(spy, inputs, out_dtype=torch.float16)

    assert seen["dtype"] is torch.float64, "the oracle must compute in double precision"
    assert seen["device"] == "cpu", "the oracle must not run on the GPU under test"
    assert out.dtype is torch.float16


def test_oracle_accepts_a_result_within_tolerance():
    expected = torch.ones(100, dtype=torch.float16)
    actual = expected + 1e-4
    r = compare_against_oracle(actual, expected, atol=1e-3, rtol=1e-3)
    assert r.passed
    assert r.max_abs_error < 1e-3


def test_oracle_rejects_a_result_outside_tolerance():
    expected = torch.ones(100, dtype=torch.float16)
    actual = expected.clone()
    actual[42] = 5.0
    r = compare_against_oracle(actual, expected, atol=1e-3, rtol=1e-3)
    assert not r.passed
    assert r.max_abs_error > 1.0
    assert r.mismatch_count == 1
    assert r.first_mismatch_index == 42


def test_oracle_reports_nan_distinctly_from_a_numeric_mismatch():
    """NaN in an output is a different failure mode from being off by 0.01."""
    expected = torch.ones(10, dtype=torch.float32)
    actual = expected.clone()
    actual[3] = float("nan")
    r = compare_against_oracle(actual, expected, atol=1e-5, rtol=1e-5)
    assert not r.passed
    assert r.has_nan


def test_oracle_reports_shape_mismatch_without_crashing():
    r = compare_against_oracle(
        torch.ones(10), torch.ones(11), atol=1e-5, rtol=1e-5
    )
    assert not r.passed
    assert r.shape_mismatch


def test_multi_output_checks_every_tensor_not_just_the_first():
    """Fused kernels return several tensors. A bug in a secondary output
    — a saved mean or rstd for the backward pass — is exactly the kind
    that survives to production if only the first is inspected."""
    good = torch.ones(10, dtype=torch.float32)
    bad = good.clone()
    bad[0] = 99.0

    results = compare_outputs((good, bad), (good, good), atol=1e-5, rtol=1e-5)

    assert len(results) == 2
    assert results[0].passed
    assert not results[1].passed, "a wrong second output must fail the check"


def test_multi_output_accepts_a_single_tensor():
    r = compare_outputs(torch.ones(4), torch.ones(4), atol=1e-5, rtol=1e-5)
    assert len(r) == 1 and r[0].passed


def test_multi_output_flags_an_arity_mismatch():
    r = compare_outputs((torch.ones(4),), (torch.ones(4), torch.ones(4)),
                        atol=1e-5, rtol=1e-5)
    assert not all(x.passed for x in r)
