"""check() with no hand-written reference.

These are the tests that matter for adoption. If a user has to author a
reference implementation before anything runs, the tool cannot sit in someone
else's CI, and it certainly cannot gate a registry with hundreds of kernels.
So: PyTorch's own operator is the answer key, and the input count is derived
rather than asked for.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from shapesandstrides.correctness import check  # noqa: E402
from shapesandstrides.reference import OracleKind, ReferenceResolutionError  # noqa: E402
from shapesandstrides.shapes import ShapeTier  # noqa: E402

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


# ------------------------------------------------------ no GPU needed here
# Arity derivation is the DX claim, and it is pure logic. Prove it on CPU so
# it stays covered on a CPU-only CI runner.


def test_unary_op_needs_no_n_inputs():
    r = check(
        lambda x: torch.relu(x),
        reference="torch.relu",
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        device="cpu",
        op_name="relu",
    )
    assert r.passed, r.minimal_failure
    assert r.total > 0
    # The caller never said "this takes one tensor".
    assert r.oracle_kind is OracleKind.TORCH_OP
    assert r.oracle_label == "torch.relu"


def test_binary_op_needs_no_n_inputs():
    r = check(
        lambda a, b: a + b,
        reference="torch.add",
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        device="cpu",
        op_name="add",
    )
    assert r.passed, r.minimal_failure
    assert r.oracle_kind is OracleKind.TORCH_OP


def test_a_lambda_expression_is_recorded_as_an_expression():
    r = check(
        lambda a, b: a + b,
        reference=lambda a, b: a + b,
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        device="cpu",
        op_name="add",
    )
    assert r.passed
    assert r.oracle_kind is OracleKind.EXPRESSION
    assert r.oracle_label == "<expression>"


def test_wrong_arity_is_caught_rather_than_silently_passing():
    # A unary kernel checked against a binary op must not quietly succeed.
    r = check(
        lambda x: torch.relu(x),
        reference="torch.add",
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        device="cpu",
        op_name="add",
    )
    assert not r.passed


def test_no_reference_says_so_clearly():
    with pytest.raises(ReferenceResolutionError) as e:
        check(lambda a, b: a + b, reference=None, device="cpu")
    msg = str(e.value)
    assert "no reference" in msg.lower()
    # Must name the alternatives rather than just refusing.
    assert "torch.add" in msg


def test_a_bad_operator_name_fails_before_any_kernel_runs():
    ran = False

    def kernel(a, b):
        nonlocal ran
        ran = True
        return a + b

    with pytest.raises(ReferenceResolutionError):
        check(kernel, reference="torch.not_an_operator", device="cpu")
    assert not ran, "resolution must fail before we spend a single shape"


# ------------------------------------------------------------- real Triton


@requires_gpu
def test_correct_triton_kernel_passes_against_a_torch_operator():
    import kernels as K

    r = check(
        K.triton_add,
        reference="torch.add",
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        op_name="add",
    )
    assert r.passed, f"minimal failure: {r.minimal_failure}"
    assert r.total > 0
    assert r.oracle_kind is OracleKind.TORCH_OP


@requires_gpu
def test_broken_triton_kernel_fails_against_a_torch_operator():
    """The tail-dropping kernel must still be caught with no hand-written
    reference — this is the whole claim of the resolver."""
    import kernels as K

    r = check(
        K.triton_add_drops_tail,
        reference="torch.add",
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        op_name="add",
    )
    assert not r.passed
    assert r.failed_count > 0
    assert r.minimal_failure is not None
    assert r.replay_command.startswith("shapesandstrides replay")


@requires_gpu
def test_unary_triton_kernel_against_a_one_line_expression():
    """The fused/novel case: no single torch op, so a short expression stands
    in. Arity comes from the lambda, not from the caller."""
    import kernels as K

    r = check(
        K.triton_rowsum_bad_accum,
        reference=lambda x: x.sum(dim=-1),
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        op_name="sum",
    )
    # This kernel accumulates badly on purpose; the point is that a one-line
    # reference was enough to convict it.
    assert not r.passed
    assert r.oracle_kind is OracleKind.EXPRESSION
