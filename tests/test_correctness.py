import pytest

torch = pytest.importorskip("torch")

from sns.correctness import CorrectnessReport, ShapeOutcome, check, shrink_to_minimal
from sns.shapes import ShapeSpec, ShapeTier

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _outcome(dims, passed=False):
    return ShapeOutcome(
        spec=ShapeSpec(dims=dims, dtype="float16", layout="contiguous",
                       label="x".join(map(str, dims))),
        passed=passed,
        seed=1,
    )


def test_shrink_picks_the_smallest_failing_shape():
    failures = [_outcome((4097, 512)), _outcome((17, 3)), _outcome((1024, 1024))]
    m = shrink_to_minimal(failures)
    assert m.spec.dims == (17, 3), "the minimal case is the one with fewest elements"


def test_shrink_returns_none_when_nothing_failed():
    assert shrink_to_minimal([]) is None


def test_shrink_is_deterministic_for_equal_sizes():
    """With equal element counts there is no principled winner, so the
    tiebreak exists only to make the choice stable. Assert stability, not a
    particular shape."""
    a, b = _outcome((4, 4)), _outcome((16, 1))

    first = shrink_to_minimal([a, b])
    second = shrink_to_minimal([b, a])

    assert first.spec.dims == second.spec.dims, "order of input must not change the result"
    assert first.spec.dims in {(4, 4), (16, 1)}


@requires_gpu
def test_a_correct_kernel_passes_clean():
    r = check(
        lambda a, b: a + b,
        reference=lambda a, b: a + b,
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        op_name="add",
    )
    assert r.passed
    assert r.failed_count == 0
    assert r.total > 0
    assert r.minimal_failure is None


@requires_gpu
def test_a_wrong_kernel_fails_with_a_minimal_case_and_a_seed():
    def broken(a, b):
        out = a + b
        out = out.clone()
        out.view(-1)[-1] = 999.0  # corrupt the tail, like a missing mask
        return out

    r = check(
        broken,
        reference=lambda a, b: a + b,
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        op_name="add",
    )
    assert not r.passed
    assert r.failed_count > 0
    assert r.minimal_failure is not None
    assert r.minimal_failure.seed is not None
    assert "sns replay" in r.replay_command


@requires_gpu
def test_a_failure_replays_deterministically_from_its_seed():
    """The stored seed must reproduce the same verdict, or it is not evidence."""

    def broken(a, b):
        out = (a + b).clone()
        out.view(-1)[0] = -12345.0
        return out

    first = check(broken, reference=lambda a, b: a + b, tier=ShapeTier.FAST,
                  dtypes=["float32"], op_name="add", seed=99)
    second = check(broken, reference=lambda a, b: a + b, tier=ShapeTier.FAST,
                   dtypes=["float32"], op_name="add", seed=99)

    assert first.minimal_failure.spec.label == second.minimal_failure.spec.label
    assert first.minimal_failure.seed == second.minimal_failure.seed


@requires_gpu
def test_a_kernel_that_raises_is_recorded_not_propagated():
    def explodes(a, b):
        raise RuntimeError("illegal memory access")

    r = check(explodes, reference=lambda a, b: a + b, tier=ShapeTier.FAST,
              dtypes=["float32"], op_name="add")
    assert not r.passed
    assert any(o.error for o in r.outcomes)


@requires_gpu
def test_dtype_tolerance_is_applied_per_dtype():
    """fp16 must not be failed for ordinary half-precision rounding."""
    r = check(lambda a, b: a + b, reference=lambda a, b: a + b,
              tier=ShapeTier.FAST, dtypes=["float16"], op_name="add")
    assert r.passed
