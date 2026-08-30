import pytest

torch = pytest.importorskip("torch")

from pydantic import ValidationError

from shapesandstrides.correctness import (
    CorrectnessReport,
    ShapeOutcome,
    check,
    shrink_to_minimal,
)
from shapesandstrides.reference import OracleKind
from shapesandstrides.types import CheckKind, OracleTier
from shapesandstrides.shapes import ShapeSpec, ShapeTier

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


def test_a_missing_cuda_device_fails_clearly_not_as_a_kernel_defect(monkeypatch):
    """On a host with no GPU, requesting device="cuda" (the default) used to
    have every shape fail with 'No CUDA GPUs are available' rendered as
    INCORRECT — an environment problem misreported as a kernel bug. Fail
    once, upfront, with a message that names the environment."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        check(
            lambda a, b: a + b,
            reference=lambda a, b: a + b,
            tier=ShapeTier.FAST,
            dtypes=["float32"],
            op_name="add",
        )


def test_tiles_argument_reaches_generate_shapes_through_check():
    """check() never accepted or forwarded tiles, so declared block sizes
    could not change the shape space reaching a kernel through the public
    entry point, even though generate_shapes itself already supported it.
    Runs on CPU: no GPU needed to prove the argument gets threaded through."""
    from shapesandstrides.tiles import TileSpace

    def spy_factory(collector):
        def fn(a, b):
            collector.append(tuple(a.shape))
            return a + b
        return fn

    seen_without_tiles: list = []
    check(
        spy_factory(seen_without_tiles),
        reference=lambda a, b: a + b,
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        op_name="add",
        device="cpu",
    )

    seen_with_tiles: list = []
    ts = TileSpace(names=["BLOCK_M"], candidates={"BLOCK_M": [96]}, source="declared")
    check(
        spy_factory(seen_with_tiles),
        reference=lambda a, b: a + b,
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        op_name="add",
        device="cpu",
        tiles=ts,
    )

    assert set(seen_without_tiles) != set(seen_with_tiles), (
        "tiles= must change the shape set that reaches the kernel"
    )
    assert any(s[0] % 96 == 1 for s in seen_with_tiles), (
        "must include a shape straddling the declared block's tail"
    )


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
    assert r.replay_hint == f"shape={r.minimal_failure.spec.label} seed={r.minimal_failure.seed}"
    assert "replay" not in r.replay_hint


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


@requires_gpu
def test_kernels_actually_receive_noncontiguous_inputs():
    """The stride bug has appeared twice: once in make_inputs and once at the
    device transfer. Both times the non-contiguous shape class silently proved
    nothing. Assert the property itself, not a downstream symptom."""
    contiguity_seen = []

    def spy(a, b):
        contiguity_seen.append(a.is_contiguous())
        return a + b

    check(
        spy,
        reference=lambda a, b: a + b,
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        op_name="add",
    )

    assert contiguity_seen, "the spy was never called"
    assert False in contiguity_seen, (
        "no non-contiguous input ever reached the kernel — the shape class "
        "is being silently re-contiguified somewhere"
    )


@requires_gpu
def test_a_multi_output_kernel_passes_through_check():
    """reference_fp64 used to call .to() on the tuple itself, raising
    AttributeError before compare_outputs' multi-output path ever ran.
    Fused kernels are the stated target audience, and they are exactly
    the ones that return several tensors."""
    r = check(
        lambda a, b: (a + b, a * b),
        reference=lambda a, b: (a + b, a * b),
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        op_name="add",
    )
    assert r.passed, f"multi-output kernel failed: {r.outcomes[0].error}"


@requires_gpu
def test_a_bug_in_a_secondary_output_is_caught():
    """A wrong second tensor — a saved mean or rstd — must fail the check."""

    def broken(a, b):
        return (a + b, torch.full_like(a, 99.0))

    r = check(
        broken,
        reference=lambda a, b: (a + b, a * b),
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        op_name="add",
    )
    assert not r.passed


@requires_gpu
def test_a_kernel_that_returns_garbage_fails_every_shape():
    """Pins the failure path independently of any specific bug."""
    r = check(
        lambda a, b: torch.full_like(a, 12345.0),
        reference=lambda a, b: a + b,
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        op_name="add",
    )
    assert not r.passed
    assert r.failed_count == r.total, "every shape should fail for a constant-garbage kernel"


# -------------------------------------------------- the oracle tier on a report
#
# A verdict that does not say what adjudicated it is the failure mode this
# whole field exists to prevent: three very different claims ("matches
# PyTorch", "matches your own prototype", "did not contradict itself") all
# rendering as an identical PASS.


def _passing_report(**overrides):
    base = dict(
        oracle_kind=OracleKind.TORCH_OP,
        oracle_label="torch.add",
        oracle_tier=OracleTier.A,
        checks=[CheckKind.REFERENCE],
        outcomes=[],
        passed=True,
        total=0,
        failed_count=0,
    )
    base.update(overrides)
    return CorrectnessReport(**base)


def test_a_report_cannot_be_built_without_an_oracle_tier():
    """Structural enforcement, in the spirit of TimingResult having no
    __float__: forgetting the tier must be impossible, not merely discouraged."""
    with pytest.raises(ValidationError):
        CorrectnessReport(passed=True, total=1, failed_count=0)


def test_the_oracle_fields_have_no_defaults():
    """A default is exactly the hole this closes -- with one, an untiered
    report silently claims the value the author happened to pick."""
    for field in ("oracle_kind", "oracle_label", "oracle_tier"):
        assert CorrectnessReport.model_fields[field].is_required(), (
            f"{field} has a default, so a report can omit it and still "
            f"present itself as a verdict"
        )


def test_a_tier_c_pass_is_not_a_valid_correctness_verdict():
    r = _passing_report(
        oracle_kind=OracleKind.NONE, oracle_label="none",
        oracle_tier=OracleTier.C, checks=[],
    )
    assert r.passed is True
    assert r.is_correctness_valid is False, (
        "nothing contradicted itself is not the same claim as correct"
    )


def test_tier_a_and_b_are_valid_correctness_verdicts():
    assert _passing_report(oracle_tier=OracleTier.A).is_correctness_valid is True
    assert _passing_report(oracle_tier=OracleTier.B).is_correctness_valid is True


def test_the_tier_survives_a_json_round_trip():
    """An agent reads this out of --json, so it has to be there as a string."""
    dumped = _passing_report().model_dump(mode="json")
    assert dumped["oracle_tier"] == "A"
    assert dumped["checks"] == ["reference"]


@requires_gpu
def test_check_records_the_tier_earned_by_its_reference():
    r = check(torch.add, "torch.add", tier=ShapeTier.FAST, dtypes=["float32"],
              max_elements=4096)
    assert r.oracle_tier is OracleTier.A
    assert r.checks == [CheckKind.REFERENCE]


def test_check_records_a_user_callable_as_the_weaker_tier(monkeypatch):
    def my_prototype(x, y):
        return x + y

    r = check(my_prototype, my_prototype, tier=ShapeTier.FAST,
              dtypes=["float32"], max_elements=1024, device="cpu")
    assert r.oracle_tier is OracleTier.B, (
        "agreeing with the caller's own code is weaker evidence than agreeing "
        "with PyTorch, and the report must say so"
    )
