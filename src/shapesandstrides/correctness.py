"""The correctness runner.

Walks a shape space, builds deterministic inputs, adjudicates each output
against the fp64 oracle, and shrinks any failures to the smallest case that
still reproduces.
"""

from typing import Callable

from pydantic import BaseModel

from shapesandstrides.env import is_cuda_device
from shapesandstrides.reference import (
    OracleKind,
    ReferenceResolutionError,
    ResolvedReference,
    resolve,
)
from shapesandstrides.oracle import (
    OracleResult,
    compare_against_oracle,
    compare_outputs,
    make_inputs,
    reference_fp64,
)
from shapesandstrides.shapes import ShapeSpec, ShapeTier, generate_shapes
from shapesandstrides.tolerance import tolerance_for

DEFAULT_SEED = 0xC0FFEE


class ShapeOutcome(BaseModel):
    spec: ShapeSpec
    passed: bool
    seed: int
    oracle: OracleResult | None = None
    # One entry per kernel output. `oracle` mirrors outputs[0] for
    # convenience; multi-output kernels put the rest here.
    outputs: list[OracleResult] = []
    error: str | None = None


class CorrectnessReport(BaseModel):
    # What kind of answer key produced this verdict. Recorded rather than
    # inferred, so a weak oracle can never be mistaken for a strong one.
    oracle_kind: OracleKind = OracleKind.NONE
    oracle_label: str = "none"

    outcomes: list[ShapeOutcome] = []
    passed: bool = True
    total: int = 0
    failed_count: int = 0
    minimal_failure: ShapeOutcome | None = None
    replay_command: str = ""


def shrink_to_minimal(failures: list[ShapeOutcome]) -> ShapeOutcome | None:
    """The smallest failing case is the one a human can actually debug.

    We do not search for a smaller shape than the ones tested; we pick the
    smallest that already failed. That keeps shrinking free and keeps every
    reported case one we genuinely observed. The label is a secondary sort
    key purely to make ties deterministic regardless of input order; for two
    shapes with equal element counts there is no principled "more minimal"
    one, so the tiebreak need not be meaningful, only stable.
    """
    if not failures:
        return None
    return min(failures, key=lambda o: (o.spec.numel(), o.spec.label))


def check(
    fn: Callable,
    reference: object,
    tier: ShapeTier = ShapeTier.FAST,
    dtypes: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    n_inputs: int | None = None,
    op_name: str = "unknown",
    device: str = "cuda",
    max_elements: int | None = None,
    fused_ops: list[str] | None = None,
    tolerance_override: tuple[float, float] | None = None,
    tiles=None,
) -> CorrectnessReport:
    """Run a kernel across a shape space and adjudicate every output.

    ``reference`` is anything `reference.resolve` accepts: a dotted path such
    as ``"torch.add"``, a lambda holding a short torch expression, any callable
    the caller already has, or a `ResolvedReference`. It is run in float64 on
    CPU, so it needs to be neither fast nor numerically careful — only correct.

    ``n_inputs`` defaults to the reference's own arity. Asking the caller how
    many tensors their kernel takes is a question we can usually answer
    ourselves, and a wrong answer surfaces as a confusing shape error rather
    than a clear one.

    ``tiles``, if given, is a `shapesandstrides.tiles.TileSpace` describing the kernel's
    declared block sizes; it is forwarded to `generate_shapes` so the shape
    space straddles those blocks' tails, not just generic power-of-two ones.
    Auto-discovery from ``fn`` is out of scope here: `discover_tiles` needs
    the inner `@triton.jit` object, not the Python wrapper callers pass to
    `check()`, so a caller who wants tile-aware generation must call
    `discover_tiles` itself and pass the result in explicitly.
    """
    import torch

    if is_cuda_device(device) and not torch.cuda.is_available():
        raise RuntimeError(
            f"check() requested device={device!r} but no CUDA GPUs are "
            "available on this machine. This is an environment problem, "
            "not a kernel defect — a correct kernel would fail every shape "
            "here for the same reason a broken one would."
        )

    ref = resolve(reference)
    if not ref.available:
        # Deliberately not a silent pass. A kernel with no reference is the
        # interesting case, but it needs the metamorphic checks to say anything
        # honest about it, and those are not wired in here yet.
        raise ReferenceResolutionError(
            "check() was given no reference. Pass a dotted path such as "
            "'torch.add', a lambda, or a callable. Verification without any "
            "reference (config agreement, stride and dtype invariance) is not "
            "available from check() yet."
        )

    # The reference and the kernel compute the same function, so the
    # reference's arity is the kernel's input count.
    if n_inputs is None:
        n_inputs = ref.arity or 2

    dtypes = dtypes or ["float16", "float32"]
    specs = generate_shapes(tier, dtypes=dtypes, max_elements=max_elements, tiles=tiles)
    outcomes: list[ShapeOutcome] = []

    for i, spec in enumerate(specs):
        # Derive a per-shape seed from the run seed so each shape gets
        # distinct inputs while the whole run stays reproducible.
        shape_seed = seed + i
        try:
            # Build inputs twice, once per device, rather than building once
            # and calling .to(device). Moving a non-contiguous tensor to
            # CUDA silently re-contiguifies it, which would make the whole
            # non-contiguous shape class vacuous. Same seed on both calls
            # gives value-identical tensors, so the oracle and the kernel
            # see the same numbers.
            cpu_inputs = make_inputs(spec, seed=shape_seed, n_inputs=n_inputs, device="cpu")
            dev_inputs = make_inputs(spec, seed=shape_seed, n_inputs=n_inputs, device=device)
            actual = fn(*dev_inputs)
            expected = reference_fp64(
                ref.fn, cpu_inputs, out_dtype=getattr(torch, spec.dtype)
            )
            atol, rtol = tolerance_for(
                op_name, spec.dtype, fused_ops=fused_ops, override=tolerance_override
            )
            results = compare_outputs(actual, expected, atol=atol, rtol=rtol)
            outcomes.append(
                ShapeOutcome(
                    spec=spec,
                    passed=all(r.passed for r in results),
                    seed=shape_seed,
                    oracle=results[0],
                    outputs=results,
                )
            )
        except Exception as e:
            outcomes.append(
                ShapeOutcome(
                    spec=spec,
                    passed=False,
                    seed=shape_seed,
                    error=f"{type(e).__name__}: {e}",
                )
            )

    failures = [o for o in outcomes if not o.passed]
    minimal = shrink_to_minimal(failures)
    replay = ""
    if minimal is not None:
        replay = f"shapesandstrides replay --shape {minimal.spec.label} --seed {minimal.seed}"

    return CorrectnessReport(
        oracle_kind=ref.kind,
        oracle_label=ref.label,
        outcomes=outcomes,
        passed=not failures,
        total=len(outcomes),
        failed_count=len(failures),
        minimal_failure=minimal,
        replay_command=replay,
    )
