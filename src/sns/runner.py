"""The `test()` entry point: correctness, then timing, then a record."""

import hashlib
import inspect
import time
from pathlib import Path
from typing import Callable

from sns.correctness import check
from sns.env import is_cuda_device
from sns.metrics import (
    collect_device_info,
    collect_memory_metrics,
    collect_runtime_context,
    trace_dispatch,
)
from sns.oracle import make_inputs
from sns.records import RunRecord, new_run_id, save_run
from sns.shapes import ShapeTier, generate_shapes
from sns.timing import compare


def _hash_callable(fn: Callable) -> str:
    """Hash a callable so the same kernel is recognisable across runs.

    Source is preferred. The fallback must never be repr() for a plain
    function: that embeds a memory address, so the same kernel would get a
    fresh identity every process and a catalog would never match it to its
    own history.

    Known limitation: two functools.partial wrappers of the *same* function
    with different bound arguments hash identically, since only the
    underlying function's source/qualname is used. Left unsolved rather than
    hashing bound arguments, which are often tensors whose reprs are
    themselves unstable across processes.
    """
    target = getattr(fn, "func", fn)  # unwrap functools.partial
    try:
        src = inspect.getsource(target)
    except (OSError, TypeError):
        module = getattr(target, "__module__", "?")
        qualname = getattr(target, "__qualname__", None)
        if qualname is None:
            # Last resort: a type name is stable where an instance repr is not.
            qualname = type(target).__name__
        src = f"{module}.{qualname}"
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _device_index(device: str) -> int:
    """Extract the CUDA device index from a torch-style device string.

    compare()/measure() take an int index (they call torch.cuda.set_device
    and build "cuda:{index}" themselves), while make_inputs/check take the
    string form. "cuda" alone means index 0, matching torch's own default.
    """
    if ":" in device:
        suffix = device.rsplit(":", 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            raise ValueError(
                f"could not parse a device index from {device!r}"
            ) from None
    return 0


def run_test(
    kernel: Callable,
    reference: Callable,
    kernel_name: str,
    op_name: str = "unknown",
    tier: ShapeTier = ShapeTier.FAST,
    dtypes: list[str] | None = None,
    seed: int = 0xC0FFEE,
    n_inputs: int = 2,
    device: str = "cuda",
    time_it: bool = True,
    warmup: int = 200,
    iters: int = 30,
    variant: str | None = None,
    tags: list[str] | None = None,
    root: Path | None = None,
    max_elements: int | None = None,
    tiles=None,
) -> RunRecord:
    """Test a kernel and persist a run record.

    Correctness runs first across the shape space. Timing runs afterwards
    on the canonical tier only, and **only if correctness passed** — timing
    a kernel already known to be wrong produces a number with no meaning.

    The `reference` callable serves two different roles, deliberately: inside
    `check()` it runs on CPU in float64 as the correctness oracle (never the
    GPU op, which would share numerics and bugs with the thing under test),
    and here, when timing, it runs on GPU as the speed baseline, because that
    is what the user is actually trying to beat.
    """
    import torch

    started = time.time()
    dtypes = dtypes or ["float16", "float32"]

    correctness = check(
        kernel,
        reference=reference,
        tier=tier,
        dtypes=dtypes,
        seed=seed,
        n_inputs=n_inputs,
        op_name=op_name,
        device=device,
        max_elements=max_elements,
        tiles=tiles,
    )

    comparison = None
    dispatch = trace_dispatch(lambda: None)
    timing_note = None
    # A completed correctness sweep must never be discarded because timing
    # failed or was inapplicable. compare() is CUDA-only, so an environment
    # without CUDA (or a caller who explicitly asked for device="cpu") skips
    # timing rather than raising, and any other timing failure is recorded
    # as a note rather than propagated past save_run below.
    if time_it and correctness.passed:
        if not is_cuda_device(device) or not torch.cuda.is_available():
            timing_note = (
                f"timing skipped: compare() requires a CUDA device "
                f"(requested device={device!r}, cuda available="
                f"{torch.cuda.is_available()})"
            )
        else:
            try:
                canonical = generate_shapes(
                    ShapeTier.CANONICAL, dtypes=[dtypes[0]], max_elements=max_elements
                )
                if canonical:
                    spec = canonical[-1]
                    # Build inputs directly on the target device. Building
                    # on CPU and calling .to(device) silently
                    # re-contiguifies non-contiguous tensors, which would
                    # void that shape class entirely.
                    inputs = make_inputs(
                        spec, seed=seed, n_inputs=n_inputs, device=device
                    )
                    comparison = compare(
                        lambda: kernel(*inputs),
                        lambda: reference(*inputs),
                        warmup=warmup,
                        iters=iters,
                        device=_device_index(device),
                    )
                    dispatch = trace_dispatch(lambda: reference(*inputs))
            except Exception as e:
                timing_note = f"timing failed: {type(e).__name__}: {e}"

    record = RunRecord(
        run_id=new_run_id(),
        kernel_name=kernel_name,
        kernel_hash=_hash_callable(kernel),
        variant=variant,
        tags=tags or [],
        device=collect_device_info(),
        memory=collect_memory_metrics(),
        context=collect_runtime_context(),
        dispatch=dispatch,
        correctness=correctness,
        timing=comparison.candidate if comparison else None,
        comparison=comparison,
        duration_s=time.time() - started,
        notes=timing_note,
    )
    save_run(record, root=root)
    return record
