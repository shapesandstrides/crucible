"""The measurement loop. If this is wrong, nothing downstream matters."""

import math
from typing import Callable

from sns.clocks import ClockPolicy, UnlockedClockPolicy, assign_tier
from sns.env import smi_query_float, throttle_snapshot
from sns.stats import bootstrap_ci, cv_percent, percentile
from sns.types import MeasurementTier, TimingResult

MIN_ITERS = 30
DEFAULT_WARMUP = 200
MIN_DURATION_US = 10.0
MIN_FLUSH_BYTES = 256 * 1024 * 1024
# nvidia-smi is a subprocess call costing tens of ms; sampling it once per
# iteration would perturb the host duty cycle we are trying to observe.
MAX_CLOCK_SAMPLES = 8


def l2_flush_buffer(device):
    """Scratch buffer large enough to evict L2 between iterations."""
    import torch

    props = torch.cuda.get_device_properties(device)
    l2_bytes = getattr(props, "L2_cache_size", 0) or 0
    nbytes = max(MIN_FLUSH_BYTES, l2_bytes * 2)
    return torch.empty(nbytes // 4, device=device, dtype=torch.float32)


def resolve_inner_reps(single_iter_ms: float, min_duration_us: float) -> int:
    """How many times to run fn inside one timed region.

    Anything under ~10 us cannot be timed reliably: CUDA event overhead and
    system variance dominate. Loop until the window clears the floor.
    """
    if single_iter_ms <= 0:
        return 1000
    single_us = single_iter_ms * 1000.0
    if single_us >= min_duration_us:
        return 1
    return max(1, math.ceil(min_duration_us / single_us))


def measure(
    fn: Callable[[], object],
    *,
    warmup: int = DEFAULT_WARMUP,
    iters: int = MIN_ITERS,
    device: int = 0,
    flush_l2: bool = True,
    policy: ClockPolicy | None = None,
    min_duration_us: float = MIN_DURATION_US,
) -> TimingResult:
    """Time a callable honestly.

    Returns an interval and a quality tier, never a bare number.
    """
    import torch

    if iters < MIN_ITERS:
        raise ValueError(f"iters must be at least {MIN_ITERS}, got {iters}")
    if not torch.cuda.is_available():
        raise RuntimeError("measure() requires a CUDA device")

    policy = policy or UnlockedClockPolicy()
    dev = torch.device(f"cuda:{device}")
    scratch = l2_flush_buffer(dev) if flush_l2 else None

    policy.apply()
    try:
        # Warmup. The default of 200 exists because do_bench's default of 25
        # yields two calls and underestimates by ~30% (triton#2306).
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        # Calibrate the inner loop against one timed iteration.
        probe_start = torch.cuda.Event(enable_timing=True)
        probe_end = torch.cuda.Event(enable_timing=True)
        probe_start.record()
        fn()
        probe_end.record()
        torch.cuda.synchronize()
        inner_reps = resolve_inner_reps(
            probe_start.elapsed_time(probe_end), min_duration_us
        )

        # Events are allocated up front so allocation never lands inside a
        # timed region.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        throttle_before = throttle_snapshot()
        clock_samples: list[float] = []

        # Sampling the clock costs an nvidia-smi subprocess (tens of ms), so take a
        # bounded number of evenly-spaced samples rather than one per iteration.
        # Sampling every iteration would perturb the duty cycle we are measuring.
        clock_sample_every = max(1, iters // MAX_CLOCK_SAMPLES)

        for i in range(iters):
            if scratch is not None:
                scratch.zero_()
            starts[i].record()
            for _ in range(inner_reps):
                fn()
            ends[i].record()
            if i % clock_sample_every == 0:
                sm_clock = smi_query_float("clocks.sm", device)
                if sm_clock is not None:
                    clock_samples.append(sm_clock)

        torch.cuda.synchronize()
        throttle_after = throttle_snapshot()
    finally:
        policy.restore()

    samples = [
        starts[i].elapsed_time(ends[i]) / inner_reps for i in range(iters)
    ]
    throttle_fired = throttle_before != throttle_after
    tier = assign_tier(policy.locked, clock_samples, throttle_fired)
    ci_lo, ci_hi = bootstrap_ci(samples)

    return TimingResult(
        samples_ms=samples,
        median_ms=percentile(samples, 0.5),
        p10_ms=percentile(samples, 0.10),
        p90_ms=percentile(samples, 0.90),
        ci95_lo_ms=ci_lo,
        ci95_hi_ms=ci_hi,
        n=len(samples),
        tier=tier,
        warmup=warmup,
        inner_reps=inner_reps,
        throttle_fired=throttle_fired,
        clock_cv_pct=cv_percent(clock_samples) if clock_samples else None,
        clock_range_mhz=(
            max(clock_samples) - min(clock_samples) if clock_samples else None
        ),
    )
