"""The measurement loop. If this is wrong, nothing downstream matters."""

import math
from typing import Callable

from sns.clocks import ClockPolicy, UnlockedClockPolicy, assign_tier
from sns.env import throttle_snapshot
from sns.stats import bootstrap_ci, cv_percent, percentile, ratio_ci
from sns.telemetry import ClockSampler
from sns.types import ComparisonResult, TimingResult

MIN_ITERS = 30
DEFAULT_WARMUP = 200
MIN_DURATION_US = 10.0
MIN_FLUSH_BYTES = 256 * 1024 * 1024
# Historical cap from when clock sampling meant shelling out to nvidia-smi
# (tens of ms per call). NVML samples in-process at microsecond latency, so
# the cap is no longer needed in measure() — kept here only because
# clock_sample_stride is still a correct, cheap, separately-tested utility.
MAX_CLOCK_SAMPLES = 8


def l2_flush_buffer(device):
    """Scratch buffer large enough to evict L2 between iterations."""
    import torch

    props = torch.cuda.get_device_properties(device)
    l2_bytes = getattr(props, "L2_cache_size", 0) or 0
    nbytes = max(MIN_FLUSH_BYTES, l2_bytes * 2)
    return torch.empty(nbytes // 4, device=device, dtype=torch.float32)


def clock_sample_stride(iters: int, max_samples: int = MAX_CLOCK_SAMPLES) -> int:
    """Loop stride that keeps clock samples at or below max_samples.

    Ceiling division, not floor: floor lets the count exceed the cap for many
    iteration counts, including the default of 30, which is the whole point of
    having a cap.
    """
    if iters <= 0 or max_samples <= 0:
        return 1
    return max(1, math.ceil(iters / max_samples))


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

    Known limitation: the reported CI is a bootstrap over samples within a
    single measurement window, so it captures sampling error inside that
    window but not run-to-run variability between windows. On unlocked
    hardware the cross-run spread can exceed the reported interval by orders
    of magnitude — scripts/validate_timing.py measures exactly this. Folding
    between-window variance into Tier B intervals is planned for Phase 1.
    """
    import torch

    if iters < MIN_ITERS:
        raise ValueError(f"iters must be at least {MIN_ITERS}, got {iters}")
    if not torch.cuda.is_available():
        raise RuntimeError("measure() requires a CUDA device")

    torch.cuda.set_device(device)
    policy = policy or UnlockedClockPolicy()
    dev = torch.device(f"cuda:{device}")
    scratch = l2_flush_buffer(dev) if flush_l2 else None
    sampler = ClockSampler(device)

    policy.apply()
    # restore() runs in the finally below and clears policy.locked, so capture
    # the lock state now. Reading it after restore would make Tier A unreachable.
    was_locked = policy.locked
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

        throttle_before = throttle_snapshot(device)
        clock_samples: list[float] = []
        throttled_during = False

        # NVML runs in-process at microsecond latency, unlike the nvidia-smi
        # subprocess this used to shell out to, so the old bounded-sample cap
        # (clock_sample_stride) is unnecessary here — sample every iteration.
        for i in range(iters):
            if scratch is not None:
                scratch.zero_()
            starts[i].record()
            for _ in range(inner_reps):
                fn()
            ends[i].record()
            sm_clock = sampler.sample_clock_mhz()
            if sm_clock is not None:
                clock_samples.append(sm_clock)
            if sampler.throttled_now():
                throttled_during = True

        torch.cuda.synchronize()
        throttle_after = throttle_snapshot(device)
    finally:
        policy.restore()
        sampler.shutdown()

    samples = [
        starts[i].elapsed_time(ends[i]) / inner_reps for i in range(iters)
    ]
    throttle_fired = (
        throttle_before != throttle_after
        or any(v == "Active" for v in {**throttle_before, **throttle_after}.values())
        or bool(throttled_during)
    )
    tier = assign_tier(was_locked, clock_samples, throttle_fired)
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


def compare(
    candidate_fn: Callable[[], object],
    baseline_fn: Callable[[], object],
    **kwargs,
) -> ComparisonResult:
    """Measure a candidate against a freshly measured baseline.

    Both sides are timed in the same process, under the same clock policy,
    back to back. The baseline is never read from cache — a stale baseline
    makes it impossible to distinguish a kernel regression from an upstream
    improvement, which is the whole point of the tool.

    Known limitation: the candidate is measured first and the baseline second,
    so on unlocked hardware the second measurement runs on a warmer, more
    boosted GPU. Comparing identical work on an RTX 3060 laptop yielded a
    speedup of 0.962 rather than 1.0 from this effect alone. Tier A pinning
    largely removes it; Tier B and C comparisons carry it. Interleaving the
    two sides is planned for Phase 1.
    """
    candidate = measure(candidate_fn, **kwargs)
    baseline = measure(baseline_fn, **kwargs)

    if candidate.median_ms <= 0:
        raise ValueError(
            "candidate measured 0 ms: the timed region fell below CUDA event "
            "resolution. Raise iters, or check that the callable does real work."
        )

    lo, hi = ratio_ci(candidate.samples_ms, baseline.samples_ms)

    return ComparisonResult(
        candidate=candidate,
        baseline=baseline,
        speedup=baseline.median_ms / candidate.median_ms,
        speedup_ci_lo=lo,
        speedup_ci_hi=hi,
    )
