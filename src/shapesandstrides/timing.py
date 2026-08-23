"""The measurement loop. If this is wrong, nothing downstream matters."""

import math
from typing import Callable

from shapesandstrides.clocks import ClockPolicy, UnlockedClockPolicy, assign_tier
from shapesandstrides.env import (
    hw_power_brake_active,
    hw_throttle_active,
    power_cap_active,
    sw_thermal_throttle_active,
    throttle_snapshot,
)
from shapesandstrides.stats import bootstrap_ci, cv_percent, percentile, quantization_step, ratio_ci
from shapesandstrides.telemetry import ClockSampler
from shapesandstrides.types import ComparisonResult, TimingResult

MIN_ITERS = 30
DEFAULT_WARMUP = 200
MIN_DURATION_US = 1000.0
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

    CUDA event resolution is ~1 us, so a window a few microseconds long has
    every sample land on one of a handful of quantized values; a bootstrap
    CI over such data can come out narrower than the timer's resolution,
    which is a false precision claim, not a tight measurement. The floor
    exists so quantization is small relative to the effect size we're
    trying to detect, not merely so the timer can register the event at
    all. Loop until the window clears the floor.
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

    min_duration_us sets the floor for the timed window (see
    resolve_inner_reps). It exists to make CUDA event quantization small
    relative to the effect size we claim to detect, not merely to clear the
    timer's own floor for registering an event at all: a window barely above
    the timer's resolution still ties heavily, and a bootstrap CI over tied
    data can land narrower than the timer can actually resolve. As a second
    line of defense, the CI is widened to at least one quantization step
    below (see quantization_step in shapesandstrides.stats).

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
    # Hardware-asserted throttling gates the tier; driver-reported software
    # flags (sw_power_cap, sw_thermal_slowdown) do not, because they were
    # measured Active on real laptop hardware at idle, 55C, 18W, so none of
    # them are trustworthy evidence. They are still recorded below, as
    # metadata.
    hw_throttled = hw_throttle_active(throttle_before, throttle_after, throttled_during)
    power_capped = power_cap_active(throttle_before) or power_cap_active(throttle_after)
    sw_thermal_flagged = (
        sw_thermal_throttle_active(throttle_before)
        or sw_thermal_throttle_active(throttle_after)
    )
    hw_power_brake_flagged = (
        hw_power_brake_active(throttle_before) or hw_power_brake_active(throttle_after)
    )
    tier = assign_tier(was_locked, clock_samples, hw_throttled)
    ci_lo, ci_hi = bootstrap_ci(samples)

    step = quantization_step(samples)
    if step is not None and (ci_hi - ci_lo) < step:
        # An interval narrower than the smallest difference the timer can
        # resolve claims precision the instrument does not have.
        mid = percentile(samples, 0.5)
        ci_lo, ci_hi = mid - step / 2, mid + step / 2

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
        throttle_fired=hw_throttled,
        power_capped=power_capped,
        sw_thermal_flagged=sw_thermal_flagged,
        hw_power_brake_flagged=hw_power_brake_flagged,
        clock_cv_pct=cv_percent(clock_samples) if clock_samples else None,
        clock_range_mhz=(
            max(clock_samples) - min(clock_samples) if clock_samples else None
        ),
        quantization_step_ms=step,
    )


def _interleaved_samples(
    candidate_fn: Callable[[], object],
    baseline_fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    device: int,
    flush_l2: bool,
    policy: ClockPolicy,
    min_duration_us: float,
):
    """Run the interleaved measurement loop and return raw ingredients.

    This is the GPU half of compare(); the guard/CI/ratio logic lives in
    _compare_impl, which takes the lists this returns and needs no GPU.

    Returns (candidate_samples, baseline_samples, inner_reps, tier,
    hw_throttled, power_capped, sw_thermal_flagged,
    hw_power_brake_flagged, clock_samples).
    """
    import torch

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
        # Warmup both sides, alternating, so neither callable is cold relative
        # to the other when timing starts.
        for _ in range(warmup):
            candidate_fn()
            baseline_fn()
        torch.cuda.synchronize()

        # Calibrate inner_reps once, from whichever side is slower, and use
        # the same value for both. Different rep counts would reintroduce a
        # systematic difference through launch-overhead amortisation.
        c_start = torch.cuda.Event(enable_timing=True)
        c_end = torch.cuda.Event(enable_timing=True)
        b_start = torch.cuda.Event(enable_timing=True)
        b_end = torch.cuda.Event(enable_timing=True)
        c_start.record()
        candidate_fn()
        c_end.record()
        b_start.record()
        baseline_fn()
        b_end.record()
        torch.cuda.synchronize()
        probe_ms = max(
            c_start.elapsed_time(c_end), b_start.elapsed_time(b_end)
        )
        inner_reps = resolve_inner_reps(probe_ms, min_duration_us)

        # Events are allocated up front so allocation never lands inside a
        # timed region.
        c_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        c_ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        b_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        b_ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        throttle_before = throttle_snapshot(device)
        clock_samples: list[float] = []
        throttled_during = False

        def _run_candidate(i: int) -> None:
            if scratch is not None:
                scratch.zero_()
            c_starts[i].record()
            for _ in range(inner_reps):
                candidate_fn()
            c_ends[i].record()

        def _run_baseline(i: int) -> None:
            if scratch is not None:
                scratch.zero_()
            b_starts[i].record()
            for _ in range(inner_reps):
                baseline_fn()
            b_ends[i].record()

        for i in range(iters):
            # Alternate which side goes first. A fixed order would hand any
            # second-position advantage — residual boost state, warmer
            # caches — entirely to one side. Flipping splits it evenly, so
            # it cancels in the ratio rather than accumulating in it. Each
            # side always records into its own event arrays regardless of
            # which runs first, so the candidate/baseline identity of a
            # sample never depends on physical run order.
            if i % 2 == 0:
                _run_candidate(i)
                _run_baseline(i)
            else:
                _run_baseline(i)
                _run_candidate(i)

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

    candidate_samples = [
        c_starts[i].elapsed_time(c_ends[i]) / inner_reps for i in range(iters)
    ]
    baseline_samples = [
        b_starts[i].elapsed_time(b_ends[i]) / inner_reps for i in range(iters)
    ]

    # Hardware-asserted throttling gates the tier; driver-reported software
    # flags (sw_power_cap, sw_thermal_slowdown) do not, because they were
    # measured Active on real laptop hardware at idle, 55C, 18W, so none of
    # them are trustworthy evidence. They are still recorded below, as
    # metadata.
    hw_throttled = hw_throttle_active(throttle_before, throttle_after, throttled_during)
    power_capped = power_cap_active(throttle_before) or power_cap_active(throttle_after)
    sw_thermal_flagged = (
        sw_thermal_throttle_active(throttle_before)
        or sw_thermal_throttle_active(throttle_after)
    )
    hw_power_brake_flagged = (
        hw_power_brake_active(throttle_before) or hw_power_brake_active(throttle_after)
    )
    # Both sides share one measurement window, so they share one tier.
    tier = assign_tier(was_locked, clock_samples, hw_throttled)

    return (
        candidate_samples,
        baseline_samples,
        inner_reps,
        tier,
        hw_throttled,
        power_capped,
        sw_thermal_flagged,
        hw_power_brake_flagged,
        clock_samples,
    )


def _compare_impl(
    candidate_samples: list[float],
    baseline_samples: list[float],
    *,
    tier,
    warmup: int,
    inner_reps: int,
    throttle_fired: bool = False,
    power_capped: bool = False,
    sw_thermal_flagged: bool = False,
    hw_power_brake_flagged: bool = False,
    clock_samples: list[float] | None = None,
) -> ComparisonResult:
    """Turn two sample lists sharing one window into a ComparisonResult.

    Pure arithmetic on plain lists: the zero-median guard, the CI floor, and
    the speedup ratio all live here so they can be exercised on CPU, with no
    GPU and no mocking. This is the product's central claim in code form —
    it needs to run in every CI environment, not just the ones with a GPU.
    """
    clock_samples = clock_samples or []

    if percentile(candidate_samples, 0.5) <= 0:
        raise ValueError(
            "candidate measured 0 ms: the timed region fell below CUDA event "
            "resolution. Raise iters, or check that the callable does real work."
        )

    def _to_timing_result(samples: list[float]) -> TimingResult:
        ci_lo, ci_hi = bootstrap_ci(samples)
        step = quantization_step(samples)
        if step is not None and (ci_hi - ci_lo) < step:
            # An interval narrower than the smallest difference the timer can
            # resolve claims precision the instrument does not have.
            mid = percentile(samples, 0.5)
            ci_lo, ci_hi = mid - step / 2, mid + step / 2
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
            power_capped=power_capped,
            sw_thermal_flagged=sw_thermal_flagged,
            hw_power_brake_flagged=hw_power_brake_flagged,
            clock_cv_pct=cv_percent(clock_samples) if clock_samples else None,
            clock_range_mhz=(
                max(clock_samples) - min(clock_samples) if clock_samples else None
            ),
            quantization_step_ms=step,
        )

    candidate = _to_timing_result(candidate_samples)
    baseline = _to_timing_result(baseline_samples)

    lo, hi = ratio_ci(candidate.samples_ms, baseline.samples_ms)

    return ComparisonResult(
        candidate=candidate,
        baseline=baseline,
        speedup=baseline.median_ms / candidate.median_ms,
        speedup_ci_lo=lo,
        speedup_ci_hi=hi,
    )


def compare(
    candidate_fn: Callable[[], object],
    baseline_fn: Callable[[], object],
    *,
    warmup: int = DEFAULT_WARMUP,
    iters: int = MIN_ITERS,
    device: int = 0,
    flush_l2: bool = True,
    policy: ClockPolicy | None = None,
    min_duration_us: float = MIN_DURATION_US,
) -> ComparisonResult:
    """Measure a candidate against a baseline, interleaved.

    The two callables alternate inside a single measurement window rather
    than running as two separate windows. GPU clocks drift, and a drift
    between two sequential windows lands entirely in the ratio: measured on
    an unlocked laptop, comparing identical work that way gave a p90 error
    of 106% and a worst case of 141%. Interleaved, the same comparison gave
    a p90 error of 1.0%.

    The baseline is still measured fresh on every call and never cached.
    Interleaving strengthens that rule rather than replacing it: the two
    sides now share not just a process but a thermal and clock state.

    This is a thin CUDA-gated shell: it collects the raw samples via
    _interleaved_samples and hands them to _compare_impl, which does the
    actual guard/CI/ratio logic and needs no GPU to test.
    """
    import torch

    if iters < MIN_ITERS:
        raise ValueError(f"iters must be at least {MIN_ITERS}, got {iters}")
    if not torch.cuda.is_available():
        raise RuntimeError("compare() requires a CUDA device")

    policy = policy or UnlockedClockPolicy()

    (
        candidate_samples,
        baseline_samples,
        inner_reps,
        tier,
        hw_throttled,
        power_capped,
        sw_thermal_flagged,
        hw_power_brake_flagged,
        clock_samples,
    ) = _interleaved_samples(
        candidate_fn,
        baseline_fn,
        warmup=warmup,
        iters=iters,
        device=device,
        flush_l2=flush_l2,
        policy=policy,
        min_duration_us=min_duration_us,
    )

    return _compare_impl(
        candidate_samples,
        baseline_samples,
        tier=tier,
        warmup=warmup,
        inner_reps=inner_reps,
        throttle_fired=hw_throttled,
        power_capped=power_capped,
        sw_thermal_flagged=sw_thermal_flagged,
        hw_power_brake_flagged=hw_power_brake_flagged,
        clock_samples=clock_samples,
    )
