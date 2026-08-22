# Measurement tiers

Every result carries a tier recording how much it can be trusted. Tiers are assigned from observed telemetry, never assumed, and never silently upgraded.

| Tier | Conditions | What it means |
|---|---|---|
| **A** | Clocks pinned, verified held under load, range ≤ 30 MHz | Fully trustworthy. Comparable across runs. |
| **B** | Clocks floating; variance measured and disclosed | Usable, but only for differences large enough to survive the noise. |
| **C** | Clock CV > 3%, or throttling detected | **No performance verdict is valid.** |

```python
t.tier                    # MeasurementTier.B
t.is_performance_valid    # False only for Tier C
```

## The problem tiers solve

GPU clocks are not constant. They boost, they drop under thermal load, and they get capped by power limits. Timing a kernel while the clock moves measures the clock, not the kernel.

The obvious fix is to pin the clocks and refuse to measure otherwise. That's correct for hardware you control and useless for a tool that runs on hardware other people control — most consumer GPUs simply will not permit it. A tool that refuses to run for most of its audience helps nobody.

So the principle survives in a weaker form: **never *silently* measure with floating clocks.** Measure the variance, classify it, and put the classification next to every number.

## Why throttle flags alone aren't enough

You might expect `nvidia-smi`'s throttle reasons to answer "was this measurement clean?" They don't, for two independent reasons — both found by testing on real hardware.

**They miss variance that isn't throttling.** On an RTX 3060 under sustained load the SM clock swung **495 MHz — 5.1% CV** — while every throttle flag stayed inactive. Two identical runs even disagreed about whether throttling fired at all. Boost and idle transitions move the clock hundreds of megahertz without setting a single flag.

**They miss throttling that never stops.** The original implementation compared a snapshot taken before the loop against one taken after, and treated a *difference* as evidence of throttling. A GPU throttled for the entire window produces `before == after == "Active"` — identical snapshots, and a verdict of "clean." This is not an edge case: the baseline snapshot is taken *after* warmup, when the card is already loaded, so a sustained workload is precisely the case that gets missed.

Both are now handled. Clock variance observed inside the window is the governing signal, and throttling is detected from any active state in either snapshot plus a per-iteration NVML check.

## Reaching Tier A

Tier A requires a [`LockedClockPolicy`][sns.clocks.LockedClockPolicy] that successfully pinned the clocks *and* verified they held:

```python
from sns.clocks import LockedClockPolicy

policy = LockedClockPolicy(target_sm_mhz=1400, power_cap_w=250)
t = sns.measure(lambda: a @ a, policy=policy)
assert t.tier is MeasurementTier.A
```

Every `nvidia-smi` write is verified by reading the value back, because **`nvidia-smi` exits 0 when it refuses a request**. On real hardware a lock has reported success and then run 150 MHz below the requested value. The exit code proves nothing; only the readback does.

If the lock cannot be established, `apply()` restores the device and raises `ClockLockError` rather than proceeding.

!!! warning "No telemetry, no Tier A"
    Tier A requires real clock evidence. When NVML is unavailable, no clock samples are collected at all — and because Tier A demands them, it becomes unreachable. Collecting no evidence is correct; collecting bad evidence and calling it Tier A is not.

## Power headroom matters

Locking clocks is necessary and not sufficient. **Power constraints override manual clock settings**: draw enough power and the card throttles regardless of what you asked for.

This makes power headroom a hardware selection criterion. A 72 W card has nowhere to hide — you cannot pin clocks meaningfully below a limit you're already against. A 150 W or 350 W card lets you pin well under maximum so power never becomes the binding constraint. See [Choosing a host](hosts.md).

## Current status

Tier B and Tier C have both been observed on real hardware. **Tier A and CV-triggered Tier C are pinned by unit tests but have never been exercised live**, because no lockable GPU has been available. The first Tier A host should re-verify both paths end to end before its numbers are trusted.
