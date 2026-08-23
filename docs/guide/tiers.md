# Measurement tiers

Every result carries a tier recording how much it can be trusted. Tiers are assigned from observed telemetry, never assumed, and never silently upgraded.

| Tier | Conditions | What it means |
|---|---|---|
| **A** | Clocks pinned, verified held under load, range ≤ 30 MHz | Fully trustworthy. Comparable across runs. |
| **B** | Clocks floating; variance measured and disclosed | Usable, but only for differences large enough to survive the noise. |
| **C** | Clock CV > 3%, or a hardware throttle assertion fired | **No performance verdict is valid.** |

```python
t.tier                    # MeasurementTier.B
t.is_performance_valid    # False only for Tier C
```

## The problem tiers solve

GPU clocks are not constant. They boost, they drop under thermal load, and they get capped by power limits. Timing a kernel while the clock moves measures the clock, not the kernel.

The obvious fix is to pin the clocks and refuse to measure otherwise. That's correct for hardware you control and useless for a tool that runs on hardware other people control — most consumer GPUs simply will not permit it. A tool that refuses to run for most of its audience helps nobody.

So the principle survives in a weaker form: **never *silently* measure with floating clocks.** Measure the variance, classify it, and put the classification next to every number.

## Why throttle flags alone aren't enough

You might expect `nvidia-smi`'s throttle reasons to answer "was this measurement clean?" They don't, and testing on real hardware (an RTX 3060 Laptop GPU) found two separate problems: one with *all* the flags, and one specific to the *software*-reported ones.

**They miss variance that isn't throttling.** Under 30 s of sustained load the SM clock swung **495 MHz — 5.1% CV** — while every throttle flag, hardware and software, stayed silent. Two identical runs even disagreed about whether throttling fired at all. Boost and idle transitions move the clock hundreds of megahertz without setting a single flag. This is the original reason observed clock variance became the governing signal, not the flags.

**The software flags are simply not trustworthy.** `sw_power_cap` read `Active` at idle. That sounds alarming until you notice it means the card is at its power limit — the normal state of *every* GPU under load, including datacenter parts, not evidence of instability. `sw_thermal_slowdown` was worse: it read `Active` at idle, 55 °C, 18 W — a cold card doing nothing. That flag is simply stuck on. Neither is usable evidence of anything.

**The hardware assertions are different.** `hw_thermal_slowdown` and `hw_power_brake_slowdown` stayed `Not Active` throughout testing, including under the sustained load that produced the 495 MHz swing above. Both are asserted by the GPU's own hardware safety circuits, not by driver or vendor software policy, and neither was ever seen stuck. That makes them trustworthy enough to disqualify a window by themselves.

**The rule this settles on:** observed clock variance governs; a hardware-asserted throttle (`hw_thermal_slowdown` or `hw_power_brake_slowdown`) disqualifies a measurement outright; software throttle flags (`sw_power_cap`, `sw_thermal_slowdown`) are recorded on [`TimingResult`][shapesandstrides.TimingResult] as metadata — `power_capped`, `sw_thermal_flagged` — and never gate the tier. Throttling is checked from any hardware-asserted state in either the before/after snapshot or a per-iteration NVML sample; the gating decision itself lives in [`assign_tier`][shapesandstrides.clocks.assign_tier].

!!! note "Also handled: throttling that never stops"
    An earlier version compared only a snapshot before the loop against one after, and treated a *difference* as evidence of throttling. A GPU throttled for the entire window produces `before == after == "Active"` — identical snapshots, and a false verdict of "clean." The baseline snapshot is taken *after* warmup, when the card is already loaded, so a sustained workload was precisely the case that got missed. Checking each snapshot independently, plus the per-iteration NVML sample, closes this.

## Reaching Tier A

Tier A requires a [`LockedClockPolicy`][shapesandstrides.clocks.LockedClockPolicy] that successfully pinned the clocks *and* verified they held:

```python
from shapesandstrides.clocks import LockedClockPolicy

policy = LockedClockPolicy(target_sm_mhz=1400, power_cap_w=250)
t = shapesandstrides.measure(lambda: a @ a, policy=policy)
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
