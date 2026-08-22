# Measuring

`sns.measure()` times a callable and returns a [`TimingResult`][sns.TimingResult] — a distribution with an interval and a trust tier, never a bare number.

```python
t = sns.measure(lambda: a @ a, warmup=200, iters=50)
```

## What happens on each call

**1. The clock policy is applied.** By default this is [`UnlockedClockPolicy`][sns.clocks.UnlockedClockPolicy], which changes nothing and honestly reports that clocks were floating. Pass a [`LockedClockPolicy`][sns.clocks.LockedClockPolicy] to pin them — see [Measurement tiers](tiers.md).

**2. Warmup runs 200 times by default.** This number is deliberate. `triton.testing.do_bench` defaults to `warmup=25`, which on a typical kernel amounts to **two actual calls**, and [underestimates by roughly 30%](https://github.com/openai/triton/issues/2306) — 11.23 ms reported against a true 8.79 ms. That issue was filed in September 2023 and has no maintainer response. Lowering our default reintroduces exactly that error.

**3. A calibration probe decides whether to loop.** Kernels shorter than ~10 µs cannot be timed reliably: CUDA event overhead and system variance dominate the signal. When one iteration falls below that floor, `measure()` runs the callable `inner_reps` times inside a single timed region and divides. The alternative — reporting the number anyway — is measuring your own instrumentation.

**4. CUDA events are pre-allocated.** All `2 × iters` events are constructed before the loop starts. Allocating one inside a timed region would inflate that sample.

**5. Each iteration flushes L2 first.** A scratch buffer of `max(256 MB, 2 × L2 size)` is zeroed before `start.record()`, so every iteration begins with a cold cache. Without this, iteration *n* reads data iteration *n−1* left resident and reports a bandwidth you will never see in production.

**6. Telemetry is sampled in-process.** SM clock and throttle state are read through NVML every iteration, after `end.record()` so sampling never lands between the two events.

!!! info "Why NVML and not `nvidia-smi`"
    An earlier version shelled out to `nvidia-smi`. Measured on a real machine, that costs **~68 ms per call** — so sampling stalled the host long enough that the GPU went idle, and every reading reflected an idle clock. Variance came back `0.0` always, which silently disabled both tier gates. NVML runs in-process at microsecond latency and can be sampled without perturbing what it measures.

**7. The policy is restored,** in a `finally` block, even if measurement raises.

## Reading the result

```python
t.median_ms      # 0.1925
t.ci95_lo_ms     # 0.1920   95% bootstrap CI of the median
t.ci95_hi_ms     # 0.1925
t.p10_ms         # 0.1915   distribution, not just centre
t.p90_ms         # 0.1946
t.n              # 50       sample count
t.samples_ms     # [...]    every raw sample
t.tier           # MeasurementTier.B
t.clock_cv_pct   # observed clock variation during the window
t.throttle_fired # whether the GPU throttled at any point
t.inner_reps     # how many calls per timed region
```

The confidence interval is a **percentile bootstrap of the median**, seeded (`0xC0FFEE`, 2000 resamples) so identical samples always produce an identical interval. Reproducibility is a requirement here, not a convenience.

!!! danger "You cannot extract a bare number"
    ```python
    >>> float(t)
    TypeError: float() argument must be a string or a real number, not 'TimingResult'
    ```
    `TimingResult` deliberately defines no `__float__`, `__int__`, or `__index__`. Dropping the interval has to be a decision you write out, not something that happens by accident when a result flows into arithmetic.

## Configuration

| Parameter | Default | Notes |
|---|---|---|
| `warmup` | `200` | Lowering this reintroduces the `do_bench` error. |
| `iters` | `30` | Minimum enforced; fewer raises `ValueError`. |
| `device` | `0` | CUDA device index. |
| `flush_l2` | `True` | Disable only if you deliberately want warm-cache numbers. |
| `policy` | `UnlockedClockPolicy()` | See [tiers](tiers.md). |
| `min_duration_us` | `10.0` | The floor below which the inner loop engages. |

## Known limitation

The reported CI is a bootstrap over samples **within a single measurement window**. It captures sampling error inside that window, but it cannot capture run-to-run variability *between* windows.

On unlocked hardware that gap is large. A real measurement on an RTX 3060 laptop produced a cross-run spread of **0.600 ms** against a widest within-run CI of **0.0015 ms** — a 400× discrepancy. `scripts/validate_timing.py` measures exactly this, and folding between-window variance into Tier B intervals is planned work.

Treat a Tier B interval as "how precisely did I measure this window," not "how reproducible is this number tomorrow."
