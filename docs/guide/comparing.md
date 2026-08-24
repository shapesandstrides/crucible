# Comparing

`shapesandstrides.compare()` measures a candidate against a baseline and returns a [`ComparisonResult`][shapesandstrides.ComparisonResult].

```python
c = shapesandstrides.compare(
    lambda: my_kernel(x, y),   # candidate
    lambda: x + y,             # baseline
)
```

```python
c.speedup         # 1.31    baseline.median_ms / candidate.median_ms
c.speedup_ci_lo   # 1.28
c.speedup_ci_hi   # 1.34
c.candidate       # full TimingResult
c.baseline        # full TimingResult
c.tier            # the worse of the two halves
```

Speedup above `1.0` means the candidate is faster.

## The baseline is never cached

This is the single load-bearing rule of the project, and it costs roughly double the measurement time. Accept it.

Every call re-measures the baseline, in the same process, under the same clock policy, immediately adjacent to the candidate. It is never read from a file, remembered from an earlier run, or shared between calls.

Without that, you cannot distinguish two completely different situations:

> *My kernel regressed.*

> *PyTorch got faster and my kernel stood still.*

Both look identical if you compare today's kernel against a baseline you measured in March. One means you broke something; the other means upstream caught up and your kernel may no longer be worth maintaining. Telling them apart is the reason this tool exists.

It also has a consequence worth understanding: because the reported quantity is a *ratio* against a baseline measured on the same hardware in the same moment, it is dimensionless. Absolute timings across different machines are incomparable. Speedups are.

## Read the interval, not the estimate

```python
if c.speedup_ci_lo > 1.2:
    verdict = "FASTER"
elif c.speedup_ci_hi < 0.95:
    verdict = "SLOWER"
else:
    verdict = "PARITY"   # the interval spans 1.0
```

A `speedup` of `1.31` with a CI of `[0.94, 1.72]` is **not** a 31% win. It is a measurement too noisy to support any claim. The point estimate on its own tells you almost nothing.

`ratio_ci` bootstraps both sides independently and propagates the uncertainty from each into the ratio. A naive ratio-of-medians throws that away and reports false precision.

## Check the tier before you believe it

```python
if not c.is_performance_valid:
    print(f"tier {c.tier.value} — too unstable to judge")
```

A comparison is only as trustworthy as its worse half, so `ComparisonResult.tier` returns the worse of the two. A pristine candidate measurement paired with a throttled baseline is a throttled comparison. See [Measurement tiers](tiers.md).

## Known limitation: measurement order

The candidate is measured first and the baseline second. On unlocked hardware the second measurement runs on a warmer, more boosted GPU.

This is measurable. Comparing *identical work* on both sides on an RTX 3060 yields a speedup of **0.962** rather than `1.0` — about 4% of systematic bias, against a `FASTER` threshold of 1.2×.

Tier A pinning largely removes the effect. Tier B and C comparisons carry it. Interleaving the two sides so drift affects both equally is planned work.

Until then: treat sub-5% differences on unlocked hardware as unproven, whatever the interval says.

## In a test suite

```python
def test_kernel_is_faster():
    c = shapesandstrides.compare(lambda: my_kernel(x, y), lambda: x + y)

    if not c.is_performance_valid:
        pytest.skip(f"tier {c.tier.value}")

    assert c.speedup_ci_lo > 1.0, (
        f"no demonstrated speedup: {c.speedup:.3f}x "
        f"CI [{c.speedup_ci_lo:.3f}, {c.speedup_ci_hi:.3f}]"
    )
```

Asserting on `speedup_ci_lo` rather than `speedup` is what keeps this from flaking. It fails only when the whole interval clears the threshold — which is the same standard you should hold your own claims to.
