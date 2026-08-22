# crucible

**Honest correctness and timing for Triton kernels.**

You wrote a Triton kernel to replace a slower PyTorch operation. Before you ship it, two things need to be true: it returns the right answer, and it is genuinely faster.

Most people check both badly, with tools that are documented to be wrong.

```python
import sns

t = sns.measure(lambda: my_kernel(x, y))
print(t.median_ms, t.ci95_lo_ms, t.ci95_hi_ms, t.n, t.tier)
```

Runs on your own hardware. Nothing is uploaded, and no account is required.

---

## What makes it different

**Every number carries an interval.** `triton.testing.do_bench` returns a bare float. A speedup without a confidence interval isn't a result — it's a number you happened to observe once.

**The baseline is re-measured every run.** Never cached, never remembered from last month. That's the only way to tell *"my kernel regressed"* from *"PyTorch got faster."*

**Results state how much they can be trusted.** GPU clocks drift. Every measurement carries a [tier](guide/tiers.md) recording whether the clocks were pinned and verified, floating but measured, or too unstable to report at all.

**You can't accidentally drop the interval.** `TimingResult` defines no `__float__`. Reaching for a bare number raises a `TypeError` instead of silently discarding the uncertainty.

---

## A real result

```
n = 4,194,305 elements, fp32

triton.testing.do_bench (defaults)
  0.1916 ms          <- one number, no interval, no sample count

sns.measure()
  median   0.1925 ms
  95% CI   [0.1920, 0.1925] ms
  p10/p90  0.1915 / 0.1946 ms
  n        50
  tier     B
  clock    range 0.0 MHz, CV 0.0%

sns.compare()
  candidate 0.1920 ms  (triton)
  baseline  0.1915 ms  (torch)
  speedup   0.997x
  95% CI    [0.995, 1.000]
  verdict   PARITY - the interval spans 1.0, so we cannot claim a winner
```

That `PARITY` is the point. A naive Triton vector-add *should* tie `torch.add` — both are memory-bandwidth-bound and neither has room to win. A tool that claimed a victory here would be lying to you.

---

## Status

!!! warning "Pre-alpha. The measurement engine works; its intervals are not yet proven honest."

Phase 0 builds the measurement engine, and it is complete and tested. What it has **not** yet passed is its own acceptance gate: 50 runs of an identical kernel spread over 20 minutes, checking that the variation *between* runs fits inside the intervals each run reports. That requires a GPU whose clocks can be locked, and no such host has been available yet.

Correctness checking — shape-space generation, the fp64 CPU oracle, minimal failing cases — is Phase 1 and not built.

See [Choosing a host](guide/hosts.md) for what qualifies, and [Why this exists](why.md) for the evidence the design rests on.

---

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+, an NVIDIA GPU, and PyTorch with CUDA. Start at [Getting started](getting-started.md).
