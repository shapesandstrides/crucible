# crucible

**Honest correctness and timing for Triton kernels.**

You wrote a Triton kernel to replace a slower PyTorch operation. Before you ship it, two things need to be true: it returns the right answer, and it is genuinely faster.

Most people check both badly, with tools that are documented to be wrong.

```python
import shapesandstrides

t = shapesandstrides.measure(lambda: my_kernel(x, y))
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

shapesandstrides.measure()
  median   0.1925 ms
  95% CI   [0.1920, 0.1925] ms
  p10/p90  0.1915 / 0.1946 ms
  n        50
  tier     B
  clock    range 0.0 MHz, CV 0.0%

shapesandstrides.compare()
  candidate 0.1920 ms  (triton)
  baseline  0.1915 ms  (torch)
  speedup   0.997x
  95% CI    [0.995, 1.000]
  verdict   PARITY - the interval spans 1.0, so we cannot claim a winner
```

That `PARITY` is the point. A naive Triton vector-add *should* tie `torch.add` — both are memory-bandwidth-bound and neither has room to win. A tool that claimed a victory here would be lying to you.

---

## Status

!!! warning "Pre-alpha, but the engine works end to end."

The local test engine is complete: shape generation, an fp64 CPU correctness oracle,
unprivileged metric collection, timing with interleaved measurement, JSON run records,
and a catalog CLI. 174 tests.

Verified against real Triton kernels — three correct and three deliberately broken:

```
PASS  triton_add                     16/16 shapes
PASS  triton_mul                     16/16 shapes
PASS  triton_add_autotuned           16/16 shapes
FAIL  triton_add_drops_tail           5/16 shapes
      minimal failing case: 1025-contiguous-float32  seed=12648431
FAIL  triton_add_assumes_contiguous  15/16 shapes
      minimal failing case: 512x512-noncontiguous-float32  seed=12648437
```

Not built yet: cloud sync, the hosted catalog, and the web dashboard. There is no
`replay` subcommand: a failure reports its shape and seed as data, and you reproduce
it by re-running the same kernel with that seed.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+, an NVIDIA GPU, and PyTorch with CUDA. Start at [Getting started](getting-started.md).
