# Getting started

## Install

```bash
git clone https://github.com/shapesandstrides/crucible
cd crucible
pip install -e ".[dev]"
```

Requirements: Python 3.11+, an NVIDIA GPU, PyTorch built with CUDA. Triton is only needed if you intend to measure Triton kernels — the library itself doesn't require it.

Verify the install:

```bash
python -m pytest -q
```

Most of the suite is pure logic and runs without a GPU. Tests marked `gpu` are skipped automatically when no CUDA device is present.

## Check what your machine can measure

Before trusting any number, find out what your hardware permits:

```bash
PROBE_LABEL=my-laptop python scripts/probe_host.py --quick
```

```
============================================================
  my-laptop: NOT MEASUREMENT-CAPABLE (clock lock refused) -- correctness node only
============================================================
{
  "arch_family": "Ampere",
  "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
  "clock_lock_applied": false,
  "clock_range_mhz": 495,
  "clock_cv_pct": 5.1,
  ...
}
```

That output is a normal, expected result on a laptop or any consumer card. It doesn't stop you working — it tells you which [tier](guide/tiers.md) your measurements will carry. See [Choosing a host](guide/hosts.md).

## Your first measurement

```python
import torch
import sns

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

t = sns.measure(lambda: a @ a)

print(f"{t.median_ms:.4f} ms  95% CI [{t.ci95_lo_ms:.4f}, {t.ci95_hi_ms:.4f}]  n={t.n}  tier {t.tier.value}")
```

`measure()` warms up 200 times, flushes the L2 cache between iterations, times each one with a pre-allocated pair of CUDA events, and returns 30 samples with a bootstrap confidence interval. See [Measuring](guide/measuring.md) for what each of those does and why.

!!! note "The `@jit` file requirement"
    Triton's `@triton.jit` refuses to compile from `python -c` or a REPL — kernels must live in a real `.py` file. This is a Triton constraint, not ours.

## Comparing against PyTorch

```python
import torch
import sns

x = torch.randn(4_194_304, device="cuda")
y = torch.randn(4_194_304, device="cuda")

c = sns.compare(
    lambda: my_triton_add(x, y),   # candidate
    lambda: x + y,                  # baseline
)

print(f"{c.speedup:.3f}x  CI [{c.speedup_ci_lo:.3f}, {c.speedup_ci_hi:.3f}]  tier {c.tier.value}")
```

Read the interval, not the point estimate. If it spans `1.0`, you have not demonstrated a difference — regardless of what the median says. See [Comparing](guide/comparing.md).

## Using it in pytest

The library is the engine, so it drops straight into an existing test suite:

```python
import pytest
import sns


@pytest.mark.gpu
def test_my_kernel_beats_torch():
    c = sns.compare(lambda: my_kernel(x, y), lambda: x + y)

    if not c.is_performance_valid:
        pytest.skip(f"tier {c.tier.value}: measurement too unstable to judge")

    assert c.speedup_ci_lo > 1.0, (
        f"no demonstrated speedup: {c.speedup:.3f}x "
        f"CI [{c.speedup_ci_lo:.3f}, {c.speedup_ci_hi:.3f}]"
    )
```

Asserting on `speedup_ci_lo` rather than `speedup` is what makes this test non-flaky: it fails only when the *entire* interval clears 1.0, not when a noisy median happens to.

## Next

- [Measuring](guide/measuring.md) — what `measure()` does and how to configure it
- [Comparing](guide/comparing.md) — fresh baselines and reading intervals
- [Measurement tiers](guide/tiers.md) — how much to trust a result
- [Choosing a host](guide/hosts.md) — what makes hardware measurement-capable
