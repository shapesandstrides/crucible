# crucible

[![tests](https://github.com/shapesandstrides/crucible/actions/workflows/tests.yml/badge.svg)](https://github.com/shapesandstrides/crucible/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-shapesandstrides.com-blue)](https://docs.shapesandstrides.com/)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Honest correctness and timing for Triton kernels.

Answers two questions about a GPU kernel: is it correct across a wide space of shapes and dtypes, and is it actually faster than PyTorch — with a confidence interval, against a baseline re-measured in the same run.

Runs on your own hardware. Kernel source never leaves the machine.

## Install

Not on PyPI yet. From source:

```bash
git clone https://github.com/shapesandstrides/crucible.git
cd crucible
pip install -e ".[dev]"
```

## See it fail

Six kernels, three of them broken on purpose. Needs an NVIDIA GPU.

```bash
shapesandstrides verify examples/verified_kernels.py
```

```
kernel                        verdict    shapes  oracle                     minimal failing case
fused_add                     CORRECT     16/16  A:torch_op:torch.add
fused_mul                     CORRECT     16/16  A:torch_op:torch.mul
fused_add_autotuned           CORRECT     16/16  A:torch_op:torch.add
fused_add_drops_tail          INCORRECT    5/16  A:torch_op:torch.add       1025-contiguous-float32
fused_add_assumes_contiguous  INCORRECT   15/16  A:torch_op:torch.add       512x512-noncontiguous-float32
rowsum                        INCORRECT    0/16  A:expression:<expression>  1025-contiguous-float32

6 kernel(s) on device=cuda, 3 failed
```

Exit code 1, so it can block a merge. Each failure is shrunk to the smallest
shape that still reproduces it: `fused_add_drops_tail` passes every aligned
shape and dies at 1025, one element past the tile. `fused_add_assumes_contiguous`
passes 15 of 16 and only breaks on a non-contiguous input.

No GPU? `shapesandstrides.formats` runs on CPU alone — see below.

## Status

Pre-alpha. Usable locally; nothing is published to PyPI yet.

**Working and tested:**

- **Correctness across a shape and dtype space** — generates shapes that straddle tile boundaries, adjudicates every output against an fp64 CPU oracle, and shrinks any failure to the smallest case that still reproduces, with its seed.
- **Honest timing** — every measurement carries a confidence interval, a sample count and a quality tier. `TimingResult` defines no `__float__`, so a bare number cannot be extracted even by accident.
- **A one-line gate** — `@verify(against="torch.add")` on a kernel, plus a pytest plugin that collects marked kernels with zero configuration, and `shapesandstrides verify <path>` exiting 0, 1, or 5 (nothing found — deliberately not 0).
- **Graded verdicts** — every correctness verdict records what adjudicated it, so "matches PyTorch" and "matches the reference you wrote yourself" are not the same claim.
- **Numeric formats** — see below.

**Not built:** cloud sync, a hosted catalog, a web dashboard.

### Numeric formats

`shapesandstrides.formats` answers a different question: *what does this float format do to my numbers?*

Declare a format by its parameters — including formats no vendor has fully published, like Cerebras's cbfloat16 — and find out what rounds, what overflows, and what silently becomes zero.

```python
from shapesandstrides.formats import FLOAT16, BFLOAT16, round_trip

for o in round_trip(1e-8, into=[FLOAT16, BFLOAT16]).outcomes:
    print(f"{o.format_name:9} -> {o.result:<13.7g} {o.outcome.value:<9} {o.rel_error:>7.3%} error")
```

```
binary16  -> 0             underflow 100.000% error
bfloat16  -> 1.001172e-08  rounded    0.117% error
```

An ordinary gradient. fp16 destroys it silently; bf16 keeps it to 0.117%.

CPU only, no GPU needed. The simulator is validated bit-for-bit against torch's own fp16, bf16 and fp32 before any result is believed, and every result carries a grade saying whether it was. Built on [`gfloat`](https://github.com/graphcore-research/gfloat) (MIT).

Some of what it has already measured, on 841,471 gradients recorded from a real transformer backward pass:

- **Summing 1,000 values in bf16 loses 68% of the answer.** Accumulator stagnation — once the running total is large enough, each new addend falls below its ULP and is discarded.
- **fp16 silently zeroes 771 real gradients; a 6-exponent/9-mantissa format at the conventional bias zeroes 27.** A five-notch bias shift zeroes none.
- **Shifting the exponent bias by N is bit-identical to loss-scaling by 2ⁿ**, across 14,005 values and six shift sizes — and the bias shift is free, because the bias is only what you subtract when reading an exponent.
- **A clearly-labelled assumption still produced a confident wrong conclusion**, and only replacing it with a recording found that out.

See the [formats guide](https://docs.shapesandstrides.com/guide/formats/), [reconstructing cbfloat16](https://docs.shapesandstrides.com/guide/formats-cbfloat16/), and [findings](https://docs.shapesandstrides.com/guide/formats-findings/).

### Host probe

`scripts/probe_host.py` is a throwaway utility that classifies whether a given GPU host can produce trustworthy measurements — that is, whether it can pin its clocks and hold them under load.

```bash
PROBE_LABEL=my-host python scripts/probe_host.py -o my-host.json
```

It verifies every `nvidia-smi` control write by reading the value back, because `nvidia-smi` exits 0 when it refuses one. It restores every setting it touches before exiting.

## Requirements

Python 3.11+ and PyTorch. Kernel testing and timing need an NVIDIA GPU with CUDA; `shapesandstrides.formats` runs on CPU alone.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Permissive on purpose. This tool is only useful if it can sit inside other
people's CI and be cited as a neutral reference, and that rules out copyleft or
source-available terms. Two things are deliberately *not* granted: trademarks,
and the right to present your own results as certified by this project
(Apache-2.0 §6, restated in `NOTICE`).
