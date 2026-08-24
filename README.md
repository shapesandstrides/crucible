# crucible

Honest correctness and timing for Triton kernels.

Answers two questions about a GPU kernel: is it correct across a wide space of shapes and dtypes, and is it actually faster than PyTorch — with a confidence interval, against a baseline re-measured in the same run.

Runs on your own hardware. Kernel source never leaves the machine.

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

round_trip(1e-8, into=[FLOAT16, BFLOAT16])
# fp16 -> 0.0  UNDERFLOW      an ordinary gradient, silently destroyed
# bf16 -> 1.001172e-08        rounded, 0.117% error
```

CPU only, no GPU needed. The simulator is validated bit-for-bit against torch's own fp16, bf16 and fp32 before any result is believed, and every result carries a grade saying whether it was. Built on [`gfloat`](https://github.com/graphcore-research/gfloat) (MIT).

See the [formats guide](https://docs.shapesandstrides.com/guide/formats/) and [reconstructing cbfloat16](https://docs.shapesandstrides.com/guide/formats-cbfloat16/).

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
