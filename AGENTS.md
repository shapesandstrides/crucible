# AGENTS.md

Orientation for coding agents working in this repository. Humans may find it
useful too, but the audience is an agent that has just cloned this and has no
other context.

## What this project is

A library and CLI that answers two questions about a GPU kernel honestly:

1. Is it correct across a wide space of shapes and dtypes?
2. Is it actually faster than PyTorch, with a confidence interval, against a
   baseline re-measured in the same run?

Everything runs on the user's own hardware. Kernel source never leaves the
machine. That constraint is a product promise, not an implementation detail:
do not add anything that uploads source, and be careful with input shapes,
which leak model architecture and are opt-in only.

## Rules that override convenience

These are the product. If a change makes the tool nicer to use by weakening
one of them, the rule wins and the change is wrong.

1. **Re-measure the PyTorch baseline in every run. Never cache it.**
   Distinguishing "my kernel regressed" from "torch got faster" is the entire
   point.
2. **Never report a bare number.** Every timing carries a confidence interval
   and a sample count. This is enforced structurally: `TimingResult` defines no
   `__float__`, so a bare float cannot be extracted even by accident. Do not
   add one.
3. **The tool adjudicates, never the submitted code.** Pass/fail and timing are
   computed from raw observation. Kernel code never reports its own result.
   This is a deliberate countermeasure to a documented reward-hacking mode.
4. **Never present a measurement or an oracle as stronger than it was.** Label
   the quality tier. Report distributions, not means: a mean hides the cliff
   this tool exists to find.
5. **`INCOMPATIBLE`, `INCORRECT` and `ERROR` are three different things.**
   INCOMPATIBLE means the tool could not judge the kernel. INCORRECT means it
   judged it and the kernel is wrong. ERROR means the tool fell over. Never
   collapse them, in output or in types.
6. **Store the minimal failing case with its seed** so failures replay.

## Never fabricate a measurement

If you did not run it, say so. Do not write a number into a doc, a README, a
commit message or a test fixture unless it came out of an actual run. Do not
describe output you have not seen. If you cannot run something because there is
no GPU, say that plainly rather than guessing at plausible output.

This matters more here than in most repositories, because the project's only
value is that its numbers can be trusted.

## Layout

```
src/shapesandstrides/
  runner.py        the test() entry point: correctness, then timing, then a record
  correctness.py   shape sweep, verdicts, shrinking to a minimal case
  verify.py        the @verify decorator and verify_kernel()
  pytest_plugin.py collects @verify-marked kernels, zero config
  cli.py           typer app: runs, show, compare, rm, verify
  timing.py        measurement; TimingResult lives here
  stats.py         intervals, quantiles, distributions
  oracle.py        fp64 CPU adjudication
  tolerance.py     tolerance policy
  budget.py        error budget: grade against the unfused chain
  buffers.py       poison fill, canary padding
  reference.py     reference construction, incl. reference_lowp
  shapes.py        shape generation, tile-straddling
  tiles.py         tile discovery
  env.py           nvidia-smi fingerprint, throttle and clock state
  telemetry.py     in-process NVML sampling (nvidia-smi is far too slow to
                   sample inside a measurement loop)
  clocks.py        clock locking and quality tiers
  records.py       run records on disk (~/.shapesandstrides)
  metrics.py       derived metrics
  types.py         shared enums and models
  formats/         numeric-format lab, CPU only, built on gfloat
examples/          runnable kernels, three broken on purpose
scripts/           probe_host.py, validate_timing.py, record_gradients.py,
                   make_icon.py
docs/              the mkdocs site published at docs.shapesandstrides.com
```

## Running things

```bash
pip install -e ".[dev]"
python -m pytest -q          # full suite; takes several minutes
```

Tests marked `gpu` skip without a CUDA device. Most of the suite is pure logic
and runs anywhere. `shapesandstrides.formats` needs no GPU at all.

To see the tool work end to end, on a machine with an NVIDIA GPU:

```bash
shapesandstrides verify examples/verified_kernels.py   # exits 1, on purpose
```

Docs:

```bash
pip install -e ".[docs]"
mkdocs build --strict
```

A test asserts that the correctness hub links every sibling page on disk, so a
new page under `docs/correctness/` must add its row to the hub or CI fails.

## Traps that have already cost someone a day

- **`@triton.jit` needs a real file.** It reads its own source with `inspect`,
  so it dies in a heredoc or a REPL with `@jit functions should be defined in a
  Python file`. Put every Triton experiment in a file on disk.
- **A reference containing `.half()` still rounds, even in float64.** The
  "golden" answer is then no better than the thing it grades and every number
  it produces is quietly meaningless. References must be pure mathematics.
- **`error_budget` replaces the atol/rtol verdict. Do not make it additive.**
  Requiring both to pass reintroduces the exact false failure the budget exists
  to eliminate.
- **`torch.quantile` raises above ~16.7M elements** (2**24). Shapes that large
  appear in the default sweep. `budget._distribution` sorts and indexes instead.
- **Determinism checking has a real blind spot.** A kernel with a genuine race
  can return the same wrong answer every run, because block scheduling is
  stable. A determinism pass is not a correctness pass. `compute-sanitizer
  --tool racecheck` does catch these.

## Things that do not exist

Do not reference these as if they work:

- **There is no `replay` subcommand.** A failure reports `shape=... seed=...`
  as data. An earlier version printed `shapesandstrides replay ...`, which was
  a false affordance and has been removed.
- No cloud sync, no hosted catalog, no web dashboard.
- Nothing is published to PyPI yet. Install from source.

More generally: do not create false affordances. If output names a command, a
flag or a file, it must exist. An agent reading this tool's output will act on
what it says.

## Contributing

See `CONTRIBUTING.md` for the licence boundary and the CLA, and
`CODE_OF_CONDUCT.md`. Changes go through a pull request; the PR template lists
the rules above as a checklist. Every contributor signs the CLA before merge.
