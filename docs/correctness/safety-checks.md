# Safety checks

Checks that need **no answer key**. They cannot tell you a kernel computes the
right function — only that it is not doing something plainly wrong. That makes
them cheap, and it makes them the only checks available when your kernel has no
equivalent anywhere.

## Poison fill — did it write everything?

Fill the output with NaN before launching. Anything still NaN afterwards is an
element the kernel never wrote.

```python
import torch

from shapesandstrides.buffers import inspect_buffer, poisoned_output

out = poisoned_output((1000,), torch.float32, "cuda")
my_kernel[(8,)](src, out, 1000, BLOCK=128)

report = inspect_buffer(out)
assert report.unwritten_count == 0, f"{report.unwritten_count} elements unwritten"
```

`poisoned_output` returns an ordinary tensor. Pass it to a kernel, index it,
reshape it — nothing needs to be unwrapped or carried alongside.

**This is the highest-yield check here**, because the bug it finds is the most
common one in Triton. 1000 elements with `BLOCK=128` is seven full blocks and a
tail of 104. Guard the store against the wrong bound and the tail is silently
skipped:

```python
mask = offs < (n // BLOCK) * BLOCK   # wrong: drops the final partial block
mask = offs < n                      # right
```

Without poison fill, those 104 elements hold whatever was in that memory
before — usually plausible floats. The test either passes, or fails somewhere
downstream for a reason that looks unrelated to masking.

## Canary padding — did it write too far?

`poisoned_output` also allocates eight elements past the logical end and writes
a sentinel into them. If the kernel writes out of bounds, the sentinel changes.

```python
report = inspect_buffer(out)
assert report.canary_intact, "kernel wrote past the end of its output"
```

`compute-sanitizer` finds this too and finds more, but it is orders of
magnitude too slow to run on every candidate in an autotuning loop. The canary
costs eight elements and a comparison.

**`canary_present` is reported separately from `canary_intact`.** A tensor you
allocated yourself has no sentinel, so there is nothing to check — and
answering "intact" for a check that never ran would claim a guarantee that does
not exist. Same distinction [rule 7](../guide/tiers.md) draws between a failure
and an absent verdict.

## Determinism — is there a race?

Run the same input repeatedly. Any variation means a race, or memory read
before it was written.

```python
from shapesandstrides.buffers import check_determinism

report = check_determinism(lambda: run_my_kernel(x), runs=20)
assert report.passed, f"varied by {report.max_deviation} on {report.varying_runs} runs"
```

The callable must genuinely re-launch the kernel. A closure over an
already-computed tensor tests nothing and always passes.

`varying_runs` is a count, not a flag, because a race that fires on every run
and one that fires occasionally are very different debugging problems.

### A pass means less than you would hope

Twenty runs will not surface a race that fires one time in a thousand. **A pass
is weak evidence; a failure is conclusive.** That asymmetry is worth having for
a check this cheap, but it should be stated rather than implied.

Worse, and more interesting: **a race can be stably scheduled and look
perfectly deterministic.** Here is a kernel with an obvious race — 32 blocks
read-modify-write one address with no atomic:

```python
@triton.jit
def racing_sum(src, dst, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    partial = tl.sum(tl.load(src + offs))
    tl.store(dst, tl.load(dst) + partial)   # no atomic, no barrier
```

Measured on an RTX 3060:

```
true sum        2045.07
returned          63.21     <- one block's partial, last writer wins
determinism     passed=True, varying=0/49, max_deviation=0.0
```

The kernel is wrong by a factor of 32, returns the same wrong answer 50 times
in a row, and determinism checking reports it clean. Every block is scheduled
in the same order every launch, so there is no variation to detect.

**Determinism is not correctness.** It rules out one failure mode and says
nothing about the rest. This particular kernel is caught immediately by an
[accuracy budget](accuracy-budget.md) against `torch.sum`, and by
`compute-sanitizer --tool racecheck`, which reasons about the memory model
instead of sampling outcomes.

## What these cannot catch

A kernel that writes every element, stays in bounds, returns the same answer
every time, and computes entirely the wrong function passes all three checks.
The racing kernel above is exactly that.

They raise the floor. They do not establish correctness, and a report resting
only on them is [tier C](../guide/tiers.md) — the strong verdict is
*unavailable*, not failed.
