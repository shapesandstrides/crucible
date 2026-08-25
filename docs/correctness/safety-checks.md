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

## What these cannot catch

A kernel that writes every element, stays in bounds, and computes entirely the
wrong function passes both checks.

They raise the floor. They do not establish correctness, and a report resting
only on them is [tier C](../guide/tiers.md) — the strong verdict is
*unavailable*, not failed.
