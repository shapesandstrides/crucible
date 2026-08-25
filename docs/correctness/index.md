# Correctness

Two questions decide which check you need.

**Do you have an answer key?** A named function, or a short torch expression,
that computes the same thing your kernel does.

**Is your kernel fused?** If it collapses several operations into one, it will
deliberately disagree with the unfused version — often by being *more* accurate
than it.

| Your situation | Use |
|---|---|
| A named torch equivalent exists | [Testing a kernel](../guide/testing.md) |
| Fused, but you can write the unfused chain | [Accuracy budget](accuracy-budget.md) |
| You need to know how strong a verdict you have | [Oracle tiers](../guide/tiers.md) |

## Why "no answer key" is the common case

The kernels most worth checking are the hardest to check. If you fused five
operations into one, there is no single torch function that does the same thing
— that is the *point* of having fused them. Anything that generates kernels
rather than hand-writing them lives here permanently: compilers, autotuners,
and language models writing Triton.

Checks that need no reference cannot tell you a kernel computes the function
you intended — only that it is not doing something plainly wrong. That is a
weaker claim, and it is recorded as one: a report resting only on those is
[tier C](../guide/tiers.md), which means the strong verdict is **unavailable**,
not that it failed.

## Why a fused kernel fails a test it should pass

Fusion keeps intermediate values in registers at float32 rather than writing
them out to float16 memory, and it accumulates in a different order. So a
correct fused kernel returns different numbers from the unfused chain — usually
*better* ones.

`torch.allclose` cannot distinguish "different because it is more accurate"
from "different because it is broken". The [accuracy budget](accuracy-budget.md)
can, by measuring both against a float64 golden and asking whether the kernel is
any worse than the code it replaced.
