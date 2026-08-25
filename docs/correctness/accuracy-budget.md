# Accuracy budget

## The problem

You fused five operations into one kernel. You test it against the unfused
torch chain. It fails.

The kernel is not wrong. Fusion keeps intermediate values in registers at
float32 instead of round-tripping them through float16 memory, and it
accumulates in a different order. Your kernel disagrees with torch because
**it is more accurate than torch**.

`torch.allclose` cannot tell that apart from a bug. So the test fails, and
someone "fixes" a kernel that was right.

## The question worth asking

Not *"does it match the reference"* but:

> Is it at least as accurate as the code it replaces?

That takes three values, not two:

| | What it is | Where it comes from |
|---|---|---|
| **golden** | the chain in float64 — as close to truth as is cheap | `oracle.reference_fp64` |
| **reference** | the unfused chain at production precision, rounding error and all | `oracle.reference_lowp` |
| **kernel** | your fused kernel | you |

Measure how far the last two sit from the golden. Pass if the kernel is no
worse than the reference by more than a stated margin.

## Using it

```python
import torch

from shapesandstrides import check

report = check(
    fused_attention,
    reference=lambda q, k, v: torch.softmax(q @ k.mT / 8.0, -1) @ v,
    error_budget=2.0,
)

b = report.outcomes[0].budget
print(b.kernel_p99_ulp, b.reference_p99_ulp, b.ratio_p99)
```

`error_budget` is the margin. `2.0` means the kernel's p99 error may be up to
twice the reference's before it is called wrong. There is nothing sacred about
`2.0` — it is a judgement, which is why it comes back in the result as
`b.margin` rather than staying hidden inside the tool.

!!! warning "The reference must contain no dtype casts"
    The reference is evaluated in float64 to produce the golden. A reference
    with `.half()` inside it still rounds to float16 in float64, so the
    "golden" is no better than the thing it is meant to grade, and every
    result is quietly meaningless. Write the mathematics; let crucible choose
    the precision.

## The verdict it replaces

When `error_budget` is set it **replaces** the `atol`/`rtol` verdict rather
than adding to it. Requiring both would leave the false failure this exists to
fix still failing.

The tolerance comparison still runs and is still recorded on
`outcome.oracle`, so nothing is hidden — only the verdict changes.

## Reading the numbers

Error is reported in **ULP** — units in the last place — not absolute
difference. An absolute tolerance of `1e-3` is absurdly tight at magnitude
`1e8` and uselessly loose at `1e-8`, and one kernel's inputs routinely span
both. One ULP means "the smallest difference this dtype can represent here",
which is the same statement everywhere.

Six numbers come back, not one — p50, p99 and max, for both kernel and
reference. A p50 near zero with a p99 in the thousands is a kernel that is
right almost everywhere and wrong in one corner, which is the interesting
failure and precisely what a mean would hide.

The pass/fail threshold uses **p99**, not max, so a single outlier element does
not condemn a kernel. The max is reported so you can apply a stricter rule.

`ratio_p99` is `None` when the reference happens to be bit-exact. There is no
ratio to take, and the check falls back to an absolute ULP floor.

## Measured example

A three-step chain — `tanh(x @ y + b)` in float16 — compared two ways. The
"unfused" path rounds to float16 after every step, as it would if each
intermediate were written to memory. The "fused" path keeps intermediates in
float32 and rounds once at the store.

```
   n    fused p99    unfused p99    ratio
 256          0.2            0.2    0.871
 512          0.1            0.1    0.918
1024          0.0            0.0    0.979

torch.allclose(fused, unfused, atol=1e-3, rtol=1e-3)  ->  False
```

*Measured on an RTX 3060 Laptop, torch 2.7.1+cu128, seed 0.*

`allclose` reports that the two disagree and stops there. The budget says the
fused version carries **13% less error** at n=256 and correctly passes it.

Two honest observations about this example. The advantage is real but modest
here — `tanh` compresses its output, and a three-step chain has little room to
accumulate; a longer chain or a wider dynamic range separates them further.
And grading the *unfused* path as though it were the kernel gives
`ratio = 1.119`, which also passes at `margin=2.0` — being 12% worse is not
being broken, and the tool should not pretend otherwise.

## What it does not do

**It will not tell you which fused stage introduced the error.** That needs the
partially-fused variants, which only the tool that did the fusing can emit.

**It still needs an answer key** — the unfused chain. If no equivalent exists
anywhere, the checks that need no reference are the ones available to you, and
they support a weaker claim: [tier C](../guide/tiers.md).

**It grades the primary output only.** A kernel returning several tensors gets
a budget on output `0`; the rest are still covered by the tolerance comparison
in `outcome.outputs`.
