# Testing a kernel

`sns.test()` runs the whole pipeline: correctness across a shape space, then timing, then a stored record.

```python
import sns
from sns.shapes import ShapeTier

record = sns.test(
    my_triton_kernel,
    reference=lambda a, b: a + b,
    kernel_name="fused_add",
    op_name="add",
    tier=ShapeTier.FAST,
    dtypes=["float16", "float32"],
)
```

## What happens, in order

**1. Correctness runs first**, across the shape tier you asked for. Each shape gets deterministic inputs derived from a seed, the kernel's output is compared against a float64 CPU reference, and every output is checked — not just the first, since fused kernels commonly return several tensors.

**2. Timing runs only if correctness passed.** This is deliberate. Timing a kernel already known to produce wrong answers yields a number that means nothing, and someone will quote it. A failing run records `comparison = None`.

**3. Timing uses the canonical tier only** — a few aligned, well-sized shapes. Correctness is cheap and wants breadth; timing is expensive and wants stability.

**4. A record is written** to `~/.sns/runs/` as plain JSON.

## The reference is used two different ways

This looks inconsistent and is deliberate.

As the **correctness oracle**, your reference runs on CPU in float64. It is never the PyTorch GPU op, because comparing a GPU kernel against a GPU op means both sides share numerics, kernels and bugs — agreeing proves much less than it appears to.

As the **timing baseline**, the same reference runs on GPU. That is the thing your kernel is trying to beat.

## Reading a failure

```
FAIL  triton_add_drops_tail    5/16 shapes
      minimal failing case: 1025-contiguous-float32  seed=12648431
      replay: sns replay --shape 1025-contiguous-float32 --seed 12648431
```

The minimal case is the smallest shape that **already failed** — we do not search for smaller ones, so every reported case is one genuinely observed.

Note the shape: **1025**, one past a tile boundary. That is where masking bugs live, and it is why the generator produces such shapes rather than only powers of two. A test suite using 1024 finds nothing here.

## Shape tiers

| Tier | Size | Use |
|---|---|---|
| `FAST` | ~16 shapes/dtype | The default. Seconds. Built for an inner loop. |
| `EXHAUSTIVE` | ~61 shapes/dtype | Before merging. |
| `CANONICAL` | 3 shapes/dtype | Timing only, never correctness. |

`FAST` covers all five bug classes: tile boundaries, size-1 dimensions, primes, extreme aspect ratios, and non-contiguous layouts — plus combinations where two dimensions are awkward at once, which is where fused-kernel bugs hide.

## Tile awareness

If you pass a Triton kernel, `discover_tiles()` can read its configuration:

```python
from sns.tiles import discover_tiles

discover_tiles(my_kernel)
# plain @triton.jit  -> names=['BLOCK_M','BLOCK_N'], candidates={}
# @triton.autotune   -> candidates={'BLOCK_M': [32,64,128], ...}
```

A plain kernel reveals its constexpr *names*; an autotuned one reveals every candidate *value*.

That second case matters for correctness, not just shape selection. **Autotune picks a config per input shape, so the kernel you tested may not be the kernel that runs.** A config only selected for large inputs can be broken and never touched by small test shapes.

## Fused kernels

Two extras exist for kernels that stack several operations:

```python
sns.test(..., fused_ops=["layernorm", "matmul", "gelu"])
```

Error compounds through a chain, so a fused kernel's budget is wider than any single stage's. Pass `tolerance_override=(atol, rtol)` when you know better than the table.

Multi-output kernels are handled automatically — a shape passes only if **every** returned tensor passes. A bug in a saved `mean` or `rstd` for the backward pass is exactly the kind that survives to production.

## Check the tier before believing a speedup

```python
if not record.comparison.is_performance_valid:
    print("measurement too unstable for a verdict")
```

Tier C means the GPU was throttling or its clocks were moving too much for any performance claim to hold. The CLI shows `UNSTABLE` rather than a speedup in that case. See [Measurement tiers](tiers.md).
