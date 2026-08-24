# Reconstructing cbfloat16

Cerebras's **cbfloat16** is the format that motivated [the formats subpackage](formats.md). It is a good example precisely because it is *incompletely specified in public*.

!!! warning "This package ships no `CBFLOAT16` constant, and that is deliberate"
    `from shapesandstrides.formats import CBFLOAT16` would carry this project's authority for a specification we partly guessed. The name makes that claim before any grade label is read — and someone who imports it, measures, and publishes has produced a wrong result under a vendor's name, sourced from us.

    So you build it yourself, with your assumption stated. This page shows how.

## What Cerebras publishes

From their own documentation:

- **6 exponent bits, 9 mantissa bits**, 16 bits total.
- Roughly **double the dynamic range of fp16**, and two more significand bits than bf16.
- Used for **matrix multiplication and attention**, and for some reductions.
- The default for large language models at `precision_opt_level: 1`.
- **Loss scaling is mandatory** — their stack raises an error if `fp16_type: cbfloat16` is selected without it.
- **No Python or PyTorch support.** `float16` is used as a "proxy type" and conversion happens inside their compiler.

## What Cerebras does not publish

Nowhere we could find:

- **the exponent bias**
- subnormal handling
- infinity and NaN encoding
- the rounding mode
- maximum and minimum representable values

That is five unknowns against three published facts. Which is the whole reason this tool exists.

## What can be inferred, and how

Two pieces of reasoning, recorded as reasoning rather than asserted as fact.

**"Double the range of fp16" tells you nothing about the bias.** It follows automatically from the sixth exponent bit — with 6 bits you get 62 usable exponent codes instead of fp16's 30, whatever the bias happens to be. The claim constrains the *split*, not the *window position*.

**The loss-scaling requirement is the only real clue, and it holds up.**

The argument goes: loss scaling exists to lift tiny gradients into the representable range, so requiring it is evidence the window sits high, which points at the IEEE-conventional **31** rather than something shifted low.

We tested that argument rather than accepting it, and it survived — but only on the second attempt.

!!! note "We measured this twice and got opposite answers. The second one stands."
    **First attempt, with synthetic gradients: the inference looked wrong.** A log-uniform magnitude spread showed nothing underflowing at bias 31, whose cliff sits near 9e-13. If nothing underflows, loss scaling would be unnecessary, so requiring it would imply the bias is *not* 31.

    **Second attempt, with recorded gradients: the inference holds.** Real gradients from a real backward pass reach down to 1.14e-13, and **27 of 841,471 fall below the bias-31 cliff.** Small, but not zero — so a stack using that format would still need loss scaling, which is precisely what Cerebras requires.

    The synthetic generator bottomed out at 1e-11 and had *zero* values in the region that decides the question. The measurement that appeared to refute the inference was itself the thing that was wrong. Full account in [Findings](formats-findings.md#the-guess-was-blind-exactly-where-it-mattered).

So the loss-scaling argument survives contact with real data, and **31 remains the best available inference.** It is still inference from observable behaviour rather than proof, and this page does not present it as more.

## Building it

```python
from shapesandstrides.formats import FormatSpec, Provenance, ieee_bias

CB_RECONSTRUCTED = FormatSpec(
    name="cbfloat16-reconstructed",
    exponent_bits=6,
    mantissa_bits=9,
    bias=ieee_bias(6),          # 31 — inferred, see the reasoning above
    provenance={
        "exponent_bits": Provenance.DOCUMENTED,   # Cerebras docs
        "mantissa_bits": Provenance.DOCUMENTED,   # Cerebras docs
        "bias": Provenance.INFERRED,              # loss-scaling requirement,
                                                  # backed by recorded gradients
        "has_subnormals": Provenance.ASSUMED,     # IEEE convention
        "has_infinities": Provenance.ASSUMED,     # IEEE convention
    },
    notes=(
        "Widths from Cerebras documentation. Bias inferred from their "
        "loss-scaling requirement: real recorded gradients do fall below the "
        "bias-31 underflow cliff (27 of 841,471), so that requirement is "
        "consistent with bias 31. Not published, and not proof. Subnormal and "
        "infinity policy assumed to follow IEEE convention."
    ),
)
```

Derived limits, measured:

| | value |
|---|---|
| smallest subnormal | 1.8189894035458565e-12 |
| smallest normal | 9.313225746154785e-10 |
| largest normal | 4,290,772,992 |
| eps (spacing at 1.0) | 0.001953125 (= 2⁻⁹) |

Both published claims check out from these numbers: the range spans roughly double fp16's in log terms, and there are two more significand bits than bf16.

## Any result from this is grade B

```python
from shapesandstrides.formats import validate

validate(CB_RECONSTRUCTED).tier   # FormatTier.B
```

There is no native cbfloat16 to compare against, so the grade cannot be A. **It stays B until somebody runs it on Cerebras hardware.**

That is not a weakness of the measurement — the simulator itself is [validated to grade A against fp16, bf16 and fp32](formats.md#validate-is-the-simulator-telling-the-truth), so it is known to be faithful where faithfulness is checkable. The B is about cbfloat16 specifically: we have no way to confirm the *parameters*, only the arithmetic.

## Why the bias is the interesting parameter

The bias costs nothing in hardware. The chip stores the exponent as a plain unsigned number, and subtracting 31 is the same adder as subtracting 40. It is a free dial that nobody tunes.

Meanwhile the industry's remedy for gradients falling off the bottom of the window is **loss scaling**: multiply the loss by a constant, divide back out afterwards, and manage that constant every step of training.

Look at what that is — a runtime workaround for a badly positioned exponent window. Shifting the bias moves the window directly, once, for free.

So there is an unanswered question here: **could a shifted bias remove the need for loss scaling entirely?**

`sweep` runs that experiment. Measured on **real recorded gradients** — 841,471 of them, from a real transformer backward pass — for a 6-exponent/9-mantissa format:

| bias | underflow cliff | gradients silently zeroed |
|---|---|---|
| fp16, for comparison | 2.98e-08 | 771 (0.092%) |
| 28 | 7.28e-12 | 168 (0.020%) |
| **31** (inferred cbfloat16) | 9.10e-13 | **27 (0.0032%)** |
| **36** | 2.84e-14 | **0** |

Two results.

**cbfloat16 at bias 31 cuts fp16's silent gradient loss 29-fold, and still does not eliminate it.** That is consistent with Cerebras requiring loss scaling, and it is what rescued the inference above.

**Bias 36 eliminates it entirely — and reaching bias 36 costs nothing.** The bias is only what you subtract when reading an exponent; subtracting 36 is the same adder as subtracting 31. No silicon, no runtime, no scale factor.

And a bias shift is not merely *similar* to loss scaling. Measured across 14,005 values and six shift sizes, it is **bit-identical** to it — see [Findings](formats-findings.md#a-bias-shift-is-loss-scaling-for-free), including the two things that result does *not* prove. Chiefly: loss scaling is usually dynamic, and a fixed bias cannot adapt as the distribution moves — which, in our recording, it does, the median magnitude falling roughly sixfold over 400 steps.

So the open question narrows usefully. Not "would a shifted bias help" — it demonstrably would, for free, on this distribution. But "would a *fixed* bias be enough, given the distribution moves during training?" That one is still open, and it is the interesting one.

## One thing in the documentation that does not add up

Cerebras states cbfloat16 is approximately **19% faster than bf16** at similar accuracy.

But cbfloat16 has *more* mantissa bits than bf16 — 9 against 7 — and multiplier cost grows roughly with the square of significand width. On that basis each multiply should cost more, not less:

| format | significand bits | rough multiplier cost |
|---|---|---|
| bf16 | 8 | 64 |
| cbfloat16 | 10 | 100 |
| fp16 | 11 | 121 |

No explanation was found in their documentation. It may be comparing whole precision configurations rather than per-operation cost, or whole training runs rather than throughput.

Recorded as an open question, not a criticism. It is also a fair illustration of the gap: a vendor states a number and nobody outside can check it.

## Sources

- [Cerebras — precision optimization level and data formats](https://training-api.cerebras.ai/en/rel-2.3.1/wsc/how_to_guides/cs-1-data-formats.html)
- [Cerebras — data formats (1.7.0)](https://training-api.cerebras.ai/en/1.7.0/general/cs-1-data-formats.html)
- [Cerebras — To Bfloat or not to Bfloat?](https://www.cerebras.ai/blog/to-bfloat-or-not-to-bfloat-that-is-the-question)
