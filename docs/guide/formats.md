# Numeric formats

Declare a float format by its parameters — including formats no vendor has fully published — and find out what it does to real numbers.

Runs on CPU. Needs no GPU and no vendor hardware.

## The one example that explains why this exists

A network is training. A gradient comes out as `1e-8`, which is entirely ordinary for a gradient.

```python
from shapesandstrides.formats import FLOAT16, BFLOAT16, FormatSpec, ieee_bias, round_trip

cb = FormatSpec(name="cb-6-9", exponent_bits=6, mantissa_bits=9, bias=ieee_bias(6))

for o in round_trip(1e-8, into=[FLOAT16, cb, BFLOAT16]).outcomes:
    print(o.format_name, o.result, o.outcome.value, o.rel_error)
```

Measured output:

| format | `1e-8` becomes | outcome | relative error |
|---|---|---|---|
| fp16 | `0.0` | **underflow** | total loss |
| 6-exp/9-mant, bias 31 | `9.997166e-09` | rounded | 0.028% |
| bf16 | `1.001172e-08` | rounded | 0.117% |

fp16 did not raise. It did not warn. It quietly turned the gradient into zero, and that weight stops learning.

Overflow announces itself — an infinity is hard to miss. Underflow does not. That is why `UNDERFLOW` is its own outcome here rather than being reported as a large relative error.

## How a float format works

Three fields: **sign**, **exponent**, **mantissa**.

```
value = ±(1.mantissa) × 2^(exponent − bias)
```

- **Mantissa bits → precision.** How finely you can distinguish nearby numbers.
- **Exponent bits → range.** How large and how small you can go.
- **Bias → where that range sits on the number line.** With 6 exponent bits there are 64 exponent codes; the bias decides which code means 2⁰.

The bias is a *reading convention*, not a hardware feature. The chip stores the exponent as a plain unsigned number; the bias is only what you subtract to interpret it. Subtracting 31 and subtracting 40 are the same adder, so **the bias is free** — which is exactly why it is worth testing.

Mantissa bits are not free. A multiplier's cost grows roughly with the square of its width, which is why bf16's 8 significand bits became popular in hardware and why formats trade mantissa for exponent.

## The core verbs

Four to describe and check a format; two more below, after the grade, to measure one.

### `FormatSpec` — declare a format

```python
from shapesandstrides.formats import FormatSpec, Provenance, ieee_bias

fmt = FormatSpec(
    name="my-format",
    exponent_bits=6,
    mantissa_bits=9,
    bias=ieee_bias(6),          # 31
    provenance={
        "exponent_bits": Provenance.DOCUMENTED,
        "mantissa_bits": Provenance.DOCUMENTED,
        "bias": Provenance.ASSUMED,
    },
    notes="widths from the vendor's docs; bias is my assumption",
)

fmt.smallest_normal      # 9.313225746154785e-10
fmt.smallest_subnormal   # 1.8189894035458565e-12
fmt.max_value            # 4290772992.0
fmt.eps                  # 0.001953125
```

!!! warning "`bias` has no default, on purpose"
    Two of the four numbers defining a 16-bit float are routinely unpublished. A default for `bias` is how you silently inherit somebody else's guess, so construction fails without it.

    `ieee_bias(6)` returns the conventional 31, so the standard choice costs one call. You still typed it — so it is still *your* stated assumption — but you did not have to work it out.

A parameter with no stated source reports as `UNSTATED`, never as documented:

```python
fmt.provenance_of("bias")            # Provenance.ASSUMED
fmt.provenance_of("has_subnormals")  # Provenance.UNSTATED
```

**Pre-built constants** exist for formats whose every parameter is published, each citing its standard: `FLOAT64`, `FLOAT32`, `FLOAT16`, `BFLOAT16`, `FLOAT8_E4M3`, `FLOAT8_E5M2`. Nobody should compute bf16's bias by hand.

There is deliberately **no** constant for any format with an unknown parameter. See [Reconstructing cbfloat16](formats-cbfloat16.md) for why, and for how to build one yourself.

### `values_for` — the values that actually break formats

```python
from shapesandstrides.formats import FLOAT16, values_for, ValueClass

vs = values_for(FLOAT16)                 # every class
vs.of_class(ValueClass.SMALLEST_SUBNORMAL)   # [5.960464477539063e-08]
```

Formats do not break on 3.7 and 0.5. Those always work, which is why a suite built from them always passes and proves nothing.

| class | what it catches |
|---|---|
| `SMALLEST_SUBNORMAL`, `LARGEST_SUBNORMAL` | the fuzzy region below the normal floor |
| `SMALLEST_NORMAL`, `LARGEST_NORMAL` | the floor and the ceiling exactly |
| `JUST_OVER_MAX` | must overflow, not round back down to the ceiling |
| `JUST_UNDER_MIN` | must underflow to zero — the silent failure |
| `TIE` | **the important one.** Values exactly halfway between two representable numbers. The only place two rounding modes can disagree, so a suite without ties cannot tell ties-to-even from ties-to-away. |
| `POWER_OF_TWO` | must be exact; a rounding bug here is unmissable |
| `ZERO`, `NEGATIVE_ZERO` | two encodings of the same value |
| `NAN`, `POSITIVE_INFINITY`, `NEGATIVE_INFINITY` | the specials |
| `ORDINARY` | a control group that should simply work |

Every value is labelled with the class that produced it, so a failure names the *kind* of number responsible rather than only a magnitude. Everything is derived from a seed, and the seed is recorded in the output.

An unrecognised class raises rather than being skipped — a class that quietly did not run is a test that quietly did not run.

### `round_trip` — what happens to my number

```python
round_trip(1e-8, into=[FLOAT16, BFLOAT16])
```

Accepts a bare float or a sequence, and any number of formats at once. Each outcome is one of `EXACT`, `ROUNDED`, `OVERFLOW`, `UNDERFLOW` or `BECAME_NAN`.

An infinity that arrived as an infinity is `EXACT`, not `OVERFLOW` — overflow means a finite number stopped being finite.

### `validate` — is the simulator telling the truth?

This is the one that makes the rest worth reading.

Formats here are simulated in software. So how do you know the simulation is right? You cannot check a reconstructed cbfloat16 against real cbfloat16 — that needs a Cerebras machine. But you *can* check the simulator against formats that really exist.

```python
from shapesandstrides.formats import FLOAT16, validate

r = validate(FLOAT16)
r.compared, r.matched, r.mismatched, r.tier   # 84, 84, 0, FormatTier.A
```

Measured, comparing raw bit patterns against torch's own dtypes across every value class:

| format | compared | matched | mismatched | grade |
|---|---|---|---|---|
| `binary16` vs `torch.float16` | 84 | 84 | 0 | **A** |
| `bfloat16` vs `torch.bfloat16` | 84 | 84 | 0 | **A** |
| `binary32` vs `torch.float32` | 84 | 84 | 0 | **A** |

The gate is also demonstrably capable of failing, which an always-green gate is not. fp16's widths with bias 14 instead of 15 produces 75 mismatches, grades C, and reports its first divergence at `smallest_subnormal`, `0x1` against `0x2`.

## The grade

Every result carries one, and it is a required field with no default.

| grade | meaning |
|---|---|
| **A** | this exact format was validated bit-for-bit against a native implementation. The simulator was checked, not trusted. |
| **B** | no native counterpart exists, so results rest on the simulator. The normal case for any reconstructed format, **and the default** — so forgetting to validate can only ever under-claim. |
| **C** | validation ran and failed. No numeric claim from this format is valid. |

As with [measurement and oracle tiers](tiers.md), **C is the absence of a verdict, not a failure.** `is_format_valid` is false only at C.

This separation does real work. In the `1e-8` table at the top, the fp16 underflow carries **grade A** — it is validated against torch's actual behaviour, not a simulator artifact. Only the reconstructed format's row is grade B.

### `error_over` — what a format costs a calculation

```python
from shapesandstrides.formats import FLOAT16, error_over, gradient_like
from shapesandstrides.formats.ops import softmax

d = error_over(softmax, [gradient_like(400, seed=1)], FLOAT16)
d.p50_rel_error, d.p90_rel_error, d.max_rel_error   # a distribution
d.ci95_lo, d.ci95_hi                                # with an interval
d.input_loss.underflowed                            # and a loss census
```

Measured, softmax over 400 gradient-magnitude values:

| format | p50 rel error | p90 | max | underflowed on input |
|---|---|---|---|---|
| fp32 | 3.06e-08 | 3.98e-08 | 4.65e-08 | 0 / 400 |
| fp16 | 2.18e-04 | 2.42e-04 | 3.72e-04 | **174 / 400** |
| 6/9 bias 31 | 5.45e-04 | 5.58e-04 | 7.54e-04 | 0 / 400 |
| bf16 | 9.81e-04 | 1.01e-03 | 1.85e-03 | 0 / 400 |

Read that table carefully. **fp16 has the lowest error of the three 16-bit formats — because 43% of the gradients underflowed to zero before the arithmetic began.** The survivors genuinely are more precise; the data is gone. That is why the loss census sits beside the distribution and is never folded into it.

!!! warning "This models storage precision, not arithmetic"
    Inputs are rounded into the format, the calculation runs exactly in double precision, and the result is rounded again. **Real hardware rounds every intermediate, so these error figures are a lower bound.**

    Every result records `quantization_model`, which currently has exactly one value, `STORAGE_ONLY`. There is deliberately no `EVERY_STEP` member, because no code path can produce one yet — promising one in the JSON would be a lie.

    The loss census is *not* a lower bound. Values are destroyed at storage time, so those counts are exact.

### `sweep` — vary a parameter and read the curve

The reason for everything above.

```python
from shapesandstrides.formats import FormatSpec, ieee_bias, sweep, gradient_like
from shapesandstrides.formats.ops import total

cb = FormatSpec(name="cb-6-9", exponent_bits=6, mantissa_bits=9, bias=ieee_bias(6))
r = sweep(cb, "bias", range(12, 44, 4), total, [gradient_like(300, seed=1)])
r.best_by_silent_loss
```

Measured:

| bias | normal floor | subnormal floor | values lost | p50 rel error |
|---|---|---|---|---|
| 12 | 4.88e-04 | 9.54e-07 | 174 | 2.22e-03 |
| 16 | 3.05e-05 | 5.96e-08 | 131 | 3.22e-04 |
| 20 | 1.91e-06 | 3.73e-09 | 87 | 3.22e-04 |
| 24 | 1.19e-07 | 2.33e-10 | 38 | 3.22e-04 |
| **28** | 7.45e-09 | 1.46e-11 | **0** | 3.22e-04 |
| 40 | 1.82e-12 | 3.55e-15 | 0 | 3.22e-04 |

Three things that curve teaches, none of which were obvious in advance:

**The underflow cliff is at half the smallest subnormal** — two steps below where intuition puts it. Subnormals extend usable range below the normal floor by a factor of 2^mantissa_bits, and then round-to-nearest pulls anything above the halfway point *up* to the smallest subnormal rather than down to zero. At bias 28 an input of 1.07e-11 survives despite being below both the normal floor and the smallest subnormal.

**Too low a bias costs you twice.** Values fall off the bottom, *and* the survivors get crowded into the subnormal range where fewer significant bits remain. Bias 12's median error is seven times bias 16's.

**There is a real optimum, and it is not "as high as possible."** Raising the bias buys room at the bottom by giving up the ceiling. Once a setting loses nothing, going higher buys nothing and costs headroom — so `best_by_silent_loss` picks the *lowest* lossless setting, 28 here, rather than the highest.

An impossible setting raises rather than being skipped: a curve that quietly omitted the points that did not work would misrepresent what was actually tested.

## What this does not do yet

Honest scope. Built and tested: `FormatSpec` with provenance, `values_for`, `round_trip`, `validate`, the grade, `error_over`, `sweep`.

Not built:

- **Step-wise rounding** — the `EVERY_STEP` quantization model. Until it exists, error figures are lower bounds.
- **A loss-scaling equivalence test** — does shifting the bias by N do the same job as scaling by 2^N, and where do the two diverge? The bias sweep makes this askable; nobody has asked it yet.
- **Gradient magnitudes from a real training run.** `gradient_like` is a stated assumption, not a measurement — see below.
- A `@check_format` decorator, a CLI, and stored run records.

!!! warning "`gradient_like` is an assumption about the world"
    It is a log-uniform magnitude spread from 1e-11 to 1e-3 with random signs, chosen because that band straddles fp16's floor. **No model was trained to obtain it.** Treat any conclusion resting on it exactly as you would treat a format parameter marked `ASSUMED`.

    Replacing it with magnitudes recorded from a real training run would strengthen every result above, and is outstanding work.

## Credit

[`gfloat`](https://github.com/graphcore-research/gfloat) (MIT, Graphcore Research) is the numeric core: encode, decode and round for arbitrary formats, with the exponent bias as a first-class parameter. It is **not** reimplemented here.

What this subpackage adds is the harness `gfloat` does not have: provenance, corner-case value generation, validation against native dtypes, and a grade on every result.
