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

## The four verbs

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

## What this does not do yet

Honest scope. Built and tested:

- `FormatSpec` with provenance, `values_for`, `round_trip`, `validate`, the grade.

Designed but **not built**:

- `error_over` — error distributions with confidence intervals over a real calculation (dot product, softmax, layernorm).
- `sweep` — vary the bias and measure how many values are silently lost at each setting. The interesting experiment, and the reason for all of the above.
- A `@check_format` decorator and a CLI.

## Credit

[`gfloat`](https://github.com/graphcore-research/gfloat) (MIT, Graphcore Research) is the numeric core: encode, decode and round for arbitrary formats, with the exponent bias as a first-class parameter. It is **not** reimplemented here.

What this subpackage adds is the harness `gfloat` does not have: provenance, corner-case value generation, validation against native dtypes, and a grade on every result.
