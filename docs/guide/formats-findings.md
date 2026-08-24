# Findings

Measured results from [`shapesandstrides.formats`](formats.md). Every number here came from running the code; none is recalled or estimated. Where a result later turned out to be wrong, the wrong version is still recorded, because how it was wrong is the useful part.

All of it runs on CPU on a laptop.

---

## Summing in bf16 destroys the answer

Sum of 0.1, repeated, in bf16. Max relative error, under both quantization models:

| terms | `STORAGE_ONLY` | `EVERY_STEP` |
|---|---|---|
| 10 | 1.11e-16 | 7.81e-03 |
| 100 | 1.95e-15 | 6.25e-03 |
| 1000 | 1.41e-14 | **0.68** |
| 5000 | 9.04e-14 | **0.94** |

Summing 1,000 values in bf16 loses **68%** of the answer; 5,000 loses 94%.

This is accumulator stagnation. Once the running total is large enough, each new addend falls below its ULP and is thrown away — the accumulator simply stops growing. It is well known in numerical analysis and it is exactly why real kernels accumulate reductions in fp32 even when they store in bf16.

Two things worth drawing out.

**Storage-only would have told you it was fine.** It reported 1.4e-14 for the same 1,000-term sum, and barely moves with the length of the sum at all — because it rounds only at the edges, it structurally cannot see error compounding. Any tool that models a format by casting inputs and outputs is blind to the dominant error term in a reduction.

**It puts a number on an existing fixture.** `examples/kernels.py::triton_rowsum_bad_accum` is a deliberately broken kernel whose comment says it is "correct for small inputs and wrong for large ones." That was an assertion with no magnitude attached. Now it has one.

It also suggests `tolerance.py`'s `REDUCTION_SLACK = 10.0` is optimistic by orders of magnitude for a low-precision accumulator — a hand-picked constant that the kernel half of this project still relies on.

---

## The guess was blind exactly where it mattered

This one is a mistake, recorded on purpose.

`gradient_like()` is a synthetic generator: a log-uniform magnitude spread from 1e-11 to 1e-3. It was clearly labelled as an assumption. Using it, the bias sweep said that a 6-exponent-bit format at bias 31 loses **no** gradients — which appeared to contradict the inference that cbfloat16's bias is 31 (an inference that rests on Cerebras requiring loss scaling, which implies gradients *are* underflowing).

So the documentation was updated to say the inference was contradicted.

Then the gradients were actually recorded. A real 867k-parameter transformer, 400 steps, every gradient magnitude captured. At step 400, of 841,471 non-zero gradients:

| below | count | fraction |
|---|---|---|
| 1e-04 | 580,615 | 69% |
| 1e-08 | 615 | 0.073% |
| **1e-11** | **224** | **0.027%** |
| 1e-12 | 35 | 0.0042% |
| 1e-13 | 0 | 0% |

**`gradient_like` bottomed out at 1e-11.** It had *zero* values in the region that decides the answer. It was not merely imprecise; it was blind precisely where the question lived, because a log-uniform spread has no tail and the tail was the whole question.

The inference was right. The measurement that appeared to refute it was the thing that was wrong.

The lesson is not "label your assumptions" — it *was* labelled. The lesson is that a labelled assumption can still produce a confident, published, wrong conclusion, and only replacing it with a recording finds that out.

---

## What each format actually does to real gradients

Real recorded gradients, step 400. Underflow cliff is half the smallest subnormal — round-to-nearest pulls anything above that point up rather than down to zero. Counts are exact, not sampled.

| format | underflow cliff | gradients destroyed | fraction |
|---|---|---|---|
| fp16 | 2.98e-08 | **771** | 0.092% |
| 6/9 bias 28 | 7.28e-12 | 168 | 0.020% |
| **6/9 bias 31** (inferred cbfloat16) | 9.10e-13 | **27** | 0.0032% |
| 6/9 bias 36 | 2.84e-14 | **0** | 0% |

A 6/9 format at the conventional bias cuts fp16's silent gradient loss **29-fold** — and still does not eliminate it, which is consistent with Cerebras requiring loss scaling for cbfloat16.

Shifting the bias five more notches, to 36, eliminates it entirely on this distribution.

---

## A bias shift is loss scaling, for free

Loss scaling keeps small gradients alive by multiplying them into the representable window at runtime and dividing back out afterwards. It costs a multiply and a divide every step, a scale factor that must be tuned or adapted, and skipped steps when the scaled values overflow.

Shifting the exponent bias does the same job by moving the window instead of the values. It costs **nothing** — the bias is only what you subtract when reading an exponent, and subtracting 36 is the same adder as subtracting 31.

Measured across 14,005 values — real gradients, their extreme tail, and the boundary values at and around the ceiling — at shifts of 1, 3, 5, 8, 12 and −4:

**Zero differences. Bit-identical.**

Which stands to reason once stated: multiplying by a power of two is exact, and a bias-shifted format's grid *is* the base grid scaled. Even the overflow boundaries coincide — the shifted ceiling comes out at exactly `base_max / 2^N`.

### What this does not prove

The claim needs stating as narrowly as it deserves. A bias shift reproduces the **static** part of loss scaling exactly. It does not reproduce the adaptive part.

- **Loss scaling is usually dynamic**, tracking the gradient distribution as it moves — and it does move. In our own recording, the median gradient magnitude fell about sixfold between step 1 and step 400, and the fraction below fp16's normal floor rose from 12% to 53%. A bias is fixed when the format is designed.
- **Loss scaling also changes intermediate magnitudes**, through accumulations and optimiser state, not only what gets stored. This measures storage.

So: wherever you control the format and the distribution is stable, a bias shift is strictly better — identical numerics, none of the runtime machinery. Where the distribution moves, adaptivity is doing work a fixed bias cannot.

---

## Three things the bias sweep teaches

None of these were obvious in advance; two of them corrected tests that had been written the wrong way round.

**The underflow cliff sits at half the smallest subnormal** — two steps below where intuition puts it. Subnormals extend usable range below the normal floor by a factor of 2^mantissa_bits, and then round-to-nearest pulls anything above the halfway point *up* to the smallest subnormal rather than down to zero. At bias 28, an input of 1.07e-11 survives despite sitting below both the normal floor (7.45e-09) and the smallest subnormal (1.46e-11).

**Too low a bias costs you twice.** Values fall off the bottom, *and* the survivors get crowded into the subnormal range where fewer significant bits remain. Bias 12's median error is seven times bias 16's.

**The optimum is not "as high as possible."** Raising the bias buys room at the bottom by surrendering the ceiling. Once a setting loses nothing, going higher buys nothing and costs headroom — so the best bias is the *lowest* lossless one.

---

## The simulator itself

None of the above would mean anything if the simulator were wrong. It is checked against dtypes that really exist, bit for bit, across every value class:

| format | compared | matched | mismatched | grade |
|---|---|---|---|---|
| `binary16` vs `torch.float16` | 84 | 84 | 0 | **A** |
| `bfloat16` vs `torch.bfloat16` | 84 | 84 | 0 | **A** |
| `binary32` vs `torch.float32` | 84 | 84 | 0 | **A** |

And the check is demonstrably capable of failing, which an always-green check is not: fp16's widths with bias 14 instead of 15 produces 75 mismatches, grades C, and reports its first divergence at `smallest_subnormal`, `0x1` against `0x2`.

Every result about a format with **no** native counterpart — which includes any reconstructed cbfloat16 — is [grade B](formats.md#the-grade), and stays there until somebody runs it on the vendor's hardware.

---

## Reproducing all of it

```bash
pip install -e .
python -m pytest tests/formats -q          # 111 tests, no GPU needed
python scripts/record_gradients.py         # regenerate the gradient fixture
```

The gradient fixture is committed, so every number on this page reproduces without a GPU and without retraining.
