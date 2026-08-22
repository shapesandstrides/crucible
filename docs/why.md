# Why this exists

Every design decision here traces to a documented failure. This page is the evidence.

## The benchmarking tool everyone uses is 30% wrong by default

[triton#2306](https://github.com/openai/triton/issues/2306): `triton.testing.do_bench` defaults to `warmup=25`, which on a typical kernel amounts to **two actual function calls**. Reliable results need roughly fifteen.

The measured gap: **11.23 ms reported against a true 8.79 ms** — a 30% underestimate.

Filed September 2023. No maintainer response.

This is the number in most Triton benchmark posts, PR descriptions, and blog comparisons you have read. It's why `measure()` warms up 200 times and why lowering that default is not a tuning knob.

## The masking bug is not an edge case

[triton#737](https://github.com/openai/triton/issues/737): omit the `other=` argument to `tl.load` and masked lanes return **whatever bytes happen to be at that address**. In fp16 the first two columns come back as garbage; fp32 usually survives.

How often does it fire? *"Every time your data size isn't perfectly divisible by your block size — which is almost always."* A tensor of 4097 elements with `BLOCK_SIZE=1024` hits it on the last of five tiles.

GPUs perform no bounds checking. `tl.load` reads the address you gave it.

This is why correctness testing has to include shapes like `4097×4096`, and why testing on powers of two proves almost nothing.

## Version upgrades break kernels, with numbers

| What happened | Source |
|---|---|
| Triton 3.6 swapped reduction kernels — **6.4× slower kernel**, ~20% end-to-end latency regression | [vllm#37441](https://github.com/vllm-project/vllm/issues/37441) |
| **5–10× regression** from Triton 2.x → 3.6; old-style pointer math stopped being optimized, loads became uncoalesced | [triton#9640](https://github.com/triton-lang/triton/issues/9640) |
| Nightly Triton produced *incorrect* fused-attention output | [triton#4310](https://github.com/triton-lang/triton/issues/4310) |
| Triton 3.0.0 produced wrong reduction results | [triton#4379](https://github.com/triton-lang/triton/issues/4379) |
| Triton 3.6 import failure caused a silent fallback to a much slower path | [vllm#39664](https://github.com/vllm-project/vllm/issues/39664) |

Note that these are not all performance regressions. Some are *correctness* regressions introduced by a compiler upgrade, in kernels whose source never changed.

## Compiler bugs can be architecture-specific

[pytorch#176426](https://github.com/pytorch/pytorch/issues/176426): Triton kernels containing two or more `tl.load()` calls **segfault on sm_120** (RTX PRO 6000 Blackwell). They compile without errors and emit invalid code.

A kernel that is correct on your Ampere dev box can be broken on Blackwell through no fault of your own. This is why architecture is tracked as a first-class property, and why `sm_100` and `sm_120` are [reported separately](guide/hosts.md).

## Existing correctness benchmarks give false confidence

["The Correctness Illusion in LLM-Generated GPU Kernels"](https://arxiv.org/pdf/2606.20128) examined how KernelBench, TritonBench and GEAK check correctness: **fixed-shape, small-sample `allclose`**.

Re-evaluating a controlled corpus with opschema-aware seeded fuzzing against a high-precision CPU reference, the authors' oracle caught **9 of 9 buggy kernels and passed 15 of 15 correct controls** — bugs the existing suites had cleared.

The paper also documents a reward-hacking mode where **a kernel is emitted but never actually executed in the entry function**, producing a timing measurement of nothing at all.

That is why the harness computes timing and pass/fail from its own observation, and why submitted code never reports its own result.

## Locking clocks is necessary and not sufficient

[Standard Kernel's benchmarking study](https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/) found that **power constraints override manual clock settings** — a kernel drawing enough power throttles regardless of the lock, silently. They also found execution times under 10 µs cannot be reliably measured, and that identical instance types differ across cloud providers by SKU, driver, and power configuration.

Hence the post-hoc throttle assertion, the minimum-duration guard, and the advice to pin one provider and one instance type.

## Things we found ourselves

Two came out of building this, on real hardware.

**Throttle flags are not a sufficient validity signal.** Under 30 seconds of sustained load an RTX 3060 laptop swung **495 MHz — 5.1% CV** — with every throttle flag inactive, and two identical runs disagreed about whether throttling fired. Observed clock variance had to become the governing signal.

**Subprocess telemetry destroys what it measures.** Sampling the clock via `nvidia-smi` costs **~68 ms per call**. Sampling inside a measurement loop stalled the host long enough that the GPU went idle between iterations, so every reading reflected an idle clock — variance came back `0.0` always, silently disabling both tier gates. NVML in-process is microseconds.

---

## The bar we are clearing

Put plainly: the incumbent way to evaluate a Triton kernel is `do_bench` — documented to be 30% wrong — plus `torch.allclose` on three shapes, all of them powers of two.

That is a low bar, and it is the one nearly every published Triton speedup number was measured against.
