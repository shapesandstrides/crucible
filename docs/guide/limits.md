# What this cannot catch

A tool that tells you what it misses is more useful than one that implies it misses nothing. These limits were found by building deliberately broken kernels and discovering that two of them passed.

## Memory-safety bugs

The oracle compares **output values**. A kernel that reads or writes out of bounds but still produces correct values in the compared region passes clean.

A concrete case from our own fixtures: a kernel with `mask = offs <= n` instead of `offs < n` writes one element past the end of its output buffer. That is a genuine memory-safety bug. It was undetectable here, because the write landed in allocator slack and every element we compared was correct.

**Use `compute-sanitizer` for this class.** It is what it exists for, and no value-comparison oracle can substitute:

```bash
compute-sanitizer --tool memcheck python your_kernel_test.py
```

## Undefined values that happen to be benign

Omitting `other=` on a masked `tl.load` leaves masked lanes undefined. This is a real bug — [triton#737](https://github.com/openai/triton/issues/737) — but whether it *manifests* depends on what happens to be in memory and on what the kernel does next.

We could not make it fail reliably. In an elementwise kernel the store is masked identically, so the garbage never reaches the output. Even feeding those lanes into a reduction read clean, because Triton appears to zero-fill masked lanes in practice. The original report describes the corruption as intermittent and fp16-specific, which matches.

So: a passing run does not prove the absence of undefined reads. It proves they did not change the answer *this time, on this hardware, with this memory state*.

## Anything that needs elevated privileges

By design, crucible collects only what works without reconfiguring your machine. That excludes:

- L1 and L2 cache hit rates
- DRAM traffic and memory-throughput counters
- Achieved occupancy
- Warp-level stall reasons

All of these need GPU performance counters, which are admin-restricted ([`ERR_NVGPUCTRPERM`](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)). Use Nsight Compute directly when you need them.

We *do* collect memory allocation peaks, clock and thermal state, throttle reasons, and which CUDA kernels actually ran — all unprivileged. See [Measuring](measuring.md).

## Race conditions and nondeterminism

Each shape is tested with one seed and one execution. A kernel with a race that fires occasionally will pass most runs. Nothing here does repeated execution looking for divergence, and shrinking assumes a failure reproduces deterministically.

## Numerical edge cases we do not generate

Inputs are drawn from a standard normal distribution. That means these are **not** exercised unless you supply them yourself:

- Denormals, infinities, NaN inputs
- Extreme magnitudes where fp16 overflows
- Adversarial values chosen to maximise cancellation

A kernel can be correct on well-conditioned random data and wrong on the distribution your model actually produces.

!!! note "Half-fixed"
    `shapesandstrides.formats.values_for` now generates exactly these classes -- subnormals, both zeros, one step past the ceiling, below the underflow floor, ties, NaN and the infinities -- for any format. See [Numeric formats](formats.md).

    It is **not yet wired into shape generation**, so `check()` still draws standard-normal inputs and the gap above remains real for kernel testing. Connecting the two is outstanding work.

## Performance claims below the noise floor

On hardware with floating clocks, comparing identical work still shows around 1–2% variation. A speedup claim below that is not distinguishable from noise, regardless of what the interval says.

Interleaved measurement reduced this from a p90 error above 100% to roughly 1%, but it does not reach zero. Read the interval, and treat sub-2% differences on unlocked hardware as unproven.

## The honest summary

This tool is good at finding **wrong values at awkward shapes** — dropped tails, bad accumulator dtypes, unhandled strides, boundary errors. That is the largest category of Triton bug and the one that existing three-shape `allclose` checks miss.

It is not a memory checker, not a race detector, and not a profiler.
