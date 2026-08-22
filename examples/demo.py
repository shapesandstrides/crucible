"""End-to-end demo: test real Triton kernels, correct and broken.

    python examples/demo.py

Runs the full pipeline against the kernels in `kernels.py` — three that are
correct and three that are deliberately broken in ways Triton kernels break
in the wild. Every result below is measured, not scripted.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import kernels as K  # noqa: E402

import sns  # noqa: E402
from sns.shapes import ShapeTier  # noqa: E402
from sns.tiles import discover_tiles  # noqa: E402

STORE = Path(__file__).parent.parent / ".demo-runs"

CASES = [
    ("triton_add", K.triton_add, K.ref_add, "correct"),
    ("triton_mul", K.triton_mul, K.ref_mul, "correct"),
    ("triton_add_autotuned", K.triton_add_autotuned, K.ref_add, "correct"),
    ("triton_add_drops_tail", K.triton_add_drops_tail, K.ref_add, "BROKEN"),
    ("triton_add_assumes_contiguous", K.triton_add_assumes_contiguous, K.ref_add, "BROKEN"),
]


def rule(title):
    print(f"\n\033[1m{title}\033[0m")
    print("-" * len(title))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this demo needs a CUDA device")

    rule("1. What can we discover about these kernels?")
    for label, inner in (
        ("plain @triton.jit", K._add_kernel),
        ("@triton.autotune", K._autotuned_add_kernel),
    ):
        ts = discover_tiles(inner)
        if ts is None:
            print(f"  {label:20} no tile info")
        else:
            print(f"  {label:20} source={ts.source:10} names={ts.names} candidates={ts.candidates}")
    print("\n  An autotuned kernel exposes every config it might pick. A config only")
    print("  selected for large inputs can be broken and never touched by small tests.")

    rule("2. Testing each kernel")
    results = []
    for name, fn, ref, expectation in CASES:
        rec = sns.test(
            fn,
            reference=ref,
            kernel_name=name,
            op_name="add",
            tier=ShapeTier.FAST,
            dtypes=["float32"],
            root=STORE,
            time_it=False,
        )
        c = rec.correctness
        ok = c.total - c.failed_count
        verdict = "PASS" if c.passed else "FAIL"
        colour = "\033[32m" if (c.passed == (expectation == "correct")) else "\033[31m"
        print(f"  {colour}{verdict}\033[0m  {name:32} {ok:2}/{c.total} shapes   (expected {expectation})")
        if c.minimal_failure:
            m = c.minimal_failure
            print(f"          minimal failing case: {m.spec.label}  seed={m.seed}")
            print(f"          replay: {c.replay_command}")
        results.append((name, expectation, c.passed))

    rule("3. Did the tool get it right?")
    wrong = [n for n, exp, passed in results if passed != (exp == "correct")]
    if wrong:
        print(f"  \033[31mMISCLASSIFIED: {wrong}\033[0m")
    else:
        print("  Every correct kernel passed and every broken kernel failed.")

    rule("4. Timing a correct kernel against PyTorch")
    rec = sns.test(
        K.triton_add,
        reference=K.ref_add,
        kernel_name="triton_add",
        op_name="add",
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        root=STORE,
        time_it=True,
        warmup=50,
        iters=30,
    )
    if rec.comparison:
        c = rec.comparison
        print(f"  candidate {c.candidate.median_ms:.4f} ms   baseline {c.baseline.median_ms:.4f} ms")
        print(f"  speedup   {c.speedup:.3f}x   95% CI [{c.speedup_ci_lo:.3f}, {c.speedup_ci_hi:.3f}]")
        print(f"  tier      {c.tier.value}")
        if not c.is_performance_valid:
            print(f"\n  \033[33mTier {c.tier.value}: this measurement was too unstable to")
            print("  support a performance verdict. The numbers above are recorded,")
            print("  not endorsed.\033[0m")
        print("\n  The baseline is re-measured in the same interleaved window, never cached.")
        print("  That is what separates 'my kernel regressed' from 'torch got faster'.")

    rule("5. What got recorded")
    print(f"  device   {rec.device.gpu_name}  sm_{(rec.device.compute_capability or '').replace('.','')}"
          f"  {rec.device.sm_count} SMs")
    print(f"  toolchain torch {rec.device.torch_version}  cuda {rec.device.cuda_version}"
          f"  triton {rec.device.triton_version}")
    if rec.context.sm_clock_mhz:
        print(f"  context  {rec.context.sm_clock_mhz:.0f}/{rec.context.max_sm_clock_mhz:.0f} MHz"
              f"  {rec.context.temperature_c:.0f}C  {rec.context.power_draw_w:.1f}W")
    if rec.memory.peak_allocated_bytes:
        print(f"  memory   peak {rec.memory.peak_allocated_bytes / 1024**2:.1f} MB")
    if rec.dispatch.kernels:
        k = rec.dispatch.kernels[0]
        print(f"  dispatch torch ran {k.name[:52]} ({k.device_time_us:.1f} us)")

    rule("6. The catalog")
    print(f"  {len(sns.list_runs(root=STORE))} runs stored in {STORE}")
    print("\n  Browse them with:")
    print(f"    sns runs --root {STORE}")
    print(f"    sns show <run-id> --root {STORE}")


if __name__ == "__main__":
    main()
