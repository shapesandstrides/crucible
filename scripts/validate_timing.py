#!/usr/bin/env python3
"""Phase 0 acceptance: does the harness report intervals it actually honours?

Times one identical kernel 50 separate times over at least 20 minutes. The
spread across runs must sit inside the confidence intervals each run
reports. If it does not, the CIs are lying and nothing downstream can be
trusted.

This cannot pass on unlocked consumer hardware, and that is correct
behaviour rather than a bug to engineer around. It needs a Tier A host:
Linux, root, datacenter GPU.

    python scripts/validate_timing.py --runs 50 --minutes 20 --lock-clock 1400
"""

import argparse
import json
import statistics
import time

import torch

from sns.clocks import ClockLockError, LockedClockPolicy, UnlockedClockPolicy
from sns.env import capture_fingerprint
from sns.timing import measure
from sns.types import MeasurementTier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--lock-clock", type=int, default=None,
                    help="SM clock in MHz. Omit to measure unlocked (Tier B/C).")
    ap.add_argument("-o", "--out", default="validate_timing.json")
    args = ap.parse_args()

    if args.runs < 3:
        raise SystemExit(
            "acceptance requires at least 3 runs: cross-run spread is the "
            "measurement, and fewer than 3 runs cannot establish it"
        )

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")

    policy = UnlockedClockPolicy()
    if args.lock_clock is not None:
        policy = LockedClockPolicy(target_sm_mhz=args.lock_clock)
        try:
            policy.apply()
            policy.restore()
        except ClockLockError as e:
            raise SystemExit(
                f"cannot lock clocks on this host: {e}\n"
                "Acceptance requires a Tier A host. Run scripts/probe_host.py "
                "on candidates to find one."
            )

    fingerprint = capture_fingerprint()
    print(json.dumps(fingerprint.model_dump(), indent=2))

    a = torch.randn(args.size, args.size, device="cuda", dtype=torch.float16)
    gap = (args.minutes * 60.0) / max(1, args.runs - 1)
    print(f"\n{args.runs} runs, {gap:.1f}s apart, ~{args.minutes:.0f} min total\n")

    runs = []
    for i in range(args.runs):
        r = measure(
            lambda: a @ a,
            warmup=args.warmup,
            iters=args.iters,
            policy=policy,
        )
        runs.append(r)
        print(
            f"  run {i + 1:2d}/{args.runs}  median {r.median_ms:.5f} ms  "
            f"CI [{r.ci95_lo_ms:.5f}, {r.ci95_hi_ms:.5f}]  "
            f"width {r.ci95_hi_ms - r.ci95_lo_ms:.5f}  tier {r.tier.value}"
        )
        if i < args.runs - 1:
            time.sleep(gap)

    medians = [r.median_ms for r in runs]
    spread = max(medians) - min(medians)
    widest_ci = max(r.ci95_hi_ms - r.ci95_lo_ms for r in runs)
    tiers = {r.tier.value for r in runs}
    passed = spread <= widest_ci and MeasurementTier.C.value not in tiers

    report = {
        "fingerprint": fingerprint.model_dump(),
        "config": vars(args),
        "runs": [r.model_dump() for r in runs],
        "median_of_medians_ms": statistics.median(medians),
        "cross_run_spread_ms": spread,
        "widest_within_run_ci_ms": widest_ci,
        "cross_run_cv_pct": round(
            100 * statistics.pstdev(medians) / statistics.mean(medians), 4
        ),
        "tiers_observed": sorted(tiers),
        "passed": passed,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 64)
    print(f"  cross-run spread      {spread:.5f} ms")
    print(f"  widest within-run CI  {widest_ci:.5f} ms")
    print(f"  tiers observed        {sorted(tiers)}")
    print(f"  ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    print("=" * 64)
    print(f"\nreport -> {args.out}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
