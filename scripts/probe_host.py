#!/usr/bin/env python3
"""
THROWAWAY SPIKE CODE. Not part of the shapesandstrides runner. Delete after we pick a host.

Answers one question: can this host actually be a measurement node?

A host qualifies only if it can pin the GPU into a fixed operating point and
KEEP it there under sustained load. nvidia-smi exits 0 when it refuses a write,
so every write here is verified by reading the value back.

Usage:
    python probe_host.py                 # full probe
    python probe_host.py --quick         # skip the 10-run stability preview
    python probe_host.py -o result.json

Run it on each candidate host and compare the JSON blobs.
"""

import argparse
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time

# SYS_ADMIN is bit 21 of the capability bitmask; it gates NVML control ops.
CAP_SYS_ADMIN_BIT = 21


def sh(*args, timeout=120):
    """Run a command, capturing everything. Never raises."""
    try:
        p = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout
        )
        return {
            "cmd": " ".join(args),
            "rc": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except FileNotFoundError:
        return {"cmd": " ".join(args), "rc": 127, "stdout": "", "stderr": "not found"}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(args), "rc": 124, "stdout": "", "stderr": "timeout"}


def smi_query(field):
    """Query one nvidia-smi field for GPU 0. Returns a stripped string or None."""
    r = sh(
        "nvidia-smi",
        f"--query-gpu={field}",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
    )
    if r["rc"] != 0 or not r["stdout"]:
        return None
    v = r["stdout"].splitlines()[0].strip()
    # nvidia-smi reports unsupported fields as the literal "[N/A]". Carrying
    # that through as a string turns every downstream float() into a silent
    # None, so normalize it at the boundary.
    if v.strip("[]").upper() in ("N/A", "NOT SUPPORTED", ""):
        return None
    return v


def smi_query_int(field):
    v = smi_query(field)
    if v is None:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


# ---------------------------------------------------------------- environment


def in_container():
    """Best-effort container detection. Matters because container != host privs."""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup") as f:
            blob = f.read()
        return any(k in blob for k in ("docker", "kubepods", "containerd", "lxc"))
    except OSError:
        return False


def effective_caps():
    """Read CapEff from /proc/self/status and decode whether SYS_ADMIN is held."""
    try:
        with open("/proc/self/status") as f:
            m = re.search(r"^CapEff:\s*([0-9a-fA-F]+)", f.read(), re.M)
        if not m:
            return {"capeff": None, "sys_admin": None}
        mask = int(m.group(1), 16)
        return {
            "capeff": m.group(1),
            "sys_admin": bool(mask & (1 << CAP_SYS_ADMIN_BIT)),
        }
    except OSError:
        return {"capeff": None, "sys_admin": None}


def arch_family(compute_cap):
    """
    Map compute capability to architecture family.

    Architecture is a first-class matrix axis, so every probe must name it. A
    kernel's speedup ratio is comparable across families; its absolute timing
    is not.
    """
    if not compute_cap:
        return None
    try:
        major, minor = (int(x) for x in str(compute_cap).split("."))
    except ValueError:
        return None
    # sm_100 and sm_120 are both "Blackwell" in marketing but are NOT
    # interchangeable Triton targets: sm_120 lacks the TMEM subsystem that
    # datacenter Blackwell uses for persistent-kernel optimizations. Treat them
    # as separate axis values or the matrix will report cross-SKU nonsense.
    return {
        (7, 0): "Volta", (7, 5): "Turing",
        (8, 0): "Ampere", (8, 6): "Ampere", (8, 7): "Ampere",
        (8, 9): "Ada",
        (9, 0): "Hopper",
        (10, 0): "Blackwell-DC", (10, 3): "Blackwell-DC",
        (12, 0): "Blackwell-RTX", (12, 1): "Blackwell-RTX",
    }.get((major, minor), f"unknown-sm{major}{minor}")


def capture_env():
    env = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "is_root": (os.geteuid() == 0) if hasattr(os, "geteuid") else None,
        "in_container": in_container(),
        "caps": effective_caps(),
        "has_nvidia_smi": shutil.which("nvidia-smi") is not None,
    }
    for key, field in [
        ("gpu_name", "name"),
        ("driver_version", "driver_version"),
        ("vbios", "vbios_version"),
        ("uuid", "uuid"),
        ("compute_cap", "compute_cap"),
        ("memory_total_mb", "memory.total"),
        ("power_limit_w", "power.limit"),
        ("power_max_w", "power.max_limit"),
        ("power_min_w", "power.min_limit"),
        ("persistence_mode", "persistence_mode"),
        ("clocks_max_sm", "clocks.max.sm"),
        ("clocks_max_mem", "clocks.max.memory"),
    ]:
        env[key] = smi_query(field)

    env["arch_family"] = arch_family(env.get("compute_cap"))

    # Power headroom decides whether a clock lock can survive load at all:
    # a 72W L4 has nowhere to hide, a 350W L40S has room to spare.
    try:
        env["power_headroom_w"] = float(env["power_max_w"]) - float(env["power_limit_w"])
    except (TypeError, ValueError):
        env["power_headroom_w"] = None

    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["sm_count"] = props.multi_processor_count
            env["l2_bytes"] = getattr(props, "L2_cache_size", None)
    except Exception as e:  # torch is optional for the permission half
        env["torch"] = None
        env["torch_error"] = repr(e)

    try:
        import triton

        env["triton"] = triton.__version__
    except Exception:
        env["triton"] = None

    return env


# ------------------------------------------------------------- write probing


def probe_write(label, argv, verify_field, expect=None, tolerance=0):
    """
    Attempt an nvidia-smi write, then READ IT BACK.

    nvidia-smi returns 0 on "not supported" and on "no permission", so the exit
    code is worthless on its own. Only the readback tells the truth.
    """
    before = smi_query(verify_field)
    r = sh(*argv)
    time.sleep(0.5)
    after = smi_query(verify_field)

    applied = None
    if expect is not None and after is not None:
        try:
            applied = abs(float(after) - float(expect)) <= tolerance
        except ValueError:
            applied = str(after).strip().lower() == str(expect).strip().lower()

    blob = " ".join([r["stdout"], r["stderr"]]).lower()
    return {
        "label": label,
        "cmd": r["cmd"],
        "rc": r["rc"],
        "output": (r["stdout"] + " " + r["stderr"]).strip()[:400],
        "value_before": before,
        "value_after": after,
        "applied": applied,
        "denied_permission": "permission" in blob,
        "unsupported": "not supported" in blob or "deprecated" in blob,
    }


def probe_privileges():
    """The core question: which control writes actually stick on this host?"""
    results = []

    results.append(
        probe_write("persistence_mode", ["nvidia-smi", "-pm", "1"],
                    "persistence_mode", expect="Enabled")
    )

    max_sm = smi_query_int("clocks.max.sm")
    max_mem = smi_query_int("clocks.max.memory")

    # Target ~80% of max SM clock: low enough to be sustainable under load,
    # high enough that we'd notice a silent fallback to idle clocks.
    target_sm = int(max_sm * 0.8) if max_sm else 1000
    results.append(
        probe_write(
            "lock_gpu_clocks",
            ["nvidia-smi", "-lgc", f"{target_sm},{target_sm}"],
            "clocks.sm",
            expect=target_sm,
            tolerance=30,
        )
    )

    if max_mem:
        results.append(
            probe_write(
                "lock_memory_clocks",
                ["nvidia-smi", "-lmc", f"{max_mem},{max_mem}"],
                "clocks.mem",
                expect=max_mem,
                tolerance=30,
            )
        )

    # Power cap: aim just under the default so the write is unambiguous.
    cur_limit = smi_query("power.limit")
    min_limit = smi_query("power.min_limit")
    if cur_limit and min_limit:
        try:
            target_w = max(float(min_limit), float(cur_limit) - 10.0)
            results.append(
                probe_write(
                    "power_limit",
                    ["nvidia-smi", "-pl", str(int(target_w))],
                    "power.limit",
                    expect=int(target_w),
                    tolerance=2,
                )
            )
        except ValueError:
            pass

    return {"target_sm_clock": target_sm, "attempts": results}


def restore():
    """Undo everything. A host we cannot cleanly restore is a host we cannot use."""
    return [
        sh("nvidia-smi", "-rgc"),
        sh("nvidia-smi", "-rmc"),
        sh("nvidia-smi", "-pl", str(smi_query("power.default_limit") or "")),
    ]


# ------------------------------------------------------- throttle accounting


THROTTLE_FIELDS = [
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown",
]


def throttle_snapshot():
    snap = {}
    for f in THROTTLE_FIELDS:
        snap[f.split(".")[-1]] = smi_query(f)
    return snap


def hold_under_load(target_sm, seconds=30):
    """
    The decisive test.

    Standard Kernel's finding: power constraints override manual clock settings.
    A host that accepts -lgc but throttles under real load is useless to us, and
    the acceptance is silent. So: apply sustained load, watch the clock.
    """
    try:
        import torch
    except Exception as e:
        return {"skipped": "torch unavailable", "error": repr(e)}

    if not torch.cuda.is_available():
        return {"skipped": "cuda unavailable"}

    dev = torch.device("cuda:0")
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)

    samples = []
    before = throttle_snapshot()
    t_end = time.time() + seconds
    while time.time() < t_end:
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()
        samples.append(
            {
                "t": round(time.time() - (t_end - seconds), 2),
                "sm_clock": smi_query_int("clocks.sm"),
                "power_w": smi_query("power.draw"),
                "temp_c": smi_query_int("temperature.gpu"),
            }
        )
    after = throttle_snapshot()

    clocks = [s["sm_clock"] for s in samples if s["sm_clock"] is not None]
    held = None
    if clocks and target_sm:
        # Allow a small boost-bin tolerance, but nothing resembling a drop.
        held = min(clocks) >= target_sm - 50

    # Throttle flags are NOT sufficient on their own: a GPU can swing hundreds
    # of MHz through boost/idle transitions without ever setting a throttle
    # reason. Observed clock variance is the signal that actually matters.
    clock_range = (max(clocks) - min(clocks)) if clocks else None
    clock_cv = (
        round(100 * statistics.pstdev(clocks) / statistics.mean(clocks), 2)
        if len(clocks) > 1 and statistics.mean(clocks)
        else None
    )

    return {
        "duration_s": seconds,
        "samples": samples,
        "clock_min": min(clocks) if clocks else None,
        "clock_median": statistics.median(clocks) if clocks else None,
        "clock_max": max(clocks) if clocks else None,
        "clock_range_mhz": clock_range,
        "clock_cv_pct": clock_cv,
        "clock_stable": (clock_range <= 30) if clock_range is not None else None,
        "target_sm_clock": target_sm,
        "lock_held_under_load": held,
        "throttle_before": before,
        "throttle_after": after,
        "throttle_fired": before != after,
    }


# ------------------------------------------------ M1 acceptance mini-preview


def bootstrap_ci(samples, n=2000, alpha=0.05):
    """Percentile bootstrap CI of the median. No scipy dependency."""
    if len(samples) < 2:
        return (None, None)
    rng = random.Random(0xC0FFEE)
    meds = []
    k = len(samples)
    for _ in range(n):
        meds.append(statistics.median(rng.choices(samples, k=k)))
    meds.sort()
    lo = meds[int((alpha / 2) * n)]
    hi = meds[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)


def time_once(warmup=200, iters=30, size=2048):
    """One measurement run: cuda events, pre-allocated, L2 flushed between iters."""
    import torch

    dev = torch.device("cuda:0")
    a = torch.randn(size, size, device=dev, dtype=torch.float16)
    b = torch.randn(size, size, device=dev, dtype=torch.float16)
    scratch = torch.empty(256 * 1024 * 1024 // 4, device=dev, dtype=torch.float32)

    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        scratch.zero_()  # flush L2 so every iteration starts cold
        starts[i].record()
        a @ b
        ends[i].record()
    torch.cuda.synchronize()

    return [starts[i].elapsed_time(ends[i]) for i in range(iters)]


def stability_preview(runs=10, gap_s=10):
    """
    Scaled-down M1 acceptance: does run-to-run spread sit inside per-run CI?

    Note the gap between runs is deliberate and adversarial. Standard Kernel
    found that sleeping between trials pulls the GPU out of steady state, which
    is exactly the condition a queue-fed measurement node lives in.
    """
    try:
        import torch
    except Exception as e:
        return {"skipped": "torch unavailable", "error": repr(e)}
    if not torch.cuda.is_available():
        return {"skipped": "cuda unavailable"}

    out = []
    for i in range(runs):
        s = time_once()
        lo, hi = bootstrap_ci(s)
        out.append(
            {
                "run": i,
                "n": len(s),
                "median_ms": statistics.median(s),
                "p10_ms": statistics.quantiles(s, n=10)[0],
                "p90_ms": statistics.quantiles(s, n=10)[8],
                "ci95_lo_ms": lo,
                "ci95_hi_ms": hi,
                "sm_clock": smi_query_int("clocks.sm"),
                "temp_c": smi_query_int("temperature.gpu"),
            }
        )
        if i < runs - 1:
            time.sleep(gap_s)

    medians = [r["median_ms"] for r in out]
    # The acceptance question: is cross-run spread contained by within-run CI?
    widest_ci = max((r["ci95_hi_ms"] - r["ci95_lo_ms"]) for r in out)
    spread = max(medians) - min(medians)
    return {
        "runs": out,
        "median_of_medians_ms": statistics.median(medians),
        "cross_run_spread_ms": spread,
        "widest_within_run_ci_ms": widest_ci,
        "spread_within_ci": spread <= widest_ci,
        "cross_run_cv_pct": round(
            100 * statistics.pstdev(medians) / statistics.mean(medians), 3
        ),
    }


# ---------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="probe_result.json")
    ap.add_argument("--quick", action="store_true", help="skip stability preview")
    ap.add_argument("--label", default=os.environ.get("PROBE_LABEL", "unlabeled"),
                    help="provider/instance name, e.g. runpod-l40s")
    args = ap.parse_args()

    report = {"label": args.label, "timestamp": time.time()}

    print(f"[1/5] environment ...")
    report["env"] = capture_env()

    if not report["env"]["has_nvidia_smi"]:
        report["verdict"] = "UNUSABLE: no nvidia-smi"
        print(json.dumps(report, indent=2))
        return

    print(f"[2/5] privilege writes (readback-verified) ...")
    report["privileges"] = probe_privileges()

    target = report["privileges"]["target_sm_clock"]
    print(f"[3/5] holding {target} MHz under 30s load ...")
    report["under_load"] = hold_under_load(target)

    if args.quick:
        report["stability"] = {"skipped": "--quick"}
    else:
        print(f"[4/5] stability preview: 10 runs, 10s gaps (~3 min) ...")
        report["stability"] = stability_preview()

    print(f"[5/5] restoring defaults ...")
    report["restore"] = restore()

    # Verdict
    attempts = {a["label"]: a for a in report["privileges"]["attempts"]}
    lock_ok = attempts.get("lock_gpu_clocks", {}).get("applied") is True
    power_ok = attempts.get("power_limit", {}).get("applied") is True
    held = report["under_load"].get("lock_held_under_load")
    stable = report["under_load"].get("clock_stable")

    if lock_ok and held and stable is False:
        verdict = "LOCK HELD BUT CLOCK UNSTABLE -- investigate before trusting"
    elif lock_ok and held:
        verdict = "MEASUREMENT-CAPABLE"
    elif lock_ok and held is False:
        verdict = "LOCK ACCEPTED BUT THROTTLED UNDER LOAD -- unusable"
    elif lock_ok:
        verdict = "LOCK OK, load test inconclusive (torch missing?)"
    else:
        verdict = "NOT MEASUREMENT-CAPABLE (clock lock refused) -- correctness node only"

    report["verdict"] = verdict
    report["summary"] = {
        "arch_family": report["env"].get("arch_family"),
        "gpu_name": report["env"].get("gpu_name"),
        "compute_cap": report["env"].get("compute_cap"),
        "power_headroom_w": report["env"].get("power_headroom_w"),
        "clock_lock_applied": lock_ok,
        "power_cap_applied": power_ok,
        "lock_held_under_load": held,
        "clock_range_mhz": report["under_load"].get("clock_range_mhz"),
        "clock_cv_pct": report["under_load"].get("clock_cv_pct"),
        "throttle_fired": report["under_load"].get("throttle_fired"),
        "spread_within_ci": report["stability"].get("spread_within_ci"),
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  {args.label}: {verdict}")
    print("=" * 60)
    print(json.dumps(report["summary"], indent=2))
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
