# Choosing a host

Not every GPU can produce trustworthy measurements. This page is about telling the difference before you rely on a number.

## Classify a machine

```bash
PROBE_LABEL=my-host python scripts/probe_host.py -o my-host.json
```

The probe attempts each control operation and **verifies it by reading the value back**, then applies sustained load and checks whether the setting actually held. It restores everything it touches before exiting.

It returns one of:

- `MEASUREMENT-CAPABLE` — clocks locked and held under load
- `LOCK ACCEPTED BUT THROTTLED UNDER LOAD` — the dangerous one, see below
- `LOCK HELD BUT CLOCK UNSTABLE` — investigate before trusting
- `NOT MEASUREMENT-CAPABLE` — correctness work only

## The failure mode worth knowing about

`nvidia-smi` exits `0` when it refuses a request. A consumer card can accept `--lock-gpu-clocks=2900`, print `GPU clocks set to '(gpuClkMin 2900, gpuClkMax 2900)'`, and then never exceed **2750 MHz** under load — [a documented, unresolved report on the RTX 4090](https://forums.developer.nvidia.com/t/can-not-to-lock-gpu-clock-rtx-4090/286603).

A 150 MHz silent lie is worse than a loud refusal, because you'll believe the number. This is why the probe verifies by readback and by holding load, not by exit code.

## Consumer cards are not suitable for measurement

Not because locking fails loudly — because it fails quietly. NVIDIA's own framing is that `nvmlDeviceSetGpuLockedClocks` targets "fully supported devices," and GeForce cards explicitly do not offer the level of control datacenter cards do.

Laptop GPUs are worse again: power limits are usually vBIOS-locked, and persistence mode is unsupported entirely on Windows WDDM.

You can still develop against a consumer card. Results will simply carry [Tier B or C](tiers.md), which is honest.

## Power headroom is a selection criterion

Power caps override clock locks, so the card must have room to be pinned *below* the point where power binds.

| Card | TDP | Suitability |
|---|---|---|
| L4 | 72 W | Poor — no headroom to pin below |
| A10G | 150 W | Good |
| L40S | 350 W | Excellent |
| A100 | 300–400 W | Excellent |

**The cheapest instance is usually the wrong one here.** An L4 is cheaper than an A10G and worse at the only job that matters.

## Recommended cloud host

`g5.xlarge` on AWS — A10G, 150 W, Ampere (sm_86).

AWS is the only major provider that [publicly documents GPU clock locking and persistence mode](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/optimize_gpu.html) on its GPU instances. When your entire trust model depends on `nvidia-smi -lgc` sticking, a vendor commitment is worth more than a lower hourly rate.

Two practical notes. New accounts default to **zero** G-instance quota, and approval is the long pole — request it before you need it, and ask for one instance rather than twenty, since small requests clear faster. And inside a container you need `--cap-add=SYS_ADMIN` for NVML control operations, which serverless GPU platforms do not grant. Clock locking is not available on serverless at all.

## Architecture families

Architecture is a first-class property, because a kernel tuned for one generation can lose badly on another — and Triton compiler bugs are sometimes architecture-specific.

`Ampere` (sm_80/86/87) · `Ada` (sm_89) · `Hopper` (sm_90) · `Blackwell-DC` (sm_100/103) · `Blackwell-RTX` (sm_120/121)

!!! note "Blackwell is two targets, not one"
    `sm_100` and `sm_120` are both marketed as Blackwell but are not interchangeable: `sm_120` lacks the TMEM subsystem that datacenter Blackwell uses for persistent-kernel optimization. The library reports them separately. Treating them as one produces cross-SKU nonsense.

Toolchain floors matter too. Blackwell requires torch ≥ 2.7, Triton ≥ 3.3, and CUDA ≥ 12.8. Turing runs the other way — Triton dropped `sm_75` after 3.2.

## Comparing hosts

The probe writes one JSON blob per host. Run it on each candidate and compare `summary`:

```json
{
  "arch_family": "Ampere",
  "gpu_name": "NVIDIA A10G",
  "power_headroom_w": 30.0,
  "clock_lock_applied": true,
  "lock_held_under_load": true,
  "clock_range_mhz": 0,
  "clock_cv_pct": 0.0,
  "throttle_fired": false
}
```

`clock_lock_applied: true` with `lock_held_under_load: false` is the case to reject. The lock was accepted and did not hold — exactly the silent failure described above.
