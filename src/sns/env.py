"""Environment capture via nvidia-smi and torch introspection."""

import subprocess

from sns.types import EnvironmentFingerprint

_ARCH_BY_CAP = {
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 7): "Ampere",
    (8, 9): "Ada",
    (9, 0): "Hopper",
    # sm_100 and sm_120 are both marketed as Blackwell but are not
    # interchangeable Triton targets: sm_120 lacks the TMEM subsystem used
    # for persistent-kernel optimization on datacenter Blackwell.
    (10, 0): "Blackwell-DC",
    (10, 3): "Blackwell-DC",
    (12, 0): "Blackwell-RTX",
    (12, 1): "Blackwell-RTX",
}

THROTTLE_FIELDS = {
    "sw_power_cap": "clocks_throttle_reasons.sw_power_cap",
    "hw_thermal_slowdown": "clocks_throttle_reasons.hw_thermal_slowdown",
    "sw_thermal_slowdown": "clocks_throttle_reasons.sw_thermal_slowdown",
    "hw_power_brake_slowdown": "clocks_throttle_reasons.hw_power_brake_slowdown",
}

# The governing rule: hardware-asserted throttling disqualifies a
# measurement; driver-reported software flags are recorded but do not,
# because they were measured stuck Active at idle on consumer hardware.
#
# Every *software* throttle flag we measured — sw_power_cap AND
# sw_thermal_slowdown — was Active on real consumer hardware (RTX 3060
# Laptop GPU) at idle, 55C, 18W: the card cold and doing nothing. Driver/
# vendor software policy is simply not trustworthy evidence of anything.
#
# hw_thermal_slowdown and hw_power_brake_slowdown are different: both are
# asserted by the GPU's own hardware safety circuits, not by software
# policy, and both are unambiguous. Both flags (plus observed clock
# variance, in clocks.assign_tier) gate the tier; every software flag is
# recorded as metadata only.
HW_THERMAL_REASON_KEYS = ("hw_thermal_slowdown",)
SW_THERMAL_REASON_KEYS = ("sw_thermal_slowdown",)
HW_POWER_BRAKE_REASON_KEYS = ("hw_power_brake_slowdown",)


def hw_thermal_throttle_active(snapshot: dict[str, str | None]) -> bool:
    """True if the hardware thermal assertion fired.

    This is a hardware-asserted throttle flag that gates the tier: it comes
    from the GPU's hardware safety circuit, not driver/vendor software
    policy. See hw_power_brake_active for the other one.
    """
    return any(snapshot.get(k) == "Active" for k in HW_THERMAL_REASON_KEYS)


def sw_thermal_throttle_active(snapshot: dict[str, str | None]) -> bool:
    """True if the software thermal flag is set. Metadata only — do not gate.

    Measured Active at idle (55C, 18W) on real laptop hardware; the flag
    is stuck, not evidence of anything.
    """
    return any(snapshot.get(k) == "Active" for k in SW_THERMAL_REASON_KEYS)


def hw_power_brake_active(snapshot: dict[str, str | None]) -> bool:
    """True if the hardware power-brake assertion fired.

    Same naming and same mechanism as hw_thermal_slowdown: asserted by the
    GPU's own hardware safety circuit, not driver/vendor software policy,
    so it gates the tier too. Measured Not Active throughout on real
    laptop hardware, including under sustained load.
    """
    return any(snapshot.get(k) == "Active" for k in HW_POWER_BRAKE_REASON_KEYS)


def hw_throttle_active(
    before: dict[str, str | None],
    after: dict[str, str | None],
    throttled_during: bool = False,
) -> bool:
    """True if any hardware-asserted throttle fired, before/after/mid-window.

    The rule: hardware-asserted throttling disqualifies a measurement.
    hw_thermal_slowdown and hw_power_brake_slowdown are both asserted by
    the GPU's own hardware safety circuits, not by driver/vendor software
    policy, so either one alone — in either snapshot, or observed live via
    NVML mid-window — is enough. This is the only thing (plus observed
    clock variance, in clocks.assign_tier) that gates the tier.
    """
    return (
        hw_thermal_throttle_active(before)
        or hw_thermal_throttle_active(after)
        or hw_power_brake_active(before)
        or hw_power_brake_active(after)
        or bool(throttled_during)
    )


def power_cap_active(snapshot: dict[str, str | None]) -> bool:
    """True if sw_power_cap is Active. Metadata only — do not gate.

    This is the normal state of every GPU under sustained load, including
    an A100 or H100 doing real work, and was also measured Active on an
    idle laptop GPU. Neither reading is instability.
    """
    return snapshot.get("sw_power_cap") == "Active"


def is_cuda_device(device: str) -> bool:
    """True when a torch-style device string names a CUDA device at all.

    compare()/measure() are CUDA-only tools; a device string of "cpu" (or
    anything else that isn't "cuda"/"cuda:N") must never be routed through
    them regardless of whether CUDA happens to be available elsewhere.
    """
    return device.split(":", 1)[0] == "cuda"


def arch_family(compute_cap: str | None) -> str | None:
    if not compute_cap:
        return None
    try:
        major, minor = (int(x) for x in str(compute_cap).split("."))
    except ValueError:
        return None
    return _ARCH_BY_CAP.get((major, minor), f"unknown-sm{major}{minor}")


def _run_smi(args: list[str], timeout: int = 30) -> tuple[int, str]:
    """Isolated for testability. Returns (returncode, stdout)."""
    try:
        p = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, p.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 127, ""


def smi_query(field: str, index: int = 0) -> str | None:
    """Query one field for one GPU, normalizing unsupported values to None."""
    rc, out = _run_smi(
        [f"--query-gpu={field}", "--format=csv,noheader,nounits", "-i", str(index)]
    )
    if rc != 0 or not out:
        return None
    value = out.splitlines()[0].strip()
    # nvidia-smi prints the literal "[N/A]" for unsupported fields. Letting
    # that through turns every downstream float() into a silent None.
    if value.strip("[]").upper() in ("N/A", "NOT SUPPORTED", ""):
        return None
    return value


def smi_query_float(field: str, index: int = 0) -> float | None:
    value = smi_query(field, index)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def throttle_snapshot(index: int = 0) -> dict[str, str | None]:
    return {name: smi_query(field, index) for name, field in THROTTLE_FIELDS.items()}


def capture_fingerprint(device_index: int = 0) -> EnvironmentFingerprint:
    import torch

    triton_version = None
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        pass

    compute_cap = smi_query("compute_cap", device_index)
    sm_count = None
    if torch.cuda.is_available():
        sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
        if compute_cap is None:
            major, minor = torch.cuda.get_device_capability(device_index)
            compute_cap = f"{major}.{minor}"

    return EnvironmentFingerprint(
        torch_version=torch.__version__,
        triton_version=triton_version,
        cuda_version=torch.version.cuda,
        driver_version=smi_query("driver_version", device_index),
        gpu_name=smi_query("name", device_index),
        compute_cap=compute_cap,
        arch_family=arch_family(compute_cap),
        sm_count=sm_count,
    )
