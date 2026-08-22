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
