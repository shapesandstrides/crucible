"""Unprivileged metric collection.

Everything here works without root, without admin, and without changing a
machine's configuration. GPU performance counters (L1/L2 hit rates, DRAM
traffic, achieved occupancy) are deliberately excluded: they require
elevated privileges, and demanding those would put a barrier in front of
the users we most want.
"""

from pydantic import BaseModel


class DeviceInfo(BaseModel):
    gpu_name: str | None = None
    compute_capability: str | None = None
    arch_family: str | None = None
    sm_count: int | None = None
    total_memory_mb: int | None = None
    l2_cache_bytes: int | None = None
    max_threads_per_sm: int | None = None
    warp_size: int | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    cudnn_version: int | None = None
    triton_version: str | None = None
    driver_version: str | None = None
    pcie_gen: int | None = None


class MemoryMetrics(BaseModel):
    current_allocated_bytes: int | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    alloc_retries: int | None = None
    oom_count: int | None = None


class RuntimeContext(BaseModel):
    """Conditions at measurement time. Explains why two runs differ."""

    sm_clock_mhz: float | None = None
    mem_clock_mhz: float | None = None
    max_sm_clock_mhz: float | None = None
    temperature_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    utilization_pct: float | None = None
    throttle_reasons: str | None = None


class KernelRecord(BaseModel):
    name: str
    device_time_us: float
    count: int


class DispatchTrace(BaseModel):
    """Which CUDA kernels actually ran, from the unprivileged torch profiler."""

    kernels: list[KernelRecord] = []
    total_device_time_us: float = 0.0
    error: str | None = None


def collect_device_info(device: int = 0) -> DeviceInfo:
    import torch

    from sns.env import arch_family

    p = torch.cuda.get_device_properties(device)
    cap = f"{p.major}.{p.minor}"

    triton_version = None
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        pass

    driver_version = None
    pcie_gen = None
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device)
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode()
        pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h)
    except Exception:
        pass

    return DeviceInfo(
        gpu_name=p.name,
        compute_capability=cap,
        arch_family=arch_family(cap),
        sm_count=p.multi_processor_count,
        total_memory_mb=p.total_memory // 1024**2,
        l2_cache_bytes=getattr(p, "L2_cache_size", None),
        max_threads_per_sm=getattr(p, "max_threads_per_multi_processor", None),
        warp_size=getattr(p, "warp_size", None),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        triton_version=triton_version,
        driver_version=driver_version,
        pcie_gen=pcie_gen,
    )


def collect_memory_metrics(device: int = 0) -> MemoryMetrics:
    import torch

    if not torch.cuda.is_available():
        return MemoryMetrics()
    s = torch.cuda.memory_stats(device)
    return MemoryMetrics(
        current_allocated_bytes=s.get("allocated_bytes.all.current"),
        peak_allocated_bytes=s.get("allocated_bytes.all.peak"),
        peak_reserved_bytes=s.get("reserved_bytes.all.peak"),
        alloc_retries=s.get("num_alloc_retries"),
        oom_count=s.get("num_ooms"),
    )


def collect_runtime_context(device: int = 0) -> RuntimeContext:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device)
    except Exception:
        return RuntimeContext()

    def _try(fn, scale=1.0):
        try:
            return fn() * scale
        except Exception:
            return None

    reasons = None
    try:
        reasons = hex(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h))
    except Exception:
        pass

    return RuntimeContext(
        sm_clock_mhz=_try(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)),
        mem_clock_mhz=_try(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)),
        max_sm_clock_mhz=_try(lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)),
        temperature_c=_try(lambda: pynvml.nvmlDeviceGetTemperature(h, 0)),
        power_draw_w=_try(lambda: pynvml.nvmlDeviceGetPowerUsage(h), 0.001),
        power_limit_w=_try(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h), 0.001),
        utilization_pct=_try(lambda: pynvml.nvmlDeviceGetUtilizationRates(h).gpu),
        throttle_reasons=reasons,
    )


def trace_dispatch(fn, top_n: int = 10) -> DispatchTrace:
    """Record which CUDA kernels a callable dispatches.

    Uses the torch profiler without hardware counters, which needs no
    elevated privileges. Never raises: a failed trace must not lose a run.
    """
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()

        records = []
        for e in prof.key_averages():
            t = getattr(e, "device_time_total", 0) or 0
            if t > 0:
                records.append(
                    KernelRecord(name=e.key, device_time_us=float(t), count=int(e.count))
                )
        records.sort(key=lambda k: k.device_time_us, reverse=True)
        return DispatchTrace(
            kernels=records[:top_n],
            total_device_time_us=sum(k.device_time_us for k in records),
        )
    except Exception as e:
        return DispatchTrace(error=f"{type(e).__name__}: {e}")
