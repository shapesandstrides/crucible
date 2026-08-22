import pytest

torch = pytest.importorskip("torch")

from sns.metrics import (
    DeviceInfo,
    DispatchTrace,
    MemoryMetrics,
    RuntimeContext,
    collect_device_info,
    collect_memory_metrics,
    collect_runtime_context,
    trace_dispatch,
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def test_models_tolerate_missing_fields():
    """Every metric is optional: a machine without NVML still produces a record."""
    assert RuntimeContext().sm_clock_mhz is None
    assert MemoryMetrics().peak_allocated_bytes is None
    assert DispatchTrace().kernels == []


def test_device_info_degrades_without_a_gpu(monkeypatch):
    """A machine with no GPU must still yield a record. Every test in this
    file is @requires_gpu, so nothing else covers this path — and our CI
    runner has no GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = collect_device_info()

    assert d.gpu_name is None
    assert d.sm_count is None
    # Toolchain versions do not need a device and must still be present.
    assert d.torch_version


@requires_gpu
def test_device_info_is_populated():
    d = collect_device_info()
    assert d.gpu_name
    assert d.sm_count and d.sm_count > 0
    assert d.compute_capability and "." in d.compute_capability
    assert d.torch_version
    assert d.total_memory_mb and d.total_memory_mb > 0


@requires_gpu
def test_memory_metrics_track_an_allocation():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = collect_memory_metrics()
    big = torch.empty(64 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    after = collect_memory_metrics()
    del big
    assert after.peak_allocated_bytes >= before.peak_allocated_bytes
    assert after.peak_allocated_bytes >= 64 * 1024 * 1024


@requires_gpu
def test_runtime_context_reads_unprivileged_telemetry():
    c = collect_runtime_context()
    # NVML reads need no privileges; writes do. If NVML is present these
    # are populated, and if it is absent they are None. Both are valid.
    if c.sm_clock_mhz is not None:
        assert c.sm_clock_mhz > 0
        assert c.max_sm_clock_mhz >= c.sm_clock_mhz
        assert c.temperature_c is not None


@requires_gpu
def test_dispatch_trace_names_the_kernel_torch_actually_ran():
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    t = trace_dispatch(lambda: a @ a)
    assert t.kernels, "a matmul must dispatch at least one CUDA kernel"
    assert t.total_device_time_us > 0
    # cuBLAS kernel names encode the architecture and tiling.
    assert any(k.name for k in t.kernels)


@requires_gpu
def test_dispatch_trace_degrades_rather_than_raising():
    """Profiling can fail for many reasons; a run record must still be produced."""
    t = trace_dispatch(lambda: None)
    assert isinstance(t, DispatchTrace)
