"""In-process GPU telemetry.

Shelling out to nvidia-smi costs tens of milliseconds per call, which is far
too slow to sample inside a measurement loop — the sampling stalls the host
long enough that the GPU idles and every reading reflects an idle clock. NVML
runs in-process at microsecond latency, so it can be sampled without
perturbing what it measures.
"""


class ClockSampler:
    """Samples SM clock and throttle state cheaply, or not at all.

    If NVML is unavailable we collect NO clock evidence rather than collecting
    bad evidence. Tier A requires clock evidence, so a host without NVML can
    never reach it. That is the intended, conservative degradation.
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.available = False
        self._handle = None
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            self._nvml = pynvml
            self.available = True
        except Exception:
            self.available = False

    def sample_clock_mhz(self) -> float | None:
        if not self.available:
            return None
        try:
            return float(
                self._nvml.nvmlDeviceGetClockInfo(
                    self._handle, self._nvml.NVML_CLOCK_SM
                )
            )
        except Exception:
            return None

    def throttled_now(self) -> bool | None:
        """True if any throttle reason other than 'GpuIdle' is currently set."""
        if not self.available:
            return None
        try:
            reasons = self._nvml.nvmlDeviceGetCurrentClocksThrottleReasons(
                self._handle
            )
        except Exception:
            return None
        # Bit 0 is nvmlClocksThrottleReasonGpuIdle, which is not a throttle we
        # care about — an idle GPU between iterations is expected.
        idle_bit = getattr(self._nvml, "nvmlClocksThrottleReasonGpuIdle", 1)
        return bool(reasons & ~idle_bit & ~0x1)

    def shutdown(self) -> None:
        if self.available:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
