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
        """True if the *hardware* thermal assertion is currently set.

        Checks only HwThermalSlowdown (bit 0x40) — the GPU's own hardware
        safety circuit — not any software-reported reason. Measured on real
        consumer laptop hardware: SwPowerCap (0x4) and SwThermalSlowdown
        (0x20) were both set at idle, 55C, 18W, card cold and doing
        nothing. A software flag stuck on at idle is not evidence of
        anything, so no software reason is trusted here, including the
        generic HwSlowdown (0x8) parent bit, which aggregates
        HwThermalSlowdown with HwPowerBrakeSlowdown and so is not the
        unambiguous hardware-thermal signal by itself. Only the specific
        thermal bit gates; everything else is metadata sampled separately
        from the before/after nvidia-smi snapshot in sns.timing.
        """
        if not self.available:
            return None
        try:
            reasons = self._nvml.nvmlDeviceGetCurrentClocksThrottleReasons(
                self._handle
            )
        except Exception:
            return None
        hw_thermal_bit = getattr(
            self._nvml, "nvmlClocksThrottleReasonHwThermalSlowdown", 0x40
        )
        return bool(reasons & hw_thermal_bit)

    def shutdown(self) -> None:
        if self.available:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
