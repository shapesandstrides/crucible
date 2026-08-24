"""GPU operating-point control and measurement-tier assignment."""

from typing import Protocol

from shapesandstrides.env import _run_smi, smi_query_float
from shapesandstrides.stats import cv_percent
from shapesandstrides.types import MeasurementTier

# Above this coefficient of variation the clock moved too much for the
# window to be trusted at all.
TIER_C_CV_PCT = 3.0
# Tier A additionally requires the clock to have barely moved.
TIER_A_RANGE_MHZ = 30.0
# nvidia-smi may land a few MHz off the requested boost bin.
LOCK_READBACK_TOLERANCE_MHZ = 30.0
# A power cap that lands a couple of watts off is fine; one that was refused is not.
POWER_READBACK_TOLERANCE_W = 2.0


class ClockLockError(RuntimeError):
    """Raised when a requested operating point could not be established."""


def assign_tier(
    locked: bool, clock_samples: list[float], hw_throttled: bool
) -> MeasurementTier:
    """Classify a measurement window.

    Throttle flags alone are not sufficient: on an RTX 3060 laptop the SM
    clock swung 495 MHz (5.1% CV) while the flags stayed silent, and two
    identical runs disagreed on whether throttling fired at all. Observed
    variance is the governing signal.

    Only ``hw_throttled`` and observed clock variance gate the tier —
    nothing else. The rule: hardware-asserted throttling disqualifies a
    measurement; driver-reported software flags are recorded but do not,
    because they were measured stuck ``Active`` at idle on consumer
    hardware. Concretely: ``hw_throttled`` is true when either
    ``hw_thermal_slowdown`` or ``hw_power_brake_slowdown`` fired — both are
    asserted by the GPU's own hardware safety circuits, not by software
    policy, so either one alone disqualifies a window. Every *software*
    throttle flag we tried turned out to be untrustworthy: measured on real
    consumer laptop hardware, ``sw_power_cap`` and ``sw_thermal_slowdown``
    were both `Active` at idle — 55C, 18W, the card cold and doing nothing.
    A flag stuck on at idle is not evidence of anything, so driver/vendor
    software policy is excluded entirely, and stable observed variance is
    what governs everything else. Software throttle activity is still
    recorded on TimingResult (``power_capped``, ``sw_thermal_flagged``) as
    metadata, never as a gate.
    """
    if hw_throttled:
        return MeasurementTier.C
    if clock_samples and cv_percent(clock_samples) > TIER_C_CV_PCT:
        return MeasurementTier.C
    if locked and clock_samples:
        spread = max(clock_samples) - min(clock_samples)
        if spread <= TIER_A_RANGE_MHZ:
            return MeasurementTier.A
    return MeasurementTier.B


class ClockPolicy(Protocol):
    locked: bool

    def apply(self) -> None: ...
    def restore(self) -> None: ...


class UnlockedClockPolicy:
    """Measure without pinning. Honest, and the only option on most machines."""

    locked = False

    def apply(self) -> None:
        return None

    def restore(self) -> None:
        return None


class LockedClockPolicy:
    """Pin the SM clock, and optionally the power cap, verifying by readback."""

    def __init__(self, target_sm_mhz: int, power_cap_w: int | None = None):
        self.target_sm_mhz = target_sm_mhz
        self.power_cap_w = power_cap_w
        self.locked = False

    def apply(self) -> None:
        _run_smi(["-lgc", f"{self.target_sm_mhz},{self.target_sm_mhz}"])
        try:
            # nvidia-smi exits 0 when it refuses the write, so the exit code
            # proves nothing. Only the readback does.
            observed = smi_query_float("clocks.sm")
            if (
                observed is None
                or abs(observed - self.target_sm_mhz) > LOCK_READBACK_TOLERANCE_MHZ
            ):
                raise ClockLockError(
                    f"requested {self.target_sm_mhz} MHz, device reports {observed}"
                )
            if self.power_cap_w is not None:
                _run_smi(["-pl", str(self.power_cap_w)])
                observed_w = smi_query_float("power.limit")
                if (
                    observed_w is None
                    or abs(observed_w - self.power_cap_w) > POWER_READBACK_TOLERANCE_W
                ):
                    raise ClockLockError(
                        f"requested {self.power_cap_w} W cap, device reports {observed_w}"
                    )
        except Exception:
            # The -lgc write already reached the device. Leaving it applied
            # after a loud failure silently pins the GPU for everything after.
            self.restore()
            raise
        self.locked = True

    def restore(self) -> None:
        _run_smi(["-rgc"])
        _run_smi(["-rmc"])
        self.locked = False
