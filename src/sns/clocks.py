"""GPU operating-point control and measurement-tier assignment."""

from typing import Protocol

from sns.env import _run_smi, smi_query_float
from sns.stats import cv_percent
from sns.types import MeasurementTier

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
    locked: bool, clock_samples: list[float], throttle_fired: bool
) -> MeasurementTier:
    """Classify a measurement window.

    Throttle flags alone are not sufficient: on an RTX 3060 laptop the SM
    clock swung 495 MHz (5.1% CV) while the flags stayed silent, and two
    identical runs disagreed on whether throttling fired at all. Observed
    variance is the governing signal.
    """
    if throttle_fired:
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
