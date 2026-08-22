import pytest
from unittest.mock import patch

from sns.clocks import ClockLockError, LockedClockPolicy, UnlockedClockPolicy, assign_tier
from sns.types import MeasurementTier


def test_throttle_forces_tier_c():
    assert assign_tier(True, [1500.0] * 10, throttle_fired=True) is MeasurementTier.C


def test_high_clock_variance_forces_tier_c():
    """Measured on an RTX 3060 laptop: 5.1% CV under load, throttle flags silent."""
    unstable = [1282.0, 1777.0, 1500.0, 1777.0, 1282.0]
    assert assign_tier(True, unstable, throttle_fired=False) is MeasurementTier.C


def test_locked_and_stable_is_tier_a():
    assert assign_tier(True, [1500.0] * 10, throttle_fired=False) is MeasurementTier.A


def test_locked_but_wide_range_is_not_tier_a():
    """Lock reported success but the clock moved 60 MHz: not trustworthy."""
    drifting = [1500.0, 1530.0, 1560.0, 1540.0, 1520.0]
    assert assign_tier(True, drifting, throttle_fired=False) is MeasurementTier.B


def test_unlocked_but_stable_is_tier_b():
    assert assign_tier(False, [1500.0] * 10, throttle_fired=False) is MeasurementTier.B


def test_no_clock_samples_is_tier_b_when_unlocked():
    assert assign_tier(False, [], throttle_fired=False) is MeasurementTier.B


def test_unlocked_policy_is_a_no_op_and_reports_unlocked():
    p = UnlockedClockPolicy()
    assert p.locked is False
    p.apply()
    p.restore()


def test_locked_policy_raises_when_write_is_refused():
    """nvidia-smi exits 0 on refusal, so only the readback can be believed."""
    with patch("sns.clocks._run_smi", return_value=(0, "no permission")), patch(
        "sns.clocks.smi_query_float", return_value=1282.0
    ):
        with pytest.raises(ClockLockError, match="1500"):
            LockedClockPolicy(target_sm_mhz=1500).apply()


def test_locked_policy_succeeds_when_readback_matches():
    with patch("sns.clocks._run_smi", return_value=(0, "")), patch(
        "sns.clocks.smi_query_float", return_value=1500.0
    ):
        p = LockedClockPolicy(target_sm_mhz=1500)
        p.apply()
        assert p.locked is True


def test_locked_policy_accepts_small_readback_tolerance():
    with patch("sns.clocks._run_smi", return_value=(0, "")), patch(
        "sns.clocks.smi_query_float", return_value=1495.0
    ):
        LockedClockPolicy(target_sm_mhz=1500).apply()


def test_locked_policy_restore_resets_clocks():
    calls = []

    def record(args, **kw):
        calls.append(args)
        return (0, "")

    with patch("sns.clocks._run_smi", side_effect=record):
        LockedClockPolicy(target_sm_mhz=1500).restore()

    assert ["-rgc"] in calls
