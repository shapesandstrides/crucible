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


def test_locked_policy_restores_when_readback_fails():
    """A failed lock must not leave the device pinned."""
    calls = []

    def record(args, **kw):
        calls.append(args)
        return (0, "")

    with patch("sns.clocks._run_smi", side_effect=record), patch(
        "sns.clocks.smi_query_float", return_value=1282.0
    ):
        p = LockedClockPolicy(target_sm_mhz=1500)
        with pytest.raises(ClockLockError):
            p.apply()

    assert ["-rgc"] in calls
    assert p.locked is False


def test_locked_policy_raises_when_clock_cannot_be_read():
    with patch("sns.clocks._run_smi", return_value=(0, "")), patch(
        "sns.clocks.smi_query_float", return_value=None
    ):
        with pytest.raises(ClockLockError):
            LockedClockPolicy(target_sm_mhz=1500).apply()


def test_locked_policy_verifies_the_power_cap_too():
    def fake_query(field, index=0):
        return {"clocks.sm": 1500.0, "power.limit": 200.0}.get(field)

    with patch("sns.clocks._run_smi", return_value=(0, "")), patch(
        "sns.clocks.smi_query_float", side_effect=fake_query
    ):
        with pytest.raises(ClockLockError, match="300"):
            LockedClockPolicy(target_sm_mhz=1500, power_cap_w=300).apply()


def test_locked_with_no_clock_samples_is_tier_b_not_a():
    """Absence of clock evidence is not stability."""
    assert assign_tier(True, [], throttle_fired=False) is MeasurementTier.B


def test_tier_a_boundary_at_exactly_30_mhz_range():
    """range == 30.0 is inclusive, so this is A."""
    assert assign_tier(True, [1500.0, 1530.0], throttle_fired=False) is MeasurementTier.A


def test_tier_c_boundary_cv_exactly_at_threshold_stays_out_of_c():
    """cv_percent([97, 103]) is exactly 3.0; the C test is strict >, so this is B."""
    assert assign_tier(False, [97.0, 103.0], throttle_fired=False) is MeasurementTier.B


def test_a_throttled_locked_gpu_is_never_tier_a():
    assert assign_tier(True, [1100.0] * 8, throttle_fired=True) is MeasurementTier.C
