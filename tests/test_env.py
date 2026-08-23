from unittest.mock import patch

import pytest

from shapesandstrides.env import (
    arch_family,
    capture_fingerprint,
    hw_power_brake_active,
    hw_throttle_active,
    smi_query,
    throttle_snapshot,
)


def test_arch_family_maps_known_capabilities():
    assert arch_family("8.0") == "Ampere"
    assert arch_family("8.6") == "Ampere"
    assert arch_family("8.9") == "Ada"
    assert arch_family("9.0") == "Hopper"


def test_arch_family_splits_blackwell():
    """sm_120 lacks the TMEM subsystem sm_100 has. Not the same target."""
    assert arch_family("10.0") == "Blackwell-DC"
    assert arch_family("12.0") == "Blackwell-RTX"


def test_arch_family_handles_unknown_and_missing():
    assert arch_family("11.5") == "unknown-sm115"
    assert arch_family(None) is None
    assert arch_family("garbage") is None


def test_smi_query_normalizes_not_available():
    """nvidia-smi prints the literal '[N/A]'; it must not reach a float()."""
    with patch("shapesandstrides.env._run_smi", return_value=(0, "[N/A]")):
        assert smi_query("power.limit") is None
    with patch("shapesandstrides.env._run_smi", return_value=(0, "N/A")):
        assert smi_query("power.limit") is None


def test_smi_query_returns_value():
    with patch("shapesandstrides.env._run_smi", return_value=(0, "1695")):
        assert smi_query("clocks.sm") == "1695"


def test_smi_query_returns_none_on_failure():
    with patch("shapesandstrides.env._run_smi", return_value=(127, "")):
        assert smi_query("clocks.sm") is None


def test_smi_query_takes_first_gpu_line():
    with patch("shapesandstrides.env._run_smi", return_value=(0, "1695\n1700")):
        assert smi_query("clocks.sm") == "1695"


def test_throttle_snapshot_collects_all_reasons():
    with patch("shapesandstrides.env.smi_query", return_value="Not Active"):
        snap = throttle_snapshot()
    assert set(snap) == {
        "sw_power_cap",
        "hw_thermal_slowdown",
        "sw_thermal_slowdown",
        "hw_power_brake_slowdown",
    }
    assert all(v == "Not Active" for v in snap.values())


def test_hw_power_brake_alone_is_hardware_asserted_throttling():
    """hw_power_brake_slowdown is asserted by the same hardware safety
    circuit, by the same naming and mechanism, as hw_thermal_slowdown —
    measured Not Active throughout on real laptop hardware, including
    under sustained load, so a reading of Active is trustworthy evidence."""
    snap = {"hw_power_brake_slowdown": "Active", "hw_thermal_slowdown": "Not Active"}
    assert hw_power_brake_active(snap) is True


def test_hw_power_brake_gates_the_tier_alongside_hw_thermal():
    """The governing principle: hardware assertions gate, software flags
    do not. hw_power_brake_slowdown is a hardware assertion by the same
    naming and mechanism as hw_thermal_slowdown, so it must gate too."""
    not_active = {"hw_thermal_slowdown": "Not Active", "hw_power_brake_slowdown": "Not Active"}
    brake_fired = {"hw_thermal_slowdown": "Not Active", "hw_power_brake_slowdown": "Active"}

    assert hw_throttle_active(not_active, not_active) is False
    assert hw_throttle_active(brake_fired, not_active) is True
    assert hw_throttle_active(not_active, brake_fired) is True


def test_sw_flags_alone_do_not_feed_hw_throttle_active():
    """sw_power_cap and sw_thermal_slowdown are metadata only, even when
    hw_throttle_active is asked to consider both snapshots."""
    sw_only = {
        "sw_power_cap": "Active",
        "sw_thermal_slowdown": "Active",
        "hw_thermal_slowdown": "Not Active",
        "hw_power_brake_slowdown": "Not Active",
    }
    assert hw_throttle_active(sw_only, sw_only) is False


def test_capture_fingerprint_uses_smi_compute_cap_when_available():
    torch = pytest.importorskip("torch")

    def fake_smi(field, index=0):
        return {
            "compute_cap": "9.0",
            "name": "NVIDIA H100",
            "driver_version": "550.0",
        }.get(field)

    with patch("shapesandstrides.env.smi_query", side_effect=fake_smi), patch.object(
        torch.cuda, "is_available", return_value=False
    ):
        fp = capture_fingerprint()

    assert fp.compute_cap == "9.0"
    assert fp.arch_family == "Hopper"
    assert fp.gpu_name == "NVIDIA H100"
    assert fp.sm_count is None


def test_capture_fingerprint_falls_back_to_torch_in_the_same_format():
    """Both paths must yield an identical compute_cap string. A format
    mismatch would make one machine's fingerprints compare as two."""
    torch = pytest.importorskip("torch")

    with patch("shapesandstrides.env.smi_query", return_value=None), patch.object(
        torch.cuda, "is_available", return_value=True
    ), patch.object(
        torch.cuda, "get_device_capability", return_value=(9, 0)
    ), patch.object(torch.cuda, "get_device_properties") as props:
        props.return_value.multi_processor_count = 132
        fp = capture_fingerprint()

    # Identical to the nvidia-smi path above — that is the whole point.
    assert fp.compute_cap == "9.0"
    assert fp.arch_family == "Hopper"
    assert fp.sm_count == 132
