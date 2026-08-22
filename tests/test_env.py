from unittest.mock import patch

from sns.env import arch_family, smi_query, throttle_snapshot


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
    with patch("sns.env._run_smi", return_value=(0, "[N/A]")):
        assert smi_query("power.limit") is None
    with patch("sns.env._run_smi", return_value=(0, "N/A")):
        assert smi_query("power.limit") is None


def test_smi_query_returns_value():
    with patch("sns.env._run_smi", return_value=(0, "1695")):
        assert smi_query("clocks.sm") == "1695"


def test_smi_query_returns_none_on_failure():
    with patch("sns.env._run_smi", return_value=(127, "")):
        assert smi_query("clocks.sm") is None


def test_smi_query_takes_first_gpu_line():
    with patch("sns.env._run_smi", return_value=(0, "1695\n1700")):
        assert smi_query("clocks.sm") == "1695"


def test_throttle_snapshot_collects_all_reasons():
    with patch("sns.env.smi_query", return_value="Not Active"):
        snap = throttle_snapshot()
    assert set(snap) == {
        "sw_power_cap",
        "hw_thermal_slowdown",
        "sw_thermal_slowdown",
        "hw_power_brake_slowdown",
    }
    assert all(v == "Not Active" for v in snap.values())
