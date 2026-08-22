from unittest.mock import patch

import pytest

from sns.env import arch_family, capture_fingerprint, smi_query, throttle_snapshot


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


def test_capture_fingerprint_uses_smi_compute_cap_when_available():
    torch = pytest.importorskip("torch")

    def fake_smi(field, index=0):
        return {
            "compute_cap": "9.0",
            "name": "NVIDIA H100",
            "driver_version": "550.0",
        }.get(field)

    with patch("sns.env.smi_query", side_effect=fake_smi), patch.object(
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

    with patch("sns.env.smi_query", return_value=None), patch.object(
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
