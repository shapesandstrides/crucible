import pytest
from typer.testing import CliRunner

from sns.cli import app
from sns.metrics import DeviceInfo
from sns.records import RunRecord, new_run_id, save_run
from sns.types import ComparisonResult, MeasurementTier, TimingResult

runner = CliRunner()


def _seed_run(root, name="my_kernel", **kw):
    r = RunRecord(
        run_id=new_run_id(),
        kernel_name=name,
        kernel_hash="abc123",
        device=DeviceInfo(gpu_name="NVIDIA A10G", compute_capability="8.6"),
        **kw,
    )
    save_run(r, root=root)
    return r


def _timing(median, tier=MeasurementTier.B):
    return TimingResult(
        samples_ms=[median] * 30, median_ms=median, p10_ms=median, p90_ms=median,
        ci95_lo_ms=median, ci95_hi_ms=median, n=30, tier=tier, warmup=200,
    )


def _comparison(tier):
    return ComparisonResult(
        candidate=_timing(1.0, tier), baseline=_timing(1.0, tier),
        speedup=1.0, speedup_ci_lo=0.99, speedup_ci_hi=1.01,
    )


def test_runs_on_empty_store_says_so_rather_than_crashing(tmp_path):
    res = runner.invoke(app, ["runs", "--root", str(tmp_path)])
    assert res.exit_code == 0
    assert "no runs" in res.stdout.lower()


def test_runs_lists_a_stored_run(tmp_path):
    r = _seed_run(tmp_path)
    res = runner.invoke(app, ["runs", "--root", str(tmp_path)])
    assert res.exit_code == 0
    assert r.run_id[:20] in res.stdout or "my_kernel" in res.stdout


def test_show_renders_a_run(tmp_path):
    r = _seed_run(tmp_path)
    res = runner.invoke(app, ["show", r.run_id, "--root", str(tmp_path)])
    assert res.exit_code == 0
    assert "my_kernel" in res.stdout
    assert "A10G" in res.stdout


def test_show_missing_run_exits_nonzero_with_a_clear_message(tmp_path):
    res = runner.invoke(app, ["show", "run-nope", "--root", str(tmp_path)])
    assert res.exit_code != 0
    assert "not found" in res.stdout.lower() or "no run" in res.stdout.lower()


def test_compare_two_runs(tmp_path):
    a = _seed_run(tmp_path, name="kernel_a")
    b = _seed_run(tmp_path, name="kernel_b")
    res = runner.invoke(app, ["compare", a.run_id, b.run_id, "--root", str(tmp_path)])
    assert res.exit_code == 0
    assert "kernel_a" in res.stdout and "kernel_b" in res.stdout


def test_compare_warns_when_environments_differ(tmp_path):
    """Two runs on different hardware are not comparable, and must say so."""
    a = _seed_run(tmp_path, name="k")
    b = RunRecord(
        run_id=new_run_id(), kernel_name="k", kernel_hash="abc123",
        device=DeviceInfo(gpu_name="NVIDIA H100", compute_capability="9.0"),
    )
    save_run(b, root=tmp_path)
    res = runner.invoke(app, ["compare", a.run_id, b.run_id, "--root", str(tmp_path)])
    assert res.exit_code == 0
    assert "differ" in res.stdout.lower() or "not comparable" in res.stdout.lower()


def test_runs_json_output_is_machine_readable(tmp_path):
    import json

    _seed_run(tmp_path)
    res = runner.invoke(app, ["runs", "--root", str(tmp_path), "--json"])
    assert res.exit_code == 0
    assert isinstance(json.loads(res.stdout), list)


def test_runs_withholds_a_verdict_for_an_unstable_run(tmp_path):
    """Tier C means no performance verdict is valid. Reporting PARITY for a
    throttled measurement is the failure this tier exists to prevent."""
    rec = _seed_run(tmp_path, comparison=_comparison(MeasurementTier.C))
    res = runner.invoke(app, ["runs", "--root", str(tmp_path)])
    assert res.exit_code == 0
    assert "PARITY" not in res.stdout
    assert "FASTER" not in res.stdout
    assert "UNSTABLE" in res.stdout


def test_show_flags_an_unstable_measurement(tmp_path):
    rec = _seed_run(tmp_path, comparison=_comparison(MeasurementTier.C))
    res = runner.invoke(app, ["show", rec.run_id, "--root", str(tmp_path)])
    assert res.exit_code == 0
    assert "tier C" in res.stdout.lower() or "unstable" in res.stdout.lower()


@pytest.mark.parametrize("tier", [MeasurementTier.A, MeasurementTier.B])
def test_runs_still_renders_a_verdict_for_stable_tiers(tmp_path, tier):
    """Do not over-correct: Tier A/B measurements must still get a verdict."""
    _seed_run(tmp_path, comparison=_comparison(tier))
    res = runner.invoke(app, ["runs", "--root", str(tmp_path)])
    assert res.exit_code == 0
    assert "UNSTABLE" not in res.stdout
    assert "PARITY" in res.stdout
