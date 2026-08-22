import json

import pytest

from sns.correctness import CorrectnessReport, ShapeOutcome
from sns.metrics import DeviceInfo, MemoryMetrics, RuntimeContext
from sns.oracle import OracleResult
from sns.records import RunRecord, list_runs, load_run, new_run_id, save_run
from sns.shapes import ShapeSpec
from sns.types import ComparisonResult, MeasurementTier, TimingResult


def _record(**kw):
    base = dict(
        run_id=new_run_id(),
        kernel_name="my_kernel",
        kernel_hash="abc123",
        device=DeviceInfo(gpu_name="NVIDIA A10G", sm_count=80, compute_capability="8.6"),
    )
    base.update(kw)
    return RunRecord(**base)


def test_run_ids_are_unique_and_sortable():
    ids = [new_run_id() for _ in range(50)]
    assert len(set(ids)) == 50
    assert ids == sorted(ids), "ids must sort chronologically"


def test_save_then_load_round_trips(tmp_path):
    r = _record()
    p = save_run(r, root=tmp_path)
    assert p.exists()
    back = load_run(r.run_id, root=tmp_path)
    assert back.run_id == r.run_id
    assert back.device.gpu_name == "NVIDIA A10G"


def test_saved_record_is_plain_json(tmp_path):
    """A third party with only the file must be able to read it."""
    r = _record()
    p = save_run(r, root=tmp_path)
    data = json.loads(p.read_text())
    assert data["run_id"] == r.run_id
    assert data["schema_version"] >= 1


def test_list_runs_returns_newest_first(tmp_path):
    ids = []
    for _ in range(3):
        r = _record()
        ids.append(r.run_id)
        save_run(r, root=tmp_path)
    listed = [r.run_id for r in list_runs(root=tmp_path)]
    assert listed == list(reversed(ids))


def test_list_runs_honours_limit(tmp_path):
    for _ in range(5):
        save_run(_record(), root=tmp_path)
    assert len(list_runs(root=tmp_path, limit=2)) == 2


def test_list_runs_on_empty_root_is_empty_not_an_error(tmp_path):
    assert list_runs(root=tmp_path / "nothing") == []


def test_load_missing_run_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run("run-does-not-exist", root=tmp_path)


def test_record_never_stores_kernel_source(tmp_path):
    """Source is the most sensitive thing we touch. Hash only."""
    r = _record()
    data = json.loads(save_run(r, root=tmp_path).read_text())
    assert "kernel_source" not in data
    assert data["kernel_hash"] == "abc123"


def test_record_carries_provenance(tmp_path):
    r = _record()
    assert r.provenance.entry_point in ("library", "cli")
    assert r.provenance.attested is True


def test_a_corrupt_file_is_skipped_not_fatal(tmp_path):
    save_run(_record(), root=tmp_path)
    (tmp_path / "runs" / "broken.json").write_text("{not json")
    assert len(list_runs(root=tmp_path)) == 1


def test_a_record_with_a_shape_mismatch_round_trips(tmp_path):
    """A run whose kernel returned the wrong shape must survive save/load.
    Using inf as a sentinel made such runs vanish from the catalog entirely."""
    mismatch = OracleResult(
        passed=False,
        max_abs_error=None,
        max_rel_error=None,
        mismatch_count=-1,
        total_elements=100,
        shape_mismatch=True,
    )
    outcome = ShapeOutcome(
        spec=ShapeSpec(dims=(10, 10), dtype="float32", layout="contiguous", label="10x10"),
        passed=False,
        seed=1,
        oracle=mismatch,
        outputs=[mismatch],
    )
    correctness = CorrectnessReport(
        outcomes=[outcome],
        passed=False,
        total=1,
        failed_count=1,
        minimal_failure=outcome,
        replay_command="sns replay --shape 10x10 --seed 1",
    )
    r = _record(correctness=correctness)
    save_run(r, root=tmp_path)

    listed = list_runs(root=tmp_path)
    assert len(listed) == 1, "the shape-mismatch record must not vanish from the catalog"

    back = load_run(r.run_id, root=tmp_path)
    assert back.correctness.outcomes[0].oracle.shape_mismatch is True
    assert back.correctness.outcomes[0].oracle.max_abs_error is None
    assert back.correctness.outcomes[0].oracle.max_rel_error is None


def _timing_result(median=1.0, tier=MeasurementTier.B):
    return TimingResult(
        samples_ms=[median] * 30,
        median_ms=median,
        p10_ms=median,
        p90_ms=median,
        ci95_lo_ms=median * 0.98,
        ci95_hi_ms=median * 1.02,
        n=30,
        tier=tier,
        warmup=200,
    )


def test_a_fully_populated_record_round_trips(tmp_path):
    """No existing test round-trips correctness, timing, and comparison
    together. That gap is what hid the inf-sentinel bug."""
    good = OracleResult(
        passed=True,
        max_abs_error=1e-5,
        max_rel_error=1e-5,
        mismatch_count=0,
        total_elements=100,
    )
    outcome = ShapeOutcome(
        spec=ShapeSpec(dims=(10, 10), dtype="float32", layout="contiguous", label="10x10"),
        passed=True,
        seed=1,
        oracle=good,
        outputs=[good],
    )
    correctness = CorrectnessReport(
        outcomes=[outcome], passed=True, total=1, failed_count=0,
    )
    candidate = _timing_result(1.0)
    baseline = _timing_result(2.0)
    comparison = ComparisonResult(
        candidate=candidate,
        baseline=baseline,
        speedup=2.0,
        speedup_ci_lo=1.9,
        speedup_ci_hi=2.1,
    )
    r = _record(correctness=correctness, timing=candidate, comparison=comparison)
    save_run(r, root=tmp_path)

    listed = list_runs(root=tmp_path)
    assert len(listed) == 1

    back = load_run(r.run_id, root=tmp_path)
    assert back.correctness.passed
    assert back.timing.median_ms == 1.0
    assert back.comparison.speedup == 2.0
    assert back.comparison.tier is MeasurementTier.B
