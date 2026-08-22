import json

import pytest

from sns.metrics import DeviceInfo, MemoryMetrics, RuntimeContext
from sns.records import RunRecord, list_runs, load_run, new_run_id, save_run


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
