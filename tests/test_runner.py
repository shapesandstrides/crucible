import pytest

torch = pytest.importorskip("torch")

from sns.records import load_run
from sns.runner import run_test
from sns.shapes import ShapeTier

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


@requires_gpu
def test_run_test_produces_a_complete_record(tmp_path):
    rec = run_test(
        lambda a, b: a + b,
        reference=lambda a, b: a + b,
        kernel_name="add",
        op_name="add",
        tier=ShapeTier.FAST,
        dtypes=["float32"],
        root=tmp_path,
        time_it=False,
    )
    assert rec.run_id
    assert rec.correctness is not None and rec.correctness.passed
    assert rec.device.gpu_name
    assert rec.duration_s is not None and rec.duration_s > 0


@requires_gpu
def test_run_test_persists_the_record(tmp_path):
    rec = run_test(
        lambda a, b: a + b, reference=lambda a, b: a + b,
        kernel_name="add", op_name="add", tier=ShapeTier.FAST,
        dtypes=["float32"], root=tmp_path, time_it=False,
    )
    assert load_run(rec.run_id, root=tmp_path).run_id == rec.run_id


@requires_gpu
def test_timing_is_skipped_when_correctness_fails(tmp_path):
    """Timing a wrong kernel produces a number that means nothing."""

    def broken(a, b):
        out = (a + b).clone()
        out.view(-1)[0] = 1e9
        return out

    rec = run_test(
        broken, reference=lambda a, b: a + b, kernel_name="broken",
        op_name="add", tier=ShapeTier.FAST, dtypes=["float32"],
        root=tmp_path, time_it=True,
    )
    assert not rec.correctness.passed
    assert rec.comparison is None, "must not time a kernel known to be wrong"


@requires_gpu
def test_timing_runs_on_the_canonical_tier_when_correct(tmp_path):
    rec = run_test(
        lambda a, b: a + b, reference=lambda a, b: a + b,
        kernel_name="add", op_name="add", tier=ShapeTier.FAST,
        dtypes=["float32"], root=tmp_path, time_it=True,
        warmup=10, iters=30,
    )
    assert rec.correctness.passed
    assert rec.comparison is not None
    assert rec.comparison.speedup > 0


@requires_gpu
def test_kernel_hash_is_stable_for_the_same_function(tmp_path):
    def k(a, b):
        return a + b

    one = run_test(k, reference=k, kernel_name="k", op_name="add",
                   tier=ShapeTier.FAST, dtypes=["float32"],
                   root=tmp_path, time_it=False)
    two = run_test(k, reference=k, kernel_name="k", op_name="add",
                   tier=ShapeTier.FAST, dtypes=["float32"],
                   root=tmp_path, time_it=False)
    assert one.kernel_hash == two.kernel_hash
    assert one.run_id != two.run_id


def test_kernel_hash_is_stable_for_a_partial():
    """repr() of a plain function embeds a memory address, so a partial-wrapped
    kernel would otherwise get a new identity every process."""
    import functools

    from sns.runner import _hash_callable

    def base(a, b, scale):
        return (a + b) * scale

    p = functools.partial(base, scale=2.0)
    assert _hash_callable(p) == _hash_callable(functools.partial(base, scale=2.0))
    # And it must not contain anything address-shaped.
    assert _hash_callable(p) == _hash_callable(p)


def test_different_kernels_hash_differently():
    from sns.runner import _hash_callable

    def one(a, b):
        return a + b

    def two(a, b):
        return a * b

    assert _hash_callable(one) != _hash_callable(two)
