import pytest

triton = pytest.importorskip("triton")
import triton.language as tl

from sns.tiles import TileSpace, discover_tiles


@triton.jit
def _plain(x_ptr, out_ptr, M, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < M, other=0.0), mask=offs < M)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8),
    ],
    key=["M"],
)
@triton.jit
def _tuned(x_ptr, out_ptr, M, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < M, other=0.0), mask=offs < M)


def test_discovers_constexpr_names_from_a_plain_jit_kernel():
    ts = discover_tiles(_plain)
    assert ts is not None
    assert set(ts.names) == {"BLOCK_M", "BLOCK_N"}
    assert ts.source == "constexpr"
    # A plain kernel exposes names but not values: they arrive at call time.
    assert ts.candidates == {}


def test_discovers_every_candidate_value_from_an_autotuned_kernel():
    ts = discover_tiles(_tuned)
    assert ts is not None
    assert ts.source == "autotune"
    assert sorted(ts.candidates["BLOCK_M"]) == [64, 128]
    assert sorted(ts.candidates["BLOCK_N"]) == [32, 64]


def test_autotuned_discovery_lists_configs_one_shape_would_never_select():
    """Autotune picks one config per shape, so a config only chosen for
    large inputs can be broken and never exercised. List them all."""
    ts = discover_tiles(_tuned)
    assert ts.n_configs == 4  # 2 BLOCK_M values + 2 BLOCK_N values
    assert ts.blocks_for("BLOCK_M") == [64, 128]


def test_returns_none_for_a_plain_python_function():
    assert discover_tiles(lambda x: x) is None


def test_never_raises_on_an_odd_object():
    assert discover_tiles(object()) is None


def test_declared_tiles_round_trip():
    ts = TileSpace(names=["BLOCK_M"], candidates={"BLOCK_M": [16, 32]}, source="declared")
    assert ts.n_configs == 2
    assert ts.blocks_for("BLOCK_M") == [16, 32]
