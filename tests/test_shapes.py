import pytest

from sns.shapes import ShapeSpec, ShapeTier, generate_shapes


def _labels(specs):
    return {s.label for s in specs}


def test_fast_tier_includes_tile_boundary_cases():
    """4097 is the canonical Triton masking bug: one past a power of two."""
    specs = generate_shapes(ShapeTier.FAST, dtypes=["float16"])
    dims = {s.dims for s in specs}
    assert any(4097 in d for d in dims), "fast tier must probe tile boundaries"


def test_fast_tier_includes_size_one_dims():
    specs = generate_shapes(ShapeTier.FAST, dtypes=["float16"])
    assert any(1 in s.dims for s in specs)


def test_fast_tier_includes_a_prime_dim():
    specs = generate_shapes(ShapeTier.FAST, dtypes=["float16"])
    primes = {17, 31, 97, 127, 251, 509, 1021}
    assert any(any(d in primes for d in s.dims) for s in specs)


def test_fast_tier_includes_noncontiguous_layout():
    specs = generate_shapes(ShapeTier.FAST, dtypes=["float16"])
    assert "noncontiguous" in {s.layout for s in specs}


def test_fast_tier_is_small_enough_for_an_inner_loop():
    """The default tier must stay in the seconds range, so cap its size."""
    specs = generate_shapes(ShapeTier.FAST, dtypes=["float16"])
    assert 5 <= len(specs) <= 40


def test_exhaustive_is_a_superset_of_fast():
    fast = generate_shapes(ShapeTier.FAST, dtypes=["float16"])
    full = generate_shapes(ShapeTier.EXHAUSTIVE, dtypes=["float16"])
    assert _labels(fast) <= _labels(full)
    assert len(full) > len(fast)


def test_dtypes_multiply_the_space():
    one = generate_shapes(ShapeTier.FAST, dtypes=["float16"])
    two = generate_shapes(ShapeTier.FAST, dtypes=["float16", "float32"])
    assert len(two) == 2 * len(one)


def test_max_elements_filters_large_shapes():
    small = generate_shapes(ShapeTier.EXHAUSTIVE, dtypes=["float16"], max_elements=10_000)
    unfiltered = generate_shapes(ShapeTier.EXHAUSTIVE, dtypes=["float16"])

    assert len(small) > 0, "filtering must not drop everything"
    assert len(small) < len(unfiltered), "filtering must actually exclude something"
    assert all(_numel(s) <= 10_000 for s in small)
    # A shape well over the cap must be gone.
    assert not any(_numel(s) > 10_000 for s in small)
    assert any(_numel(s) > 10_000 for s in unfiltered), "test premise: some shapes exceed the cap"


def _numel(spec):
    n = 1
    for d in spec.dims:
        n *= d
    return n


def test_generation_is_deterministic():
    a = generate_shapes(ShapeTier.FAST, dtypes=["float16"])
    b = generate_shapes(ShapeTier.FAST, dtypes=["float16"])
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]


def test_labels_are_unique_within_a_tier():
    specs = generate_shapes(ShapeTier.EXHAUSTIVE, dtypes=["float16", "float32"])
    labels = [s.label for s in specs]
    assert len(labels) == len(set(labels))


def test_shape_spec_rejects_empty_dims():
    with pytest.raises(ValueError):
        ShapeSpec(dims=(), dtype="float16", layout="contiguous", label="x")


from sns.shapes import DimClass, dim_value


def test_tail_one_is_awkward_for_every_power_of_two_block():
    """One past a multiple of 4096 leaves a tail of exactly 1 for every
    power-of-two block up to 4096 - simultaneously."""
    v = dim_value(DimClass.TAIL_ONE)
    for block in (16, 32, 64, 128, 256, 512, 1024):
        assert v % block == 1, f"{v} is not one-past-a-multiple of {block}"


def test_prime_dim_is_indivisible_by_every_common_block():
    v = dim_value(DimClass.PRIME)
    for block in (16, 32, 64, 128, 256):
        assert v % block != 0


def test_aligned_dim_divides_every_common_block_evenly():
    v = dim_value(DimClass.ALIGNED)
    for block in (16, 32, 64, 128, 256, 512, 1024):
        assert v % block == 0


def test_sub_tile_is_smaller_than_a_typical_block():
    assert dim_value(DimClass.SUB_TILE) < 16


def test_dim_value_is_exact_when_a_block_size_is_known():
    """With the real block known we can be precise instead of generic -
    which matters for non-power-of-two blocks the generic values miss."""
    assert dim_value(DimClass.ALIGNED, block=96) % 96 == 0
    assert dim_value(DimClass.TAIL_ONE, block=96) % 96 == 1
    assert dim_value(DimClass.TAIL_PARTIAL, block=96) % 96 not in (0, 1)
    assert dim_value(DimClass.SUB_TILE, block=96) < 96


def test_exhaustive_crosses_dimension_classes():
    """The whole point: two awkward dimensions at once, not one at a time."""
    specs = generate_shapes(ShapeTier.EXHAUSTIVE, dtypes=["float16"])
    assert any(
        len(s.dims) == 2 and s.dims[0] % 128 == 1 and s.dims[1] % 128 == 1
        for s in specs
    ), "exhaustive must include a shape awkward in BOTH dimensions"


def test_fast_includes_a_double_tail_case():
    specs = generate_shapes(ShapeTier.FAST, dtypes=["float16"])
    assert any(
        len(s.dims) == 2 and s.dims[0] % 1024 == 1 and s.dims[1] % 1024 == 1
        for s in specs
    ), "even the fast tier needs one both-dimensions-awkward case"


def test_declared_tiles_produce_block_specific_shapes():
    from sns.tiles import TileSpace

    ts = TileSpace(names=["BLOCK_M"], candidates={"BLOCK_M": [96]}, source="declared")
    specs = generate_shapes(ShapeTier.FAST, dtypes=["float16"], tiles=ts)
    # 96 is not a power of two, so only block-aware generation finds its tail.
    assert any(s.dims[0] % 96 == 1 for s in specs), "must straddle the declared block"


def test_tiles_are_optional():
    assert len(generate_shapes(ShapeTier.FAST, dtypes=["float16"], tiles=None)) > 0


def test_fast_stays_inner_loop_sized_with_tiles():
    from sns.tiles import TileSpace

    ts = TileSpace(names=["BLOCK_M"], candidates={"BLOCK_M": [32, 64, 128]}, source="declared")
    specs = generate_shapes(ShapeTier.FAST, dtypes=["float16"], tiles=ts)
    assert len(specs) <= 40, "the fast tier must stay usable in an inner loop"
