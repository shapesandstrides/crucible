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
    assert all(
        _numel(s) <= 10_000 for s in small
    ), "max_elements must exclude shapes that would OOM a small card"


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
