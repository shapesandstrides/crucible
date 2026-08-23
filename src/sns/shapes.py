"""Shape space generation.

The default tier is small on purpose. A correctness sweep that takes
minutes gets run once; one that takes seconds gets run in the inner loop,
which is where it catches bugs.
"""

from enum import Enum

from pydantic import BaseModel, field_validator

# One past a power of two is where tail-mask bugs live: any kernel whose
# size is not divisible by its block size hits a partial final tile, and a
# missing `other=` in tl.load returns garbage there.
TILE_BOUNDARY_DIMS = [4097, 1025, 257]
# Primes defeat every convenient tiling.
PRIME_DIMS = [17, 97, 251, 1021]
POWER_OF_TWO_DIMS = [256, 1024, 4096]


class ShapeTier(str, Enum):
    FAST = "fast"
    CANONICAL = "canonical"
    EXHAUSTIVE = "exhaustive"


class ShapeSpec(BaseModel):
    dims: tuple[int, ...]
    dtype: str
    layout: str
    label: str

    @field_validator("dims")
    @classmethod
    def _non_empty(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("a shape must have at least one dimension")
        if any(d < 1 for d in v):
            raise ValueError(f"dimensions must be >= 1, got {v}")
        return v

    def numel(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


def _base_cases(tier: ShapeTier) -> list[tuple[tuple[int, ...], str]]:
    """(dims, layout) pairs, before dtype expansion."""
    cases: list[tuple[tuple[int, ...], str]] = [
        # Tile boundary: the single highest-value case.
        ((4097, 512), "contiguous"),
        ((1025,), "contiguous"),
        # Size-1 dims, which break naive indexing.
        ((1, 4096), "contiguous"),
        ((4096, 1), "contiguous"),
        # A prime, defeating every tiling.
        ((1021, 97), "contiguous"),
        # Extreme aspect ratios.
        ((8, 65536), "contiguous"),
        ((65536, 8), "contiguous"),
        # Non-contiguous input, which many kernels silently assume away.
        ((512, 512), "noncontiguous"),
        # An ordinary aligned case, so a passing run proves something.
        ((1024, 1024), "contiguous"),
    ]
    if tier is ShapeTier.FAST:
        return cases

    if tier is ShapeTier.CANONICAL:
        # Timing tier: aligned, large enough to dominate launch overhead.
        return [((n, n), "contiguous") for n in POWER_OF_TWO_DIMS]

    # Track dims/layout pairs already present to avoid duplicates from FAST tier.
    existing_dims = {c[0] for c in cases}

    for d in TILE_BOUNDARY_DIMS:
        if (d, 512) not in existing_dims:
            cases.append(((d, 512), "contiguous"))
            existing_dims.add((d, 512))
        cases.append(((512, d), "contiguous"))
        existing_dims.add((512, d))
    for d in PRIME_DIMS:
        cases.append(((d, d), "contiguous"))
        cases.append(((d, 1024), "noncontiguous"))
    for d in POWER_OF_TWO_DIMS:
        if (d, d) not in existing_dims:
            cases.append(((d, d), "contiguous"))
            existing_dims.add((d, d))
        cases.append(((d, d), "noncontiguous"))
    cases.append(((1,), "contiguous"))
    cases.append(((2, 3, 5, 7), "contiguous"))
    return cases


class DimClass(str, Enum):
    """A dimension is awkward in a *kind* of way.

    Describing the kinds and crossing them beats a hand-picked list,
    because it produces shapes awkward in several dimensions at once —
    which is where fused-kernel bugs live.
    """

    ALIGNED = "aligned"            # a clean multiple of every common block
    TAIL_ONE = "tail_one"          # one past a multiple: a tail of exactly 1
    TAIL_PARTIAL = "tail_partial"  # a substantial partial tile
    SUB_TILE = "sub_tile"          # smaller than one block; all masked
    PRIME = "prime"                # indivisible by every block, all at once
    UNIT = "unit"                  # size 1


# 4096 is a multiple of every power-of-two block up to itself, so 4097
# leaves a tail of exactly 1 for all of them simultaneously. That is what
# lets the generic path work without knowing anyone's block size.
_GENERIC_ALIGNED = 4096
_GENERIC_TAIL_ONE = 4097
_GENERIC_TAIL_PARTIAL = 4133
_GENERIC_SUB_TILE = 7
_GENERIC_PRIME = 1021


def dim_value(cls: DimClass, block: int | None = None) -> int:
    """A concrete dimension length for a class.

    Without a block size we use values awkward for every power-of-two block
    at once. With one, we can be exact — which matters for non-power-of-two
    blocks like 96, whose tails the generic values would miss entirely.
    """
    if block is None:
        return {
            DimClass.ALIGNED: _GENERIC_ALIGNED,
            DimClass.TAIL_ONE: _GENERIC_TAIL_ONE,
            DimClass.TAIL_PARTIAL: _GENERIC_TAIL_PARTIAL,
            DimClass.SUB_TILE: _GENERIC_SUB_TILE,
            DimClass.PRIME: _GENERIC_PRIME,
            DimClass.UNIT: 1,
        }[cls]

    # Large blocks would push a squared shape past the element cap and get
    # dropped entirely. Shrinking the multiple keeps the awkwardness — the
    # tail is what matters, not the number of whole tiles before it.
    mult = 4 if block <= 512 else 2
    if cls is DimClass.ALIGNED:
        return block * mult
    if cls is DimClass.TAIL_ONE:
        return block * mult + 1
    if cls is DimClass.TAIL_PARTIAL:
        return block * mult + max(2, block // 2)
    if cls is DimClass.SUB_TILE:
        return max(1, block - 1)
    if cls is DimClass.PRIME:
        return _GENERIC_PRIME
    return 1


# FAST cannot afford the full 6x6 cross product, so it takes the classes
# that catch the most, plus the interactions a one-dimension-at-a-time list
# can never reach.
_FAST_PAIRS: list[tuple[DimClass, DimClass]] = [
    (DimClass.ALIGNED, DimClass.ALIGNED),
    (DimClass.TAIL_ONE, DimClass.ALIGNED),
    (DimClass.ALIGNED, DimClass.TAIL_ONE),
    (DimClass.TAIL_ONE, DimClass.TAIL_ONE),        # both tails at once
    (DimClass.TAIL_ONE, DimClass.TAIL_PARTIAL),    # mixed tails
    (DimClass.PRIME, DimClass.PRIME),
    (DimClass.SUB_TILE, DimClass.TAIL_ONE),        # masked tile plus a tail
    (DimClass.UNIT, DimClass.ALIGNED),
]

# A default run must not try to allocate a shape that OOMs a small card.
_MAX_CROSSED_ELEMENTS = 40_000_000


def _class_pairs(tier: ShapeTier) -> list[tuple[DimClass, DimClass]]:
    if tier is ShapeTier.FAST:
        return _FAST_PAIRS
    if tier is ShapeTier.CANONICAL:
        return [(DimClass.ALIGNED, DimClass.ALIGNED)]
    return [(a, b) for a in DimClass for b in DimClass]


def _tile_cases(tier: ShapeTier, tiles) -> list[tuple[tuple[int, ...], str]]:
    """Class-crossed cases, block-aware when a tile space is known."""
    blocks: list[int | None] = [None]
    if tiles is not None and getattr(tiles, "candidates", None):
        # Cover every candidate: autotune may select any of them, and a
        # config only chosen for large inputs can still be broken.
        seen: set[int] = set()
        for name in tiles.names:
            seen.update(tiles.blocks_for(name))
        if seen:
            specific = sorted(seen)
            if tier is ShapeTier.FAST:
                # Smallest and largest, not the two smallest: the largest
                # block is what autotune picks for big shapes, so dropping
                # it skips the config most likely to run in production.
                specific = sorted({specific[0], specific[-1]})
            # Generic values stay: a declared block does not rule out other
            # implicit tilings, and large shapes exercise multi-block paths
            # that a small declared block never reaches.
            blocks = [None] + specific

    out: list[tuple[tuple[int, ...], str]] = []
    for block in blocks:
        for a, b in _class_pairs(tier):
            dims = (dim_value(a, block), dim_value(b, block))
            if dims[0] * dims[1] > _MAX_CROSSED_ELEMENTS:
                continue
            out.append((dims, "contiguous"))
    return out


def generate_shapes(
    tier: ShapeTier,
    dtypes: list[str],
    max_elements: int | None = None,
    tiles=None,
) -> list[ShapeSpec]:
    """Generate the shape space for a tier. Deterministic and ordered."""
    seen: set[str] = set()
    out: list[ShapeSpec] = []
    cases = _base_cases(tier) + _tile_cases(tier, tiles)
    for dtype in dtypes:
        for dims, layout in cases:
            spec = ShapeSpec(
                dims=dims,
                dtype=dtype,
                layout=layout,
                label=f"{'x'.join(map(str, dims))}-{layout}-{dtype}",
            )
            if max_elements is not None and spec.numel() > max_elements:
                continue
            if spec.label in seen:
                continue
            seen.add(spec.label)
            out.append(spec)
    return out
