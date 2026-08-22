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
        existing_dims.add(((512, d),))
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


def generate_shapes(
    tier: ShapeTier,
    dtypes: list[str],
    max_elements: int | None = None,
) -> list[ShapeSpec]:
    """Generate the shape space for a tier. Deterministic and ordered."""
    seen: set[str] = set()
    out: list[ShapeSpec] = []
    for dtype in dtypes:
        for dims, layout in _base_cases(tier):
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
