"""Tile-configuration discovery.

Triton exposes more than people realise. A plain @triton.jit kernel reveals
its constexpr parameter names; an @triton.autotune'd kernel reveals every
candidate value it might choose between. Verified against Triton 3.7.

That second fact matters for correctness, not just shape selection:
autotune picks a config per input shape, so a config only selected for
large inputs can be broken and never touched by small test shapes. A green
run does not prove the production path was covered.
"""

from pydantic import BaseModel


class TileSpace(BaseModel):
    """What we know about a kernel's tiling."""

    names: list[str] = []
    candidates: dict[str, list[int]] = {}
    source: str = "unknown"  # autotune | constexpr | declared

    @property
    def n_configs(self) -> int:
        """Total distinct tile values known across all parameters."""
        return sum(len(v) for v in self.candidates.values())

    def blocks_for(self, name: str) -> list[int]:
        return sorted(self.candidates.get(name, []))


def discover_tiles(kernel) -> TileSpace | None:
    """Introspect a Triton kernel for its tile configuration.

    Returns None for anything that is not a Triton kernel. Never raises:
    discovery is an optimisation, so failing to discover must fall back to
    the generic shape space rather than break the run.
    """
    try:
        # Autotuned kernels carry every candidate config.
        configs = getattr(kernel, "configs", None)
        if configs:
            candidates: dict[str, set[int]] = {}
            for c in configs:
                for k, v in getattr(c, "kwargs", {}).items():
                    if isinstance(v, int) and not isinstance(v, bool):
                        candidates.setdefault(k, set()).add(v)
            if candidates:
                return TileSpace(
                    names=sorted(candidates),
                    candidates={k: sorted(v) for k, v in candidates.items()},
                    source="autotune",
                )

        # A plain JITFunction exposes constexpr positions and arg names.
        arg_names = getattr(kernel, "arg_names", None)
        constexprs = getattr(kernel, "constexprs", None)
        if arg_names and constexprs is not None:
            names = [arg_names[i] for i in constexprs if 0 <= i < len(arg_names)]
            if names:
                return TileSpace(names=sorted(names), candidates={}, source="constexpr")
    except Exception:
        return None
    return None
