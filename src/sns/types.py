"""Result types. These are the product's contract with its callers."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class MeasurementTier(str, Enum):
    """How much the numbers in a result can be trusted.

    A: clocks locked and verified stable. Full verdicts, drift-eligible.
    B: clocks floating, variance measured and folded into the interval.
    C: unstable or throttled. No performance verdict is valid.
    """

    A = "A"
    B = "B"
    C = "C"


class TimingResult(BaseModel):
    """A timing measurement.

    Deliberately defines no __float__, __int__ or __index__: a caller must
    not be able to collapse this into a bare number and lose the interval.
    """

    samples_ms: list[float]
    median_ms: float
    p10_ms: float
    p90_ms: float
    ci95_lo_ms: float
    ci95_hi_ms: float
    n: int
    tier: MeasurementTier
    warmup: int
    inner_reps: int = 1
    throttle_fired: bool = False
    clock_cv_pct: float | None = None
    clock_range_mhz: float | None = None

    @field_validator("samples_ms")
    @classmethod
    def _need_two_samples(cls, v: list[float]) -> list[float]:
        if len(v) < 2:
            raise ValueError("a timing result needs at least 2 samples")
        return v

    @property
    def is_performance_valid(self) -> bool:
        return self.tier is not MeasurementTier.C


class EnvironmentFingerprint(BaseModel):
    """Identifies the toolchain and device a run happened on.

    Two runs are only comparable when their fingerprints match exactly.
    """

    torch_version: str
    triton_version: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    gpu_name: str | None = None
    compute_cap: str | None = None
    arch_family: str | None = None
    sm_count: int | None = None

    def matches(self, other: "EnvironmentFingerprint") -> bool:
        return self.model_dump() == other.model_dump()
