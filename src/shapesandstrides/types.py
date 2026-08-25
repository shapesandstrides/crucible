"""Result types. These are the product's contract with its callers."""

from enum import Enum

from pydantic import BaseModel, field_validator


class MeasurementTier(str, Enum):
    """How much the numbers in a result can be trusted.

    A: clocks locked and verified stable. Full verdicts, drift-eligible.
    B: clocks floating, variance measured and folded into the interval.
    C: unstable or throttled. No performance verdict is valid.
    """

    A = "A"
    B = "B"
    C = "C"


class OracleTier(str, Enum):
    """How strong an answer key produced a correctness verdict.

    Parallel to MeasurementTier, and with the same shape: C means the strong
    verdict is unavailable, not that it failed.

    A: an independent answer key whose own arithmetic is PyTorch's -- a torch
       operator, or an expression composed of them, run in float64 on CPU.
       Supports the claim "this kernel computes the right function".
    B: an independent answer key containing arithmetic we cannot vouch for --
       the caller's own callable, prototype or numpy version. Supports only
       "this kernel agrees with your reference". If the reference is wrong,
       the agreement is worth nothing.
    C: no independent answer key. No correctness verdict is valid. Consistency
       checks may still have run, and a failure among them is a real defect --
       but a pass means "nothing contradicted itself", not "correct".

    What this ranks is the trustworthiness of the answer key's arithmetic. It
    deliberately does not rank whether the caller chose the right function to
    compare against: that is unknowable to us and the risk is identical at
    every tier, so folding it in would make the grade dishonest.
    """

    A = "A"
    B = "B"
    C = "C"


class CheckKind(str, Enum):
    """A family of check that actually ran.

    Deliberately a set, not a scale. These do not order against each other:
    config agreement can catch a kernel that breaks at one tile size and a
    golden baseline never would; a golden baseline catches an output that
    moved under a torch upgrade and config agreement never would. Ranking
    them would encode an ordering that is not true, so a report lists what
    ran and lets the tier carry the one thing that is genuinely ordered.

    Members are added only when a code path can emit them. An enum value no
    code produces is a false promise to whoever is reading the JSON.
    """

    REFERENCE = "reference"
    # Emitted only by check(error_budget=...). Grades the kernel against the
    # unfused chain's own error rather than a fixed tolerance, so it answers a
    # different question from REFERENCE and does not rank against it.
    ERROR_BUDGET = "error_budget"


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
    # True when either hardware-asserted throttle fired: hw_thermal_slowdown
    # or hw_power_brake_slowdown. Both come from the GPU's own hardware
    # safety circuits, not driver/vendor software policy, and either one
    # alone gates the tier down to C. hw_power_brake_flagged below records
    # which one it was; throttle_fired is the OR of both.
    throttle_fired: bool = False
    hw_power_brake_flagged: bool = False
    # Everything below is metadata only — recorded, never gating. Measured
    # on real consumer laptop hardware: sw_power_cap and sw_thermal_slowdown
    # were both Active at idle (55C, 18W, card cold and doing nothing), so
    # neither software flag is trustworthy evidence of anything.
    power_capped: bool = False
    sw_thermal_flagged: bool = False
    clock_cv_pct: float | None = None
    clock_range_mhz: float | None = None
    quantization_step_ms: float | None = None

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


class ComparisonResult(BaseModel):
    """A candidate measured against a baseline timed in the same session.

    The baseline is never cached across runs. Re-measuring it every time is
    what lets us tell "my kernel regressed" from "torch got faster", and it
    is what makes the ratio comparable across machines and architectures.
    """

    candidate: TimingResult
    baseline: TimingResult
    speedup: float
    speedup_ci_lo: float
    speedup_ci_hi: float

    @property
    def tier(self) -> MeasurementTier:
        """A comparison is only as trustworthy as its worse half."""
        order = [MeasurementTier.A, MeasurementTier.B, MeasurementTier.C]
        return max(
            (self.candidate.tier, self.baseline.tier), key=order.index
        )

    @property
    def is_performance_valid(self) -> bool:
        return self.tier is not MeasurementTier.C
