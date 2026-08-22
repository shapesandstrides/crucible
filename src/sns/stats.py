"""Statistics for timing results. Pure functions, no GPU, no scipy."""

import random
import statistics


def percentile(samples: list[float], p: float) -> float:
    """Linear-interpolated percentile. p is in [0, 1]."""
    if not samples:
        raise ValueError("percentile of an empty sample set")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    s = sorted(samples)
    if len(s) == 1:
        return float(s[0])
    idx = (len(s) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def cv_percent(samples: list[float]) -> float:
    """Coefficient of variation as a percentage. 0.0 for a zero mean."""
    if not samples:
        raise ValueError("cv of an empty sample set")
    if len(samples) == 1:
        return 0.0
    mean = statistics.mean(samples)
    if mean == 0:
        return 0.0
    return 100.0 * statistics.pstdev(samples) / abs(mean)


def bootstrap_ci(
    samples: list[float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0xC0FFEE,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the median.

    Seeded so a given sample set always yields the same interval — a
    reproducibility requirement, not a convenience.
    """
    if len(samples) < 2:
        raise ValueError("bootstrap CI needs at least 2 samples")
    rng = random.Random(seed)
    k = len(samples)
    medians = sorted(
        statistics.median(rng.choices(samples, k=k)) for _ in range(n_resamples)
    )
    alpha = 1.0 - confidence
    lo_i = int((alpha / 2) * n_resamples)
    hi_i = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return (float(medians[lo_i]), float(medians[hi_i]))


def ratio_ci(
    candidate: list[float],
    baseline: list[float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0xC0FFEE,
) -> tuple[float, float]:
    """CI of speedup = median(baseline) / median(candidate).

    Values above 1.0 mean the candidate is faster. Resampling both sides
    independently propagates uncertainty from each into the ratio, which a
    naive ratio-of-medians throws away.
    """
    if len(candidate) < 2 or len(baseline) < 2:
        raise ValueError("ratio CI needs at least 2 samples on each side")
    rng = random.Random(seed)
    ratios = []
    for _ in range(n_resamples):
        c = statistics.median(rng.choices(candidate, k=len(candidate)))
        b = statistics.median(rng.choices(baseline, k=len(baseline)))
        if c == 0:
            raise ValueError("candidate median resampled to zero")
        ratios.append(b / c)
    ratios.sort()
    alpha = 1.0 - confidence
    lo_i = int((alpha / 2) * n_resamples)
    hi_i = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return (float(ratios[lo_i]), float(ratios[hi_i]))
