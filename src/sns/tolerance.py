"""Per-(op, dtype) tolerances.

A single global allclose is how correctness suites produce false
confidence: too tight for fp16 reductions, too loose to catch a real fp32
bug. Tolerances live in a table so they can be argued about explicitly.
"""

# (atol, rtol) per dtype for elementwise work.
DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    "float64": (1e-12, 1e-12),
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (8e-3, 8e-3),
}

# Reductions accumulate error across terms, so they need more room. The
# multiplier is applied to the dtype's elementwise tolerance.
REDUCTION_OPS = {"sum", "mean", "matmul", "mm", "bmm", "dot", "softmax", "norm"}
REDUCTION_SLACK = 10.0

# A fused kernel stacks several operations, so its rounding error is larger
# than any single stage's budget. Slack compounds with the number of stages
# rather than being a flat multiplier, because that is how the error grows.
FUSION_SLACK_PER_STAGE = 2.0


def tolerance_for(
    op: str,
    dtype: str,
    fused_ops: list[str] | None = None,
    override: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return (atol, rtol). Raises KeyError for an unknown dtype.

    An unknown *op* falls back to the dtype default, because a custom
    kernel name tells us nothing. An unknown *dtype* raises, because
    guessing a tolerance is how you ship a wrong pass.

    `fused_ops` names the stages a fused kernel composes, e.g.
    ["layernorm", "matmul", "gelu"]. Its budget is wider than any single
    stage's: error compounds through the chain. `override` wins outright,
    for the cases where the author knows better than the table.
    """
    if override is not None:
        return override
    if dtype not in DEFAULT_TOLERANCES:
        raise KeyError(
            f"no tolerance defined for dtype {dtype!r}; add one to DEFAULT_TOLERANCES"
        )
    atol, rtol = DEFAULT_TOLERANCES[dtype]

    if fused_ops:
        # Widest single stage sets the floor, then compound per extra stage.
        for stage in fused_ops:
            if stage.lower() in REDUCTION_OPS:
                atol, rtol = atol * REDUCTION_SLACK, rtol * REDUCTION_SLACK
                break
        factor = FUSION_SLACK_PER_STAGE ** max(0, len(fused_ops) - 1)
        return (atol * factor, rtol * factor)

    if op.lower() in REDUCTION_OPS:
        return (atol * REDUCTION_SLACK, rtol * REDUCTION_SLACK)
    return (atol, rtol)
