"""How much a format result can be trusted.

Parallel to ``MeasurementTier`` and ``OracleTier``, with the same semantics: C
is the absence of a verdict, not a failure.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class FormatTier(str, Enum):
    """The evidence behind a format result.

    A: this exact format was validated bit-for-bit against a native
       implementation -- torch's own fp16, bf16 or fp32. The simulator is not
       being taken on trust here; it was checked against reality.
    B: the simulator passed validation on other formats, but this one has no
       native counterpart to check against, so results rest on the simulator.
       The normal case for any reconstructed or proposed format, and the
       default -- so forgetting to validate can only ever under-claim.
    C: validation ran for this format and failed. No numeric claim derived
       from it is valid.

    A reconstructed cbfloat16 is grade B by construction and stays grade B
    until somebody runs it on Cerebras hardware. Saying that on every result is
    the difference between this and a confident blog post.
    """

    A = "A"
    B = "B"
    C = "C"


class Graded(BaseModel):
    """Mixin for any result carrying an evidence grade.

    The field is required and has no default. A result that could omit its
    grade would be a result that looks like a verdict without saying what backs
    it -- the same hole that was closed on ``CorrectnessReport``.
    """

    format_tier: FormatTier

    @property
    def is_format_valid(self) -> bool:
        return self.format_tier is not FormatTier.C
