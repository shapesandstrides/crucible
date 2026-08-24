"""A numeric format, and where each of its parameters came from.

Two of the four numbers that define a 16-bit float are routinely unpublished.
Cerebras publishes cbfloat16's exponent and mantissa widths and nothing else --
not the bias, not the subnormal policy, not the rounding mode, not the NaN
encoding. So a format object that cannot say where its own values came from
cannot support an honest result.

Hence the two rules this module exists to enforce:

1. ``bias`` has no default. You state what you assumed, or construction fails.
   A default is precisely how a user silently inherits somebody else's guess.
2. A parameter with no stated source reports as UNSTATED rather than being
   quietly filled in. Silence must never read as documentation.

``gfloat`` is the numeric core. Nothing here reimplements it.
"""

from __future__ import annotations

from enum import Enum

import gfloat
import gfloat.formats
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Provenance(str, Enum):
    """Where a parameter's value came from.

    UNSTATED is not an error and not a gap to fill in silently -- it is
    reported as itself, because a reader who cannot tell a documented value
    from an assumed one has no way to judge the result.
    """

    DOCUMENTED = "documented"
    INFERRED = "inferred"
    ASSUMED = "assumed"
    UNSTATED = "unstated"


def ieee_bias(exponent_bits: int) -> int:
    """The IEEE-conventional bias for a given exponent width.

    Exists so the conventional choice costs one call rather than a silent
    default. You still typed it, so it is still your stated assumption -- but
    you did not have to work it out. That is how the no-default rule stays
    honest without being hostile to use.
    """
    if exponent_bits < 1:
        raise ValueError("exponent_bits must be at least 1")
    return 2 ** (exponent_bits - 1) - 1


# Names reserved for the fully-published formats shipped at the bottom of this
# module. A user-constructed format may not take one: in a log or a JSON
# record, an unsourced spec must never be mistakable for a sourced one.
_RESERVED_NAMES = {
    "float64",
    "float32",
    "float16",
    "bfloat16",
    "float8_e4m3",
    "float8_e5m2",
    "binary16",
    "binary32",
    "binary64",
}


def _reject_reserved_name(name: str) -> None:
    if name.lower() in _RESERVED_NAMES:
        raise ValueError(
            f"{name!r} is reserved for a fully-published format shipped by this "
            f"package. Choose another name -- an unsourced spec must not be "
            f"mistakable for a sourced one in a log."
        )


class FormatSpec(BaseModel):
    """A float format, carrying the provenance of every parameter."""

    model_config = ConfigDict(frozen=True)

    name: str
    exponent_bits: int
    mantissa_bits: int
    # Deliberately no default. See the module docstring.
    bias: int
    has_subnormals: bool = True
    has_infinities: bool = True
    has_negative_zero: bool = True
    # None means "the IEEE-like count", derived in to_gfloat().
    num_high_nans: int | None = None
    provenance: dict[str, Provenance] = {}
    notes: str = ""

    @field_validator("exponent_bits")
    @classmethod
    def _exponent_at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("exponent_bits must be at least 1")
        return v

    @field_validator("mantissa_bits")
    @classmethod
    def _mantissa_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("mantissa_bits must be non-negative")
        return v

    @model_validator(mode="after")
    def _coherent(self) -> "FormatSpec":
        _reject_reserved_name(self.name)
        max_bias = 2**self.exponent_bits - 1
        if not 0 <= self.bias <= max_bias:
            raise ValueError(
                f"bias={self.bias} is outside the range an {self.exponent_bits}-bit "
                f"exponent can express (0..{max_bias}). For the IEEE convention, "
                f"use ieee_bias({self.exponent_bits})."
            )
        return self

    @property
    def total_bits(self) -> int:
        return 1 + self.exponent_bits + self.mantissa_bits

    @property
    def precision(self) -> int:
        """Significand bits, including the implicit leading bit."""
        return self.mantissa_bits + 1

    def provenance_of(self, field: str) -> Provenance:
        """The stated source of one parameter, or UNSTATED."""
        if field not in type(self).model_fields:
            raise KeyError(
                f"{field!r} is not a field of FormatSpec. Valid fields: "
                f"{sorted(type(self).model_fields)}"
            )
        return self.provenance.get(field, Provenance.UNSTATED)

    def to_gfloat(self) -> gfloat.FormatInfo:
        """The gfloat description of this format.

        Rebuilt on each call rather than cached: a frozen pydantic model and
        functools.cached_property do not compose cleanly, and construction is
        cheap enough that correctness is the better trade.
        """
        return gfloat.FormatInfo(
            name=self.name,
            k=self.total_bits,
            precision=self.precision,
            bias=self.bias,
            is_signed=True,
            domain=(
                gfloat.Domain.Extended if self.has_infinities else gfloat.Domain.Finite
            ),
            has_nz=self.has_negative_zero,
            num_high_nans=(
                self.num_high_nans
                if self.num_high_nans is not None
                else 2**self.mantissa_bits - 1
            ),
            has_subnormals=self.has_subnormals,
            is_twos_complement=False,
        )

    @classmethod
    def from_gfloat(
        cls,
        fi: gfloat.FormatInfo,
        *,
        provenance: dict[str, Provenance] | None = None,
        notes: str = "",
        _shipped: bool = False,
    ) -> "FormatSpec":
        """Wrap a gfloat FormatInfo.

        Uses model_construct, so this is the trusted path: gfloat's own
        predefined formats are already coherent and need no re-validation. The
        reserved-name rule is still applied by hand, so this cannot be used as
        a way around it.
        """
        if not _shipped:
            _reject_reserved_name(fi.name)
        return cls.model_construct(
            name=fi.name,
            exponent_bits=fi.k - fi.precision,
            mantissa_bits=fi.precision - 1,
            bias=fi.bias,
            has_subnormals=fi.has_subnormals,
            has_infinities=fi.domain is gfloat.Domain.Extended,
            has_negative_zero=fi.has_nz,
            num_high_nans=fi.num_high_nans,
            provenance=provenance or {},
            notes=notes,
        )

    # Derived limits are taken from gfloat rather than recomputed here, so
    # there is exactly one implementation of the arithmetic.
    @property
    def smallest_normal(self) -> float:
        return float(self.to_gfloat().smallest_normal)

    @property
    def smallest_subnormal(self) -> float:
        return float(self.to_gfloat().smallest_subnormal)

    @property
    def max_value(self) -> float:
        return float(self.to_gfloat().max)

    @property
    def eps(self) -> float:
        return float(self.to_gfloat().eps)


_IEEE = "IEEE 754-2019"
_OCP = "OCP 8-bit Floating Point Specification (OFP8) 1.0"

_SOURCED_FIELDS = (
    "name",
    "exponent_bits",
    "mantissa_bits",
    "bias",
    "has_subnormals",
    "has_infinities",
    "has_negative_zero",
    "num_high_nans",
)


def _shipped(fi: gfloat.FormatInfo, source: str) -> FormatSpec:
    """A fully-published format. Every parameter cites its standard.

    Only formats whose every parameter is published may be shipped under a name
    from this package; anything with an unknown parameter is constructed by the
    user, with their assumption stated. That is why there is no CBFLOAT16 here.
    """
    return FormatSpec.from_gfloat(
        fi,
        provenance={f: Provenance.DOCUMENTED for f in _SOURCED_FIELDS},
        notes=f"Fully specified by {source}.",
        _shipped=True,
    )


FLOAT64 = _shipped(gfloat.formats.format_info_binary64, _IEEE)
FLOAT32 = _shipped(gfloat.formats.format_info_binary32, _IEEE)
FLOAT16 = _shipped(gfloat.formats.format_info_binary16, _IEEE)
BFLOAT16 = _shipped(
    gfloat.formats.format_info_bfloat16, "the bfloat16 de-facto standard"
)
FLOAT8_E4M3 = _shipped(gfloat.formats.format_info_ocp_e4m3, _OCP)
FLOAT8_E5M2 = _shipped(gfloat.formats.format_info_ocp_e5m2, _OCP)
