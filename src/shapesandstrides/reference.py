"""Turning whatever the user gave us into an oracle.

Requiring a hand-written reference implementation is the largest single tax on
adopting this tool: it is the difference between adding a line and writing your
kernel twice. So `against=` accepts four things, in descending order of how
much work they cost the caller:

1. ``"torch.add"`` — a dotted path into PyTorch's own operator set. Costs
   nothing. PyTorch is the answer key.
2. ``lambda q, k, v: torch.softmax(q @ k.T / d**0.5, -1) @ v`` — a short torch
   expression. Most kernels described as having "no equivalent op" are fused
   compositions that do have a torch equivalent, just not a single one. Three
   lines of slow, obvious torch is categorically different work from a kernel.
3. Any callable the user already has — a numpy version, a for-loop, the
   research prototype written before it was optimised. It needs to be neither
   fast nor numerically careful: `oracle.reference_fp64` runs it in float64 on
   CPU.
4. ``None`` — genuinely novel, nothing to compare against. Not an error. The
   run degrades to a weaker class of check and says so.

Which of the four we got is recorded, never inferred away, so a verdict can
never present itself as stronger evidence than it is.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Callable

MAX_PROBE_ARITY = 4


class ReferenceResolutionError(Exception):
    """We could not turn what was passed into a callable oracle."""


class OracleKind(str, Enum):
    """Where the reference came from. Determines the oracle's strength."""

    TORCH_OP = "torch_op"
    EXPRESSION = "expression"
    USER_CALLABLE = "user_callable"
    NONE = "none"


@dataclass(frozen=True)
class ResolvedReference:
    kind: OracleKind
    label: str
    fn: Callable | None = None
    arity: int | None = None

    @property
    def available(self) -> bool:
        return self.fn is not None


NO_REFERENCE = ResolvedReference(kind=OracleKind.NONE, label="none")


def infer_arity(fn: Callable) -> int | None:
    """How many positional arguments `fn` requires, or None if we cannot tell.

    None is deliberate. Most C-bound torch operators have no introspectable
    signature, and defaulting to 2 there would be a silent wrong guess that
    surfaces later as a confusing shape error. Callers who need a number
    should fall back to `probe_arity`.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None

    n = 0
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            # *args tells us nothing about how many are actually wanted.
            return None
        if p.kind is p.KEYWORD_ONLY:
            continue
        if p.default is inspect.Parameter.empty:
            n += 1
    return n


def probe_arity(fn: Callable, max_arity: int = MAX_PROBE_ARITY) -> int | None:
    """Call `fn` with 1..max_arity tiny tensors and report the first arity
    that works.

    Empirical because introspection fails on exactly the callables we most
    want to support. Tensors are 2x2 float32 on CPU, so a wrong guess is
    cheap. Returns None rather than guessing if nothing works.
    """
    import torch

    for n in range(1, max_arity + 1):
        args = [torch.ones(2, 2, dtype=torch.float32) for _ in range(n)]
        try:
            fn(*args)
        except Exception:
            continue
        return n
    return None


def _import_dotted(path: str) -> object:
    """Resolve ``a.b.c`` by importing the longest importable prefix, then
    walking attributes for the rest."""
    parts = path.split(".")
    module = None
    consumed = 0
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        consumed = i
        break

    if module is None:
        raise ReferenceResolutionError(
            f"could not resolve reference {path!r}: no importable module in that path. "
            f"Give a dotted path such as 'torch.add', a lambda, or a callable."
        )

    obj: object = module
    for attr in parts[consumed:]:
        try:
            obj = getattr(obj, attr)
        except AttributeError as e:
            raise ReferenceResolutionError(
                f"could not resolve reference {path!r}: {attr!r} not found on "
                f"{'.'.join(parts[:consumed])}."
            ) from e
    return obj


def resolve(against: object) -> ResolvedReference:
    """Normalise `against` into a `ResolvedReference`.

    Accepts a dotted string, a callable, an already-resolved reference, or
    None. Anything else is an error rather than a best guess.
    """
    if isinstance(against, ResolvedReference):
        return against

    if against is None:
        return NO_REFERENCE

    if isinstance(against, str):
        if "." not in against:
            raise ReferenceResolutionError(
                f"reference {against!r} is not a dotted path. Use a fully "
                f"qualified name such as 'torch.add' so there is no ambiguity "
                f"about which operator is the answer key."
            )
        obj = _import_dotted(against)
        if not callable(obj):
            raise ReferenceResolutionError(
                f"reference {against!r} resolved to {type(obj).__name__}, which "
                f"is not callable."
            )
        kind = OracleKind.TORCH_OP if against.split(".")[0] == "torch" else OracleKind.USER_CALLABLE
        return ResolvedReference(
            kind=kind,
            label=against,
            fn=obj,
            arity=infer_arity(obj) or probe_arity(obj),
        )

    if callable(against):
        name = getattr(against, "__name__", None) or type(against).__name__
        is_lambda = name == "<lambda>"
        return ResolvedReference(
            kind=OracleKind.EXPRESSION if is_lambda else OracleKind.USER_CALLABLE,
            label="<expression>" if is_lambda else name,
            fn=against,
            arity=infer_arity(against),
        )

    raise ReferenceResolutionError(
        f"cannot use {type(against).__name__} as a reference. Pass a dotted path "
        f"('torch.add'), a lambda, a callable, or None."
    )
