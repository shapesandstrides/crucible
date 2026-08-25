"""Two checks that need no answer key at all.

**Poison fill.** Fill the output with NaN before launching. Any NaN still there
afterwards is an element the kernel did not write. Without this the buffer
holds whatever was in memory -- usually plausible-looking floats -- so a
partial write reads as a slightly wrong answer, or as no failure at all.

This catches the most common Triton bug there is: 1000 elements with
``BLOCK=128`` is seven full blocks and a tail of 104, and a mask that is off by
one silently skips the tail.

**Canary padding.** Allocate past the logical end and write a sentinel there.
If it changes, the kernel wrote out of bounds. ``compute-sanitizer`` catches
this and more, but is orders of magnitude too slow to run per candidate in an
autotuning loop. The canary costs eight elements.

Neither check can tell you a kernel computes the right function. They raise the
floor; they do not establish correctness. A report resting only on these is
tier C.
"""

from __future__ import annotations

from pydantic import BaseModel

# Recognisable in a hex dump, and not a value a real kernel is likely to write.
CANARY_VALUE = 1234.5678
CANARY_ELEMENTS = 8


class BufferReport(BaseModel):
    unwritten_count: int
    total_elements: int
    canary_intact: bool
    # False when the buffer was not allocated by `poisoned_output` and so has
    # no sentinel to check. Distinguished from `canary_intact` because
    # reporting "intact" for a check that never ran claims a guarantee we do
    # not have -- the same distinction rule 7 draws between INCORRECT and
    # ERROR.
    canary_present: bool
    passed: bool


def poisoned_output(shape, dtype, device):
    """A NaN-filled tensor of ``shape``, backed by storage with canary padding.

    Returns an ordinary tensor you can pass straight to a kernel. The padding
    sits after it in the same allocation and is reachable only by a write that
    runs past the end.

    ``inspect_buffer`` recovers the padding from the tensor's own storage, so
    nothing needs to be tracked alongside it and the tensor stays a plain
    tensor -- no wrapper to unwrap, no attribute to preserve through a
    ``.view()`` or a ``.reshape()``.
    """
    import torch

    numel = 1
    for dim in shape:
        numel *= dim

    backing = torch.empty(numel + CANARY_ELEMENTS, dtype=dtype, device=device)
    backing[:numel] = float("nan")
    backing[numel:] = CANARY_VALUE

    return backing[:numel].view(*shape)


def inspect_buffer(buf) -> BufferReport:
    """Count unwritten elements and verify the canary, if there is one."""
    import torch

    flat = buf.detach().reshape(-1)
    total = int(flat.numel())
    unwritten = int(torch.isnan(flat).sum().item())

    # Recover the whole allocation behind this view. A tensor from
    # `poisoned_output` has exactly CANARY_ELEMENTS of slack; anything else is
    # treated as having no canary rather than as having a broken one.
    storage_elements = buf.untyped_storage().nbytes() // buf.element_size()
    canary_present = storage_elements >= total + CANARY_ELEMENTS

    canary_intact = True
    if canary_present:
        padding = torch.empty(0, dtype=buf.dtype, device=buf.device)
        padding.set_(
            buf.untyped_storage(),
            storage_offset=total,
            size=(CANARY_ELEMENTS,),
        )
        expected = torch.full_like(padding, CANARY_VALUE)
        canary_intact = bool(torch.isclose(padding, expected).all().item())

    return BufferReport(
        unwritten_count=unwritten,
        total_elements=total,
        canary_intact=canary_intact,
        canary_present=canary_present,
        passed=(unwritten == 0 and canary_intact),
    )


class DeterminismReport(BaseModel):
    runs: int
    # How many repeats differed from the first. Counted rather than flagged so
    # a race that fires occasionally is distinguishable from one that fires
    # every time -- the two feel very different to debug.
    varying_runs: int
    max_deviation: float
    shape_varied: bool
    passed: bool


def check_determinism(launch, runs: int = 20) -> DeterminismReport:
    """Run the same launch repeatedly; any variation is a race.

    ``launch`` takes no arguments and returns a tensor. It must genuinely
    re-run the kernel -- a closure over an already-computed tensor tests
    nothing and will always pass.

    A race that fires one time in a thousand will not appear in twenty runs,
    so **a pass here is weak evidence and a failure is conclusive**. That
    asymmetry is the right one for a check this cheap, but it has to be stated
    rather than implied.
    """
    import torch

    if runs < 2:
        raise ValueError(
            f"check_determinism(runs={runs}) cannot detect variation: it needs "
            "at least 2 runs to have anything to compare. Pass runs>=2."
        )

    first = launch().detach().to("cpu", dtype=torch.float64)
    varying = 0
    max_dev = 0.0
    shape_varied = False

    for _ in range(runs - 1):
        out = launch().detach().to("cpu", dtype=torch.float64)

        if tuple(out.shape) != tuple(first.shape):
            shape_varied = True
            varying += 1
            continue

        # NaN != NaN, so comparing raw values would report variation on every
        # run of a kernel that legitimately produces NaN. Compare finiteness
        # patterns, then magnitudes among the finite entries.
        if not torch.equal(torch.isnan(out), torch.isnan(first)):
            varying += 1
            continue

        deviation = (out - first).abs()
        deviation = deviation[torch.isfinite(deviation)]
        if deviation.numel() and deviation.max().item() > 0.0:
            varying += 1
            max_dev = max(max_dev, float(deviation.max().item()))

    return DeterminismReport(
        runs=runs,
        varying_runs=varying,
        max_deviation=max_dev,
        shape_varied=shape_varied,
        passed=(varying == 0),
    )
