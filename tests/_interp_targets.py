"""Importable kernel targets for the interpreter cross-check tests.

Must be a real module, not a fixture defined inside a test: the subprocess
imports it by dotted path, and `@triton.jit` reads its own source through
`inspect`, so it cannot live in a heredoc, a REPL, or a closure.

Every target has the same signature -- ``(seed: int, device: str) -> Tensor`` --
so the GPU leg and the interpreter leg are the same call with one argument
changed.
"""

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only where triton is absent
    triton = None


if triton is not None:

    @triton.jit
    def _scale_kernel(src, dst, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask, other=0.0) * 3.0, mask=mask)

    @triton.jit
    def _unmasked_kernel(src, dst, n, BLOCK: tl.constexpr):
        """Reads past the end of `src` on the final partial block.

        On the GPU the over-read lands in whatever the allocator left behind and
        usually looks harmless. The interpreter is stricter about the same
        access, so the two legs can disagree -- which is the class of bug this
        check exists to surface.
        """
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        store_mask = offs < n
        tl.store(dst + offs, tl.load(src + offs) * 3.0, mask=store_mask)


def scale(seed: int, device: str):
    """A correct kernel. Both legs must agree exactly."""
    n = 1000
    gen = torch.Generator(device="cpu").manual_seed(seed)
    src = torch.randn(n, generator=gen, dtype=torch.float32).to(device)
    dst = torch.zeros(n, dtype=torch.float32, device=device)
    _scale_kernel[(triton.cdiv(n, 128),)](src, dst, n, BLOCK=128)
    return dst


def unmasked_load(seed: int, device: str):
    """A kernel whose load runs past the end of its input."""
    n = 1000
    gen = torch.Generator(device="cpu").manual_seed(seed)
    src = torch.randn(n, generator=gen, dtype=torch.float32).to(device)
    dst = torch.zeros(n, dtype=torch.float32, device=device)
    _unmasked_kernel[(triton.cdiv(n, 128),)](src, dst, n, BLOCK=128)
    return dst


def not_a_kernel(seed: int, device: str):
    """No Triton at all. Used to prove the harness itself is sound."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(64, generator=gen, dtype=torch.float32) * 2).to(device)


def raises(seed: int, device: str):
    """Fails in the subprocess, so the error path can be tested."""
    raise RuntimeError("deliberate failure for the cross-check tests")
