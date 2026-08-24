"""Record real gradient magnitudes from a real backward pass.

    python scripts/record_gradients.py

`gradient_like()` is a log-uniform guess. It is honest about being a guess, but
a guess cannot settle the question it is used for: whether a shifted exponent
bias would remove the need for loss scaling. That turns entirely on the *tail*
of the real gradient distribution, and a log-uniform spread has no tail.

So this trains a small but genuinely real transformer -- attention, layernorm,
softmax, residuals, backprop through all of it -- and records the magnitude of
every gradient at several points in training. The architecture and the backward
pass are real. The *task* is synthetic, and that limitation is recorded in the
fixture's metadata rather than glossed.

Output: src/shapesandstrides/formats/data/gradient_magnitudes.json

This is a one-off. It is committed so the numbers are reproducible without a GPU
and without re-running training, and so anyone can see exactly what they rest on.
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import date
from pathlib import Path

import torch
import torch.nn as nn

OUT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "shapesandstrides"
    / "formats"
    / "data"
    / "gradient_magnitudes.json"
)

SEED = 0xC0FFEE
VOCAB = 256
SEQ = 64
BATCH = 16
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
STEPS = 400
# Training steps at which to snapshot the gradient distribution. Early, middle
# and late, because the distribution moves as the model converges and the late
# one is where magnitudes are smallest.
SNAPSHOTS = (1, 50, 200, 400)
SAMPLE_PER_SNAPSHOT = 2500
TAIL_PER_SNAPSHOT = 1000
# Exact counts below each threshold, computed on the FULL gradient set.
# A uniform sample cannot answer a question about the tail, so the counts
# that matter are computed before sampling and recorded separately.
THRESHOLDS = [10.0**-e for e in range(4, 15)]

FP16_SMALLEST_NORMAL = 2.0**-14  # 6.10e-5
FP16_SMALLEST_SUBNORMAL = 2.0**-24  # 5.96e-8


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, SEQ, D_MODEL))
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx) + self.pos[:, : idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def synthetic_batch(g: torch.Generator, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A next-token task with real structure, so the model actually learns.

    Tokens follow a noisy repeating motif: learnable, but not trivially so. The
    point is a real optimisation trajectory, not a benchmark.
    """
    period = 7
    base = torch.arange(SEQ + 1) % period
    idx = base.repeat(BATCH, 1).clone()
    idx = idx * (VOCAB // period)
    noise = torch.randint(0, 8, (BATCH, SEQ + 1), generator=g)
    idx = (idx + noise) % VOCAB
    return idx[:, :-1].to(device), idx[:, 1:].to(device)


def main() -> int:
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    rng = random.Random(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyTransformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params:,}  steps={STEPS}")

    snapshots: list[dict] = []
    for step in range(1, STEPS + 1):
        x, y = synthetic_batch(g, device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()

        if step in SNAPSHOTS:
            mags: list[float] = []
            per_param: dict[str, float] = {}
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                flat = p.grad.detach().abs().flatten().to(torch.float32).cpu()
                nz = flat[flat > 0]
                if nz.numel():
                    per_param[name] = float(nz.median())
                mags.extend(flat.tolist())

            nonzero = sorted(m for m in mags if m > 0)
            # A uniform sample, representative of the bulk of the distribution.
            sample = (
                rng.sample(nonzero, SAMPLE_PER_SNAPSHOT)
                if len(nonzero) > SAMPLE_PER_SNAPSHOT
                else list(nonzero)
            )
            # And the extreme low tail, kept separately. Uniform sampling
            # cannot preserve a tail this thin, and the tail is the whole
            # question -- but a tail sample is not representative, so the two
            # are never mixed.
            tail = nonzero[:TAIL_PER_SNAPSHOT]
            below_normal = sum(1 for m in nonzero if m < FP16_SMALLEST_NORMAL)
            below_sub = sum(1 for m in nonzero if m < FP16_SMALLEST_SUBNORMAL)
            snapshots.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "gradients_total": len(mags),
                    "gradients_nonzero": len(nonzero),
                    "sampled": len(sample),
                    "tail_kept": len(tail),
                    "min": nonzero[0],
                    "max": nonzero[-1],
                    "median": nonzero[len(nonzero) // 2],
                    "fraction_below_fp16_smallest_normal": below_normal / len(nonzero),
                    "fraction_below_fp16_smallest_subnormal": below_sub / len(nonzero),
                    # Exact, computed on every gradient before sampling.
                    "count_below": {
                        f"{t:.0e}": sum(1 for m in nonzero if m < t)
                        for t in THRESHOLDS
                    },
                    "per_parameter_median": per_param,
                    "magnitudes": sample,
                    "smallest": tail,
                }
            )
            print(
                f"  step {step:4}  loss {loss.item():.4f}  "
                f"min {min(nonzero):.3e}  median {sorted(nonzero)[len(nonzero)//2]:.3e}  "
                f"max {max(nonzero):.3e}  "
                f"below fp16 normal {below_normal/len(nonzero):.1%}  "
                f"below fp16 subnormal {below_sub/len(nonzero):.2%}"
            )

        opt.step()

    payload = {
        "what": "Magnitudes of real gradients from a real backward pass.",
        "recorded": date.today().isoformat(),
        "honest_limitations": [
            "The task is synthetic: a noisy repeating-motif next-token problem, "
            "not a real corpus. The architecture and the backward pass are real; "
            "the data is not.",
            "The model is small (see architecture below). Deeper networks "
            "attenuate gradients further, so a large model's tail reaches lower "
            "than this one's.",
            "No loss scaling and no mixed precision were used, so these are the "
            "true fp32 gradients rather than what a mixed-precision run would "
            "produce.",
            "Magnitudes are sampled, not exhaustive. `magnitudes` is a seeded "
            "uniform sample, representative of the bulk. `smallest` is the "
            "extreme low tail, which uniform sampling cannot preserve -- it is "
            "NOT representative and must not be mixed with the bulk. Exact "
            "counts below each threshold in `count_below` are computed on every "
            "gradient before any sampling, so those are authoritative.",
        ],
        "architecture": {
            "type": "pre-norm transformer, causal",
            "layers": N_LAYERS,
            "d_model": D_MODEL,
            "heads": N_HEADS,
            "seq_len": SEQ,
            "vocab": VOCAB,
            "batch": BATCH,
            "parameters": n_params,
            "optimizer": "AdamW lr=3e-4",
            "loss": "cross entropy",
        },
        "seed": SEED,
        "device": device,
        "torch_version": torch.__version__,
        "reference_thresholds": {
            "fp16_smallest_normal": FP16_SMALLEST_NORMAL,
            "fp16_smallest_subnormal": FP16_SMALLEST_SUBNORMAL,
        },
        "snapshots": snapshots,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"\nwrote {OUT} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
