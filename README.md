# crucible

Honest correctness and timing for Triton kernels.

Answers two questions about a GPU kernel: is it correct across a wide space of shapes and dtypes, and is it actually faster than PyTorch — with a confidence interval, against a baseline re-measured in the same run.

Runs on your own hardware. Kernel source never leaves the machine.

## Status

Pre-alpha. Nothing here is usable yet.

The only code currently present is `scripts/probe_host.py`, a throwaway utility that classifies whether a given GPU host can produce trustworthy measurements — that is, whether it can pin its clocks and hold them under load.

```bash
PROBE_LABEL=my-host python scripts/probe_host.py -o my-host.json
```

It verifies every `nvidia-smi` control write by reading the value back, because `nvidia-smi` exits 0 when it refuses one. It restores every setting it touches before exiting.

## Requirements

Python 3.11+, an NVIDIA GPU, PyTorch with CUDA.
