"""Never use more than 30% of this machine's computing power for training.

Import this module before numpy, torch, lightgbm or scikit-learn:

    import fam.compute  # first, so the limits apply to everything after it

Why it must come first: the maths libraries under numpy, scikit-learn and
torch (OpenMP, MKL, OpenBLAS) read their thread count from environment
variables once, when they are first loaded. Setting them later does nothing.

What is capped, and how exactly:
  - CPU: 30% of logical cores (rounded down, at least 1). Exact: every
    library is told to use at most CPU_THREADS threads.
  - NVIDIA GPU memory: GPU_SHARE of it, via torch. Exact for torch's own memory.
    A separate setting from SHARE, so GPU memory can be raised (e.g. for LLM
    training) without also raising CPU threads and the RAM budget.
  - GPU compute: can NOT be hard-capped from Python; a running kernel uses
    the whole GPU. Only memory can be limited.
  - RAM: not hard-capped (Windows has no simple per-process limit). Instead
    batch sizes are kept small; MAX_RAM_GB is the budget to stay under.
"""

import os

SHARE = 0.30      # CPU threads and the RAM budget
GPU_SHARE = 0.30  # NVIDIA GPU memory

CPU_THREADS = max(1, int(os.cpu_count() * SHARE))

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = str(CPU_THREADS)

import psutil  # noqa: E402  (imported after the env vars on purpose)
import torch  # noqa: E402

MAX_RAM_GB = psutil.virtual_memory().total * SHARE / 1024**3

torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(CPU_THREADS)
except RuntimeError:
    pass  # already set: only allowed once per process

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(GPU_SHARE)
