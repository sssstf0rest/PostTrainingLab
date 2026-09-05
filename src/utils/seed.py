"""Determinism and reproducibility.

WHY THIS FILE EXISTS
--------------------
In M6 we will ask questions like "is lr=5e-6 better than lr=1e-5?". If two runs with
IDENTICAL settings already differ by more than the effect we are measuring, the
answer is noise and the experiment is worthless. Controlling randomness is what makes
an ablation an experiment instead of an anecdote.

WHERE RANDOMNESS ENTERS LLM TRAINING
------------------------------------
  1. Data order          — which examples land in which batch (torch/numpy RNG)
  2. Data shuffling seed — dataset .shuffle()
  3. Dropout             — usually 0.0 in modern LLM fine-tuning, so often a non-issue
  4. Weight init         — only for newly added heads (reward-model head, LoRA B matrix)
  5. GPU kernel nondeterminism — atomics/reduction order differ run to run
  6. Generation sampling — temperature/top-p during RL rollouts and evaluation

Items 1-4 are cheap to control. Item 5 costs real throughput. Item 6 must be fixed
for evaluation (or you cannot compare checkpoints) but must NOT be fixed for RL
rollouts (you need diverse samples for GRPO to have any signal at all).
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed every RNG that affects training.

    Args:
        seed: the seed. Record it in the experiment config; a run without a recorded
            seed is not reproducible.
        deterministic_algorithms: force bitwise-reproducible CUDA kernels. This is
            SLOW (some ops fall back to non-parallel implementations) and some ops
            have no deterministic variant and will raise. Use it to debug a specific
            discrepancy, not for production runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)              # CPU generator
    torch.cuda.manual_seed_all(seed)     # every CUDA device (no-op if none)

    if deterministic_algorithms:
        # cuBLAS reductions are nondeterministic unless given a fixed workspace.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
