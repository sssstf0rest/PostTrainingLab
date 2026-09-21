#!/usr/bin/env python3
"""Print (and optionally save) a full environment report.

Run this FIRST on every new machine, especially each freshly rented GPU box.
It answers: what hardware do I have, what can it do, and how much can I fit?

    python scripts/env_report.py
    python scripts/env_report.py --save experiments/m0-env/env.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import collect_env  # noqa: E402
from src.utils.memory import GB, PRESETS, estimate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=Path, default=None, help="write the report as JSON")
    args = ap.parse_args()

    env = collect_env()
    W = 72

    print("=" * W)
    print("  OpenPostTrain Lab — environment report")
    print("=" * W)
    print(f"  Python          : {env.python_version}")
    print(f"  Platform        : {env.platform}")
    print(f"  Architecture    : {env.machine}   ({env.cpu_count} CPU cores)")
    print(f"  Free disk       : {env.disk_free_gb} GB")
    print(f"  PyTorch         : {env.torch_version}")
    print(f"  Accelerator     : {env.accelerator.upper()}")

    if env.accelerator == "cuda":
        print(f"  Driver          : {env.driver_version}  (supports CUDA <= {env.driver_cuda_version})")
        print(f"  torch built for : CUDA {env.torch_cuda_version}")
        print(f"  bf16 supported  : {env.bf16_supported}")
        print("-" * W)
        for g in env.gpus:
            print(f"  GPU {g.index}: {g.name}")
            print(f"         {g.total_memory_gb} GB VRAM   compute capability {g.capability}")
    else:
        print(f"  CUDA available  : {env.cuda_available}")
        print(f"  MPS available   : {env.mps_available}")
        print(f"  bf16 supported  : {env.bf16_supported}")
        for g in env.gpus:
            # On Apple Silicon this is not a separate VRAM pool: CPU and GPU share one
            # physical memory, and Metal caps the working set below total system RAM.
            print(f"  {g.name}")
            print(f"         {g.total_memory_gb} GB allocatable (unified, shared with the OS)")

    if env.notes:
        print("-" * W)
        for n in env.notes:
            print(f"  NOTE: {n}")

    # --- What can this box actually train? -------------------------------------
    vram = env.gpus[0].total_memory_gb if env.gpus else None
    print("=" * W)
    print("  Estimated VRAM for full-parameter SFT  (bf16 mixed precision, AdamW)")
    print("  ESTIMATES from first principles — measured for real in M6/M20.")
    print("=" * W)
    print(f"  {'model':<12} {'seq':>5} {'bs':>3} {'grad-ckpt':>10} {'est. GB':>9}   {'fits?':>6}")
    print("  " + "-" * (W - 4))
    for key in ("olmo2-1b", "olmo2-7b"):
        shape = PRESETS[key]
        for ckpt in (False, True):
            b = estimate(shape, batch_size=1, seq_len=2048, gradient_checkpointing=ckpt)
            gb = b.total / GB
            verdict = "-" if vram is None else ("yes" if gb < vram * 0.92 else "OOM")
            print(f"  {key:<12} {2048:>5} {1:>3} {str(ckpt):>10} {gb:>9.1f}   {verdict:>6}")

    print("=" * W)
    print("  Component breakdown: olmo2-1b, seq=2048, batch=1, no checkpointing")
    print("-" * W)
    b = estimate(PRESETS["olmo2-1b"], batch_size=1, seq_len=2048)
    for name, val, desc in b.rows():
        print(f"  {name:<26} {val / GB:>7.2f} GB   {desc}")
    print(f"  {'TOTAL':<26} {b.total / GB:>7.2f} GB")
    print("=" * W)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(env.to_dict(), indent=2))
        print(f"  saved -> {args.save}")


if __name__ == "__main__":
    main()
