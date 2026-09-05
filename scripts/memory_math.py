#!/usr/bin/env python3
"""Where does the VRAM go? An explicit training-memory calculator.

Use this BEFORE launching any run, and again whenever you hit CUDA OOM, to see
which component to attack.

    python scripts/memory_math.py --preset olmo2-1b --batch-size 4 --seq-len 2048
    python scripts/memory_math.py --preset olmo2-1b --batch-size 4 --grad-checkpointing
    python scripts/memory_math.py --preset olmo2-1b --lora          # PEFT comparison
    python scripts/memory_math.py --preset olmo2-1b --compare       # sweep the knobs

All numbers are ESTIMATES (see src/utils/memory.py). M6 replaces them with
measurements from torch.cuda.max_memory_allocated().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.memory import GB, PRESETS, estimate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="olmo2-1b", choices=sorted(PRESETS))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd_momentum", "sgd", "adafactor"])
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--no-flash", action="store_true", help="materialize the [seq,seq] attention matrix")
    ap.add_argument("--lora", action="store_true", help="train ~0.5%% of params instead of all")
    ap.add_argument("--inference", action="store_true")
    ap.add_argument("--compare", action="store_true", help="sweep configurations instead")
    args = ap.parse_args()

    shape = PRESETS[args.preset]
    print(f"\nModel: {shape.name}")
    print(f"  hidden={shape.hidden}  layers={shape.layers}  vocab={shape.vocab}  "
          f"intermediate={shape.intermediate}")
    print(f"  derived parameter count: {shape.n_params() / 1e9:.3f}B\n")

    if args.compare:
        print(f"  batch={args.batch_size}  seq={args.seq_len}\n")
        configs = [
            ("full FT, fp32, no ckpt",     dict(param_dtype="fp32", mixed_precision=False)),
            ("full FT, bf16 mixed",        dict()),
            ("full FT, bf16 + grad ckpt",  dict(gradient_checkpointing=True)),
            ("full FT, bf16 + ckpt + SGD", dict(gradient_checkpointing=True, optimizer="sgd")),
            ("LoRA 0.5%, bf16",            dict(trainable_fraction=0.005)),
            ("LoRA 0.5%, bf16 + ckpt",     dict(trainable_fraction=0.005, gradient_checkpointing=True)),
            ("inference only",             dict(inference_only=True)),
        ]
        print(f"  {'configuration':<28} {'total GB':>9}")
        print("  " + "-" * 40)
        for label, kw in configs:
            b = estimate(shape, batch_size=args.batch_size, seq_len=args.seq_len, **kw)
            print(f"  {label:<28} {b.total / GB:>9.2f}")
        print()
        return

    b = estimate(
        shape,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        param_dtype=args.dtype,
        mixed_precision=args.dtype != "fp32",
        optimizer=args.optimizer,
        gradient_checkpointing=args.grad_checkpointing,
        flash_attention=not args.no_flash,
        trainable_fraction=0.005 if args.lora else 1.0,
        inference_only=args.inference,
    )
    print(f"  batch={args.batch_size}  seq={args.seq_len}  dtype={args.dtype}  "
          f"optim={args.optimizer}  grad_ckpt={args.grad_checkpointing}  lora={args.lora}")
    print(f"  tokens per forward pass: {args.batch_size * args.seq_len:,}\n")
    print(f"  {'component':<26} {'GB':>8}   share / meaning")
    print("  " + "-" * 74)
    for name, val, desc in b.rows():
        print(f"  {name:<26} {val / GB:>8.2f}   {desc}")
    print("  " + "-" * 74)
    print(f"  {'TOTAL':<26} {b.total / GB:>8.2f} GB\n")


if __name__ == "__main__":
    main()
