# PROJECT_STATUS

> **Read this file, `docs/learning_notes.md` and `docs/experiment_log.md` at the start
> of every session before making changes.**

Last updated: 2026-09-05

---

## Current milestone

**M0 — Repository and environment.** Scaffold complete on the local dev machine.
Blocked on renting a GPU box before M4; M1–M3 can proceed locally.

## Completed milestones

| # | Milestone | Status | Notes |
|---|---|---|---|
| M0 | Repository and environment | **Done (local half)** | Structure, config system, seeding, env probe, memory calculator, docs, tests. GPU half pending a rented box. |
| M1 | Base-model exploration | Not started | Next. |

## Current experiment

None. No model has been trained or downloaded yet.

## Next step

1. `python scripts/env_report.py`
2. `python scripts/memory_math.py --preset olmo2-1b --compare`
3. Answer the M0 knowledge check in this file's "Open questions".
4. Then begin M1: load `allenai/OLMo-2-0425-1B` and inspect tokenizer + logits.

## Key results

*(Rule 10: only measured numbers go here. Estimates are labelled as such.)*

| Quantity | Value | Source |
|---|---|---|
| OLMo 2 1B parameter count (derived from architecture) | 1.485 B | `src/utils/memory.py`, agrees with published 1.48 B |
| Est. VRAM, full FT, bf16 + AdamW, bs=1, seq=2048 | ~27 GB | **estimate**, to be measured in M6 |
| Est. VRAM, same with gradient checkpointing | ~25 GB | **estimate** |
| Est. share of VRAM that is optimizer + master + grads | ~71% | **estimate** |
| Measured training throughput | TODO | needs GPU |
| Measured peak VRAM | TODO | needs GPU |

## Problems encountered

1. **No GPU on the development machine.** This project is authored on a MacBook Air
   M3 (16 GB unified memory, MPS only). Real training requires renting a Linux CUDA
   box. Resolution: split the environment into `requirements/base.txt` (cross-platform)
   and `requirements/gpu.txt` (CUDA-only, added per milestone).
2. **PyYAML parses `1e-5` as a string, not a float.** PyYAML implements YAML 1.1,
   whose float resolver requires a decimal point and a signed exponent. A config
   containing `learning_rate: 1e-5` would silently supply a `str` to the optimizer.
   Resolution: `SciFloatLoader` in `src/utils/config.py` backports the YAML 1.2
   resolver. Covered by `tests/test_utils.py`.
3. **Initial OLMo 2 1B preset was wrong** (1.279 B vs published 1.48 B). Cause:
   assumed tied input/output embeddings. The 206 M gap is exactly one untied
   `lm_head` (100352 × 2048). Resolution: `tied_embeddings=False`; verify against the
   real `config.json` in M1.

## Decisions made

| Decision | Rationale |
|---|---|
| Two model tracks: OLMo 2 1B (lab) → modern ~1.5–4B (portfolio) | OLMo is fully open (data, code, intermediate checkpoints), so our results can be compared against an official reference pipeline. |
| Prefer full-parameter fine-tuning for the main experiments | Understanding is the goal; LoRA hides the optimizer-memory story. LoRA becomes an *ablation* (M6), not the default. |
| Hand-roll every algorithm before touching TRL | Rule 3. A `SFTTrainer(...)` call teaches nothing about masking or loss. |
| Dependencies added per milestone, not up front | Isolates breakage to the one package that caused it. |
| Configs are committed files, never shell flags | An experiment must be reproducible six months later. |
| Estimates are always labelled as estimates | Rule 10. `memory_math.py` output is arithmetic, not measurement; M6/M20 replace it with `torch.cuda.max_memory_allocated()`. |

## Open questions

**M0 knowledge check — answer these before starting M1:**

1. A 1B-parameter model needs ~3 GB of VRAM to run inference but ~27 GB to train.
   Where does the extra 24 GB go, and which single component is largest?
2. What is the difference between the CUDA version shown by `nvidia-smi` and the one
   in `torch.version.cuda`? Which must be larger?
3. BF16 and FP16 are both 16 bits. Why is BF16 preferred for training, and what extra
   machinery does FP16 require that BF16 does not?
4. "bf16 mixed precision" still keeps an FP32 copy of the weights. Why — what breaks
   without it?
5. Gradient checkpointing reduces memory. What does it cost, and which of the seven
   memory components does it actually shrink?

**Still undecided:**
- Which GPU to rent (A6000 48 GB vs A100 40/80 GB vs L40S). Decide at M4 using
  `memory_math.py` output plus measured throughput.
- Which instruction dataset for M4 (Tulu 3 SFT mixture is the natural OLMo-aligned
  choice; decide at M4).
- Portfolio base model for M24. Deliberately deferred — evaluate what exists then.
