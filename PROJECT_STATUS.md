# PROJECT_STATUS

> **Read this file, `docs/learning_notes.md` and `docs/experiment_log.md` at the start
> of every session before making changes.**

Last updated: 2026-09-07

---

## Current milestone

**M0 — Repository and environment.** Scaffold done. Training box secured and probed
(shared company server, 8 × L20X). Python env build in progress: `.venv` created, torch
cu128 installing. Knowledge check answered. Remaining: run `env_report.py` on real CUDA
and replace the estimated numbers below with measurements.

## Completed milestones

| # | Milestone | Status | Notes |
|---|---|---|---|
| M0 | Repository and environment | **In progress** | Local half done (structure, config, seeding, env probe, memory calculator, docs, tests). GPU half: box probed, venv created, dependency install running. |
| M1 | Base-model exploration | Not started | Next, once torch imports. |

## Hardware (measured 2026-09-07)

Shared company server, `47.101.174.157`, project at `/home/haosheng/workSpace/PostTrainingLab`.

| Item | Value |
|---|---|
| GPUs | 8 × NVIDIA L20X, ~140 GB each, compute capability **8.9** (native bf16) |
| Driver / max CUDA | 570.172.08 / **12.8** |
| CPU / RAM / disk | 192 cores / 2 TB / 1.1 TB free |
| OS / Python | Ubuntu 22.04.5 / 3.10.12 |
| **Shared with** | 13 colleagues; other jobs run under the same UID. Pin `CUDA_VISIBLE_DEVICES`. |

This is far larger than M0 assumed. Consequences: **OLMo 2 7B full fine-tuning fits on a
single GPU** (est. ~120 GB), so it is no longer out of reach; and **M20 (DDP / ZeRO)
is directly runnable** without a second rental.

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

1. ~~**No GPU on the development machine.**~~ **Resolved 2026-09-07** — a shared company
   Linux CUDA server is now available (see Hardware above). The
   `requirements/base.txt` + `requirements/gpu.txt` split still stands and is still useful.
4. **The SSH proxy kills long connections** (`recv() failed, 10054`, exit 255) while the
   remote command keeps running, so a naive retry starts a *second* concurrent job. This
   actually happened: two pip processes raced on one virtualenv. Resolution: every long
   job runs under `tmux`, polled by short reconnects. Applies to all M4+ training runs.
5. **Slow package egress.** Downloads from `download.pytorch.org` measured at 25–50 KB/s
   during setup (earlier the same link ran at ~2.5 MB/s), making the ~3 GB CUDA wheel
   chain take hours. `pypi.org/simple` and `github.com` HTML endpoints time out entirely,
   while `files.pythonhosted.org`, `download.pytorch.org`, `huggingface.co` and
   git-over-HTTPS all work — so pip and git function normally.
6. **`pkill -f <pattern>` is unsafe here.** It matches the invoking shell *and* same-UID
   jobs belonging to colleagues. Use exact PIDs or tmux session names.
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
| **Four docs collapsed to three: `interview_notes.md` removed** (2026-09-07) | Maintaining a separate Q&A file duplicated `learning_notes.md`. Interview-ready explanations now live inline in the learning notes, which is the file actually read at session start. Recoverable from git history if ever wanted. |
| **All long jobs run under tmux on the server** (2026-09-07) | The corporate SSH proxy kills long connections (exit 255) while the remote job keeps running — naive retries spawn concurrent duplicates. See `docs/learning_notes.md`. |
| **GPUs are pinned per run via `CUDA_VISIBLE_DEVICES`** (2026-09-07) | The box is shared with 13 colleagues, with other people's jobs under the same UID. No run may grab all 8 devices. |

## Open questions

**M0 knowledge check — ANSWERED 2026-09-07.** All five worked through; the corrections
and the reasoning behind them are written up in `docs/learning_notes.md`. Misconceptions
that were caught and are now documented:

1. Optimizer state (40.8%), not activations, is the largest training-memory component —
   mixed-precision AdamW costs **16 bytes/parameter**. KV cache is an *inference* cost,
   absent from training because of teacher forcing.
2. The CUDA direction rule was backwards: `nvidia-smi` reports the **driver's** maximum
   supported CUDA and must be **≥** `torch.version.cuda` (the wheel's build toolkit).
3. BF16 vs FP16 range/precision understood; the missing piece was that FP16 requires a
   **loss scaler** (`GradScaler`) and BF16 does not.
4. The FP32 master copy exists because bf16's ~0.4% relative resolution rounds typical
   updates away entirely — training stalls silently, with no error.
5. Gradient checkpointing costs ~20–40% throughput and shrinks **activations only**:
   2.25 → 0.27 GB (88% of that row), but total only 27.11 → 25.12 GB (7%), because
   activations are a small share at batch 1 / seq 2048.

**Still undecided:**
- Which GPU to rent (A6000 48 GB vs A100 40/80 GB vs L40S). Decide at M4 using
  `memory_math.py` output plus measured throughput.
- Which instruction dataset for M4 (Tulu 3 SFT mixture is the natural OLMo-aligned
  choice; decide at M4).
- Portfolio base model for M24. Deliberately deferred — evaluate what exists then.
