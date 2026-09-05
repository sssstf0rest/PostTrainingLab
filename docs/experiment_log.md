# Experiment Log

Append-only. One entry per run. **Rule 10: every number here is measured. Anything
not yet measured is `TODO`, never a plausible-looking guess.**

Entry template:

```
## <experiment-id>
Date        :
Milestone   :
Question    : the ONE thing this run is meant to answer
Config      : experiments/<id>/config.yaml
Git commit  :
Hardware    : GPU model, count, VRAM
Command     :
Result      : measured numbers only
Conclusion  : what this changes about the next run
```

---

## m0-env
Date        : 2026-09-05
Milestone   : M0
Question    : What hardware do we have, and what will actually fit on it?
Config      : n/a
Git commit  : (M0 scaffold commit)
Hardware    : Apple M3, 8 cores, 16 GB unified memory, MPS. **No CUDA.**
Command     : `python scripts/env_report.py`

Result (measured — environment facts):
- Python 3.12.7, PyTorch 2.9.1, macOS 26.6.2 arm64
- `torch.cuda.is_available()` = False; `torch.backends.mps.is_available()` = True
- 79.3 GB free disk
- OLMo 2 1B parameter count derived from architecture: **1.485 B**, agreeing with the
  published 1.48 B (this validated the architecture assumptions, after an initial
  1.279 B result exposed a wrong embedding-tying assumption)

Result (estimated — arithmetic, NOT measurement):
| config (bs=1, seq=2048) | est. total VRAM |
|---|---|
| olmo2-1b full FT, bf16 mixed, AdamW | ~27.1 GB |
| olmo2-1b full FT + gradient checkpointing | ~25.1 GB |
| olmo2-7b full FT, bf16 mixed, AdamW | ~120.5 GB |
| olmo2-7b full FT + gradient checkpointing | ~112.3 GB |

Conclusion:
- No training happens on this machine. M1–M3 (inspection, data pipeline, a manual
  loop on a tiny model) run locally; M4 onward needs a rented CUDA box.
- A single 40–48 GB GPU should hold full-parameter SFT of OLMo 2 1B with headroom for
  batch size > 1. Confirm by measurement before committing to a rental tier.
- OLMo 2 7B full fine-tuning is out of reach on one GPU — it would need ZeRO-3 across
  several. Reinforces choosing the 1B as the lab model.

Open: these estimates must be checked against `torch.cuda.max_memory_allocated()` in
M6. If the activation constant (`ACT_ELEMS_PER_HIDDEN = 18`) is badly wrong, correct
it in `src/utils/memory.py` and note the correction here.

---

## m1-base-exploration
Date        : TODO
Result      : TODO
