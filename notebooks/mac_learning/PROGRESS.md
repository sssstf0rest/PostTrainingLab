# Course and learner progress

Updated: 2026-10-31, Asia/Singapore. Full priority order, prerequisites, repository
files and future lesson contracts: [COURSE.md](COURSE.md).

## Delivery record

| Lessons | Generated | Execution | Learner understanding |
|---|---|---|---|
| 00: setup, route choice, tiny update, memory | Python source and notebook | See [VERIFICATION.md](VERIFICATION.md) | Not yet assessed |
| 01: tokens, causal LM, loss, gradients, optimizer | Python source and notebook | See [VERIFICATION.md](VERIFICATION.md) | Not yet assessed |
| 02–13 | Planned; no notebook files yet | Not run | Not yet assessed |

Current lesson: **00**, section **1 — what changes when a model learns?**
Next action: explain whether retrieval of a new document changes model parameters,
then run the setup cell and check your kernel/device. Study 01 after the 00 route
and gradient examples; we will build lesson 02 when progressing to supervision.

## Evidence of understanding

No new explanations, manual calculations, or independent code modifications have
been demonstrated in this learning track yet. Notebook execution verifies code,
not mastery. Historical knowledge-check notes in `docs/learning_notes.md` remain
historical evidence; they do not automatically certify current recall.

| Concept | Prediction / learner explanation | Evidence | Status |
|---|---|---|---|
| Training changes parameters; retrieval changes supplied evidence | Pending | No response yet | Unassessed |
| DPO versus reward-model-plus-PPO; optimizer versus reward source | Pending | No response yet | Unassessed |
| Token position, causal shift, masks and loss denominator | Pending | No response yet | Unassessed |
| Softmax/CE gradient and chain rule to model weights | Pending | No response yet | Unassessed |
| `.eval()` versus no-grad; reproducibility versus identical samples | Pending | No response yet | Unassessed |
| Strong baselines, paired seeds, uncertainty and generalization limits | Pending | No response yet | Unassessed |

## Questions and misconceptions to revisit

- Awaiting the learner's first answer; no new misconception attributed to the learner.
- Review the historical oversimplifications listed in [AUDIT.md](AUDIT.md), especially
  a mandatory SFT→RM→DPO→PPO chain, universal memory rules, MPS exclusions, and
  claims that post-training cannot add knowledge.
- Choose the intended consumer-intelligence deployment scenario before lesson 02's
  entity/time split exercise; use synthetic data and clearly stated assumptions.
- Optional pretrained/quantized extensions are not selected or validated. Check exact
  model size, license, OS and backend before proposing an enabled extension.

## Session workflow

For each short section: predict → run → modify one factor → explain. Record the
actual answer here, then a precise correction if needed. Only mark a concept
“demonstrated” after a correct explanation or independent task; revisit it in a
later lesson before calling it retained.
