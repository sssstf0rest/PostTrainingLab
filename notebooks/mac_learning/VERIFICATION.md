# Execution and verification evidence

Verified **2026-10-31, Asia/Singapore**, on the actual **Apple M3 MacBook Air,
16 GiB unified memory**, macOS **27.0.1**, arm64. This was not a Linux/cloud run.
Audited upstream branch `main`, commit `5b21b9575e14116ec2be2fe9cd16840f430a02b6`.
Course additions are uncommitted; each execution record includes the exact Python
source SHA-256, rather than implying the upstream commit contains these additions.

## Environment and execution

Python **3.12.15**, PyTorch **2.15.0**, NumPy **2.5.4**, Matplotlib **3.11.2**,
JupyterLab **4.6.5**, nbclient **0.11.0**, nbformat **5.11.1**, ipykernel **7.4.0**.
Exact resolved dependencies: [requirements.lock.txt](requirements.lock.txt).
Native MPS is built and available. CPU is still the required default.

Both notebooks were built from their editable percent-format sources and executed
from start to finish in **new kernels**, using the teaching environment interpreter
explicitly. `QUICK_MODE=True`, float32 training/inspection, no previous checkpoint
or prior notebook state. The small float64 finite-difference check deliberately
runs on CPU even in the MPS session. Some introductory standalone calculations and
the shape-accounting experiment also intentionally stay CPU.

| Lesson | Selected backend | Code cells executed | Full kernel wall time | Maximum sampled kernel RSS | Result |
|---|---|---:|---:|---:|---|
| 00 | CPU | 6/6 | 2.447 s | 349.3 MiB | Passed |
| 00 | MPS | 6/6 | 2.422 s | 489.6 MiB | Passed |
| 01 | CPU | 20/20 | 2.745 s | 354.0 MiB | Passed |
| 01 | MPS | 20/20 | 8.986 s | 523.5 MiB | Passed |

The canonical notebooks retain the CPU outputs. Separate MPS notebooks remain in
ignored `.mps_runs/` in this workspace. Portable JSON evidence, including timestamps,
source hashes, sampling interval/count and limitations, is under [verification/](verification/).
No silent MPS-to-CPU fallback variable was enabled by the runner.

These are **single observed runs**, not speed benchmarks or guarantees. Full wall
time includes kernel startup/shutdown and plotting; per-experiment timings inside
lesson 01 have different boundaries. Import/font/kernel caches, system load and
thermal behavior can change timings. MPS synchronization is used around the lesson
01 timed training runs. No broad CPU-versus-MPS performance conclusion follows.

RSS was polled every 50 ms through psutil. It is the maximum **observed kernel
resident set**, not an exact peak, whole-system memory, or complete MPS allocation.
It excludes the runner, browser, other apps, and some driver/shared memory; brief
spikes can be missed. The separate printed tensor-storage counts are exact payload
arithmetic for those tensors, not process peaks. A 6–8 GiB engineering ceiling does
not imply every workload below it is safe. These defaults stay far below it.

## What passed

**Notebook format and pairing:** nbformat schema validation, stable cell IDs,
correct teaching kernel metadata, current source hashes, code-cell source agreement
with `scripts/to_ipynb.py` parsing, all code cells executed, zero error outputs.
Both the route-map plot and the learning-curve plot were visually inspected.

**Lesson 00:** analytic two-logit CE gradient, a probability increase after SGD,
finite loss, tensor-size equations and doubling ratios, and a real linear-layer
forward/backward/AdamW update. Selected-backend loss/weights agree with CPU within
stated tolerances. The branching map separates DPO from reward-model-plus-PPO and
separates optimization methods from reward sources.

**Lesson 01:** tokenizer round trip within its declared alphabet, embedding/one-hot
equivalence, stable manual/framework NLL agreement, normalized probabilities,
autoregressive product/log-sum agreement, zero upper attention triangle and causal
suffix-invariance, exactly one explicit causal shift, manual CE gradient sign,
CPU float64 chain-rule/finite-difference agreement, scalar manual/PyTorch AdamW
agreement, clipping arithmetic, finite gradients and parameter changes, reset
invariants, eval/no-grad distinction, disjoint training/validation strings,
train-fitted bigram and grammar-rule baselines, finite losses for all bounded
training candidates, and bounded generation. Dataset generation has a finite attempt
budget and rejects impossible unique split sizes.

The `.py` lesson 01 source was also run as a CPU/Agg script during authoring.
The final fresh-kernel runs, after source review and bounded-data safeguards, are
the authoritative evidence above. An intermediate source-hash mismatch during
concurrent authoring was detected before execution; the final source was rebuilt
and all required paths then passed.

## Observed toy experiment, with its failure case

Lesson 01 uses **5,592 trainable parameters**, 128 training strings, 64 disjoint
validation strings, two paired initialization/minibatch seeds, 120 updates per
candidate and learning rates 0.0001 versus 0.01. It is a synthetic cue-copying task,
not a pretrained LLM or instruction-SFT benchmark.

CPU mean validation NLL was **2.0650** at 0.0001 and **1.1136** at 0.01. However,
high-learning-rate cue-recall accuracy was only **0.5000 / 0.46875** across the two
seeds. The known-grammar rule achieved **1.0** cue accuracy and **1.0012** NLL, and
the train-fitted bigram achieved **1.5684** NLL. The bounded sample generated the
wrong cue (`x` where `y` was required), followed by EOS. MPS results were close but
not identical: high-LR cue accuracy **0.515625 / 0.46875**. This difference is not
an established device-quality effect.

The result supports a learning-rate effect on this validation loss under this
small fixed budget. It does not support production gains, reliable task mastery,
statistical significance, or business uplift. The task diagnostic and simple rule
show why a decreasing average loss can conceal failure on the actual objective.
No untouched final test was used; validation comparisons are for teaching/tuning.

## Existing repository tests

A temporary `uv` environment with pytest, torch 2.15.0 and PyYAML 6.0.3 ran:

```bash
uv run --no-project --python .venv/bin/python --with pytest --with torch==2.15.0 --with pyyaml==6.0.3 python -m pytest tests/test_utils.py tests/test_sft_masking.py -ra
```

**7 passed; 1 module skipped.** The skip reason was missing `transformers`.
Independently, the target implementation `src/data/sft.py` is absent at the audited
commit. The eighteen M2 contract tests were therefore **not executed**. This result
does not certify an implemented SFT data pipeline. The temporary test tooling did
not add server/model dependencies to the teaching environment.

## Boundaries and reproducibility

No pretrained checkpoints, datasets, paid APIs, cloud/server GPUs, CUDA packages,
or proprietary data were used by the new lessons. Hugging Face offline flags were
set; source code uses only local synthetic data. Global network access was **not**
blocked, so this is an offline-capable path, not a network-isolation test.

The historical large OLMo notebooks were read, not freshly executed. No original
SFT/DPO/PPO/GRPO/agent milestone was marked complete. Lessons 02–13, optional real
pretrained-model adapters, quantized training, and MLX extensions remain planned.
No current learner mastery has been inferred from these execution results.

Use [README.md](README.md) for exact rebuild/run steps. Official API references
checked against this version include [MPS](https://docs.pytorch.org/docs/2.15/notes/mps.html),
[MPS synchronization and allocation](https://docs.pytorch.org/docs/2.15/mps.html),
[AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html), and
[autograd modes](https://docs.pytorch.org/docs/2.15/notes/autograd.html#locally-disabling-gradient-computation).
