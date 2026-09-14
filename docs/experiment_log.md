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

## m0-env-gpu
Date        : 2026-09-07
Milestone   : M0
Question    : What is the real training box, and do the M0 estimates survive contact with it?
Config      : n/a
Git commit  : 9187d7f (docs updated after)
Hardware    : shared company server, 8 x NVIDIA H200, 139.8 GB each, 192 CPU cores, 2 TB RAM
Command     : `CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python scripts/env_report.py --save experiments/m0-env/env.json`

Result (measured):
- Python 3.11.15, PyTorch 2.11.0+cu128, Ubuntu 22.04.5, glibc 2.35
- Driver 570.172.08 supports CUDA <= 12.8; torch built for CUDA 12.8 -> **matched**
  (this is the M0 rule in practice: drivers max wheels build version)
- `torch.cuda.is_available()` = True, 8 devices, `is_bf16_supported()` = True
- 139.8 GB VRAM per GPU, 132 SMs, 1110 GB free disk
- **Compute capability conflict, resolved**: `nvidia-smi` reports the name "L20X" and
  compute capability **8.9**; `torch.cuda.get_device_properties()` reports **9.0**.
  The machine owner confirmed these are **H200s**, which matches torch (132 SMs,
  139.8 GB, Hopper). The `nvidia-smi` name and capability fields are misleading here.
  **Rule: trust torch** — it reads the CUDA driver API and determines kernel selection.
  Consequence: FP8 and Hopper-class kernels are available.
- **Measured bf16 dense matmul: ~198 TFLOP/s** (8192³, 20 iterations, 5.6 ms each) on an
  idle GPU. A full H200 specs ~990 TFLOP/s dense bf16, so realised throughput here is
  roughly a fifth of that. Planning consequence: **VRAM is abundant and compute is the
  scarce resource** — favour large batches, and expect wall-clock rather than memory to
  limit how many experiments we run. Peak allocation for the test was 0.72 GB, and
  autograd was verified (finite gradients through a bf16 backward).
- Note on the report output: it lists 6 GPUs, not 8, because `CUDA_VISIBLE_DEVICES`
  was pinned to idle devices. CUDA **renumbers** visible devices from 0, so report
  "GPU 0" is physical GPU 1. Colleagues held physical GPUs 0 and 2 at the time.

Result (estimates unchanged, still arithmetic not measurement):
| config (bs=1, seq=2048) | est. total VRAM | fits on 139.8 GB? |
|---|---|---|
| olmo2-1b full FT, bf16 mixed | 27.11 GB | yes |
| olmo2-1b + gradient checkpointing | 25.12 GB | yes |
| olmo2-7b full FT, bf16 mixed | 120.5 GB | **yes** |
| olmo2-7b + gradient checkpointing | 112.3 GB | yes |

Environment note: the declared `.venv` + `requirements/base.txt` path was abandoned for
now. Package egress is throttled (download.pytorch.org measured at 25-110 kB/s, all
Chinese mirrors blocked, ~3 GB of CUDA wheels would take >10 h), while huggingface.co
runs at **11.3 MB/s**. Since an existing conda env already had the exact target build,
the environment was created by `conda create -n ptl --clone maeb` — a local copy, zero
download. Interpreter: `/home/haosheng/.miniconda3/envs/ptl/bin/python` (Python 3.11.15).
The abandoned `.venv` and the 1.5 GB of half-downloaded CUDA wheels were deleted.
`requirements/base.txt` no longer describes the live environment — the `ptl` env ships
transformers 5.14.1, datasets 5.0.0, accelerate 1.14.0, peft 0.19.1, numpy 2.4.4, which
are all MAJOR versions ahead of what most reference recipes assume. Reconcile in M1.

Conclusion:
- M0 estimates are compatible with the hardware; **OLMo 2 7B full fine-tuning now fits on
  a single GPU**, which M0 had ruled out. 1B stays the lab model for iteration speed.
- M20 (DDP / ZeRO) is runnable on this box without extra hardware.
- Still unmeasured: real peak VRAM and throughput. `max_memory_allocated()` in M6 is what
  validates the 27.11 GB figure and the `ACT_ELEMS_PER_HIDDEN = 18` constant.

---

## m0-close-tied-embeddings
Date        : 2026-09-08
Milestone   : M0 (closing) / M1 (opening)
Question    : Does OLMo 2 1B really leave input and output embeddings untied, as the
              M0 parameter derivation assumed?
Config      : n/a
Model       : allenai/OLMo-2-0425-1B @ a1847dff35000b4271fa70afc5db10fd29fedbdf
Hardware    : CPU only (config + checkpoint header inspection, no GPU)

Result (measured) - THREE independent confirmations, all agreeing:

1. Config flag:  `tie_word_embeddings: False`
   vocab_size 100352, hidden_size 2048, num_hidden_layers 16, intermediate_size 8192
   -- every M0 architectural assumption confirmed exactly.

2. Checkpoint structure: two SEPARATE tensors of identical shape, in different shards.
   model.embed_tokens.weight  [100352, 2048]  -> shard 1
   lm_head.weight             [100352, 2048]  -> shard 2
   If tied, `lm_head.weight` would not exist as its own tensor at all.

3. Parameter count summed from safetensors headers: **1.4849 B**
   M0 derived from architecture alone:                1.4850 B
   Agreement to 4 decimal places. The earlier 1.279 B error was exactly one untied
   lm_head (100352 x 2048 = 205.5 M).

Bonus confirmation: the download is 5.95 GB and `torch_dtype` is float32.
1.485 B x 4 bytes = 5.94 GB. The file size alone implies the parameter count, and it
matches -- a fourth check arriving free, from a completely different direction.

Cost of untying, in the terms that matter for training:
- +205.5 M trainable parameters (14% of the model)
- at 16 bytes/param for mixed-precision AdamW, that is **~3.3 GB of the 27.11 GB**
  training budget attributable to this one architectural choice.

Conclusion:
- **M0 open question CLOSED.** `tied_embeddings=False` in `src/utils/memory.py` is correct
  and now verified against the real model, not assumed.
- The M0 memory estimates stand; no correction needed to `PRESETS["olmo2-1b"]`.
- Forward-looking (M2): because the embeddings are UNTIED, adding chat-template special
  tokens will require resizing BOTH matrices, not one. Tied models would resize once.

---

## m1-base-exploration
Date        : 2026-09-08
Milestone   : M1
Question    : What is a BASE model, concretely, and what did post-training change?
Config      : n/a
Script      : `scripts/explore_base_model.py`
Models      : allenai/OLMo-2-0425-1B @ a1847dff35000b4271fa70afc5db10fd29fedbdf
              allenai/OLMo-2-0425-1B-Instruct
Hardware    : 1 x H200, CUDA_VISIBLE_DEVICES=5 (all 8 GPUs were in use by colleagues;
              picked the one at 0% utilisation with ~98 GB free)
Command     : `CUDA_VISIBLE_DEVICES=5 python scripts/explore_base_model.py \
                 --model allenai/OLMo-2-0425-1B \
                 --compare allenai/OLMo-2-0425-1B-Instruct`

### Result 1 (measured) — the headline: Base vs Instruct on "What is 2 + 2?"

| | Base | Instruct |
|---|---|---|
| generated tokens | **120 (hit the cap)** | **9** |
| stopped on its own (EOS)? | **No** | **Yes** |
| output | "It's a simple question, but it's also a question that can be answered i... First, let's think about what we mean when we say \"2 + 2.\"" | "2 + 2 equals 4." |

Identical parameter count (1,484,916,736), identical architecture, identical tokenizer.
**Only the weight VALUES differ.** Post-training added no capacity; it redistributed it.

### Result 2 (measured) — the Base model is not "bad at answering", it is doing a different job

Top-10 next-token predictions after "What is 2 + 2?" (Base):

    p=0.0992 ' It'      p=0.0932 ' The'     p=0.0515 ' What'
    p=0.0440 ' '        p=0.0388 ' Well'    p=0.0312 ' This'

Not one of them is "4". Every candidate is a **discourse continuation** — the model is
continuing a document that happens to contain a question, which is exactly what
next-token pretraining optimised. Entropy 4.821 nats (uniform over the vocab would be
11.516), so it is far from random; it is confidently doing the wrong task.

**This is the cleanest statement of why post-training exists**: the Base model does not
lack the knowledge, it lacks the *convention* that a question should be followed by an
answer and then a stop.

### Result 3 (measured) — the chat template is the interface post-training installed

    Base     chat_template = None
    Instruct chat_template = PRESENT (508 chars)

Instruct wraps the prompt as:

    '<|endoftext|><|user|>\nWhat is 2 + 2?\n<|assistant|>\n'

This exact string is the input to M2. A Base model has no notion of roles at all.

### Result 4 (measured) — tokenizer facts that will bite us in M2

1. **`tokenizer.vocab_size = 100278`, but `embed_tokens` is `[100352, 2048]`.**
   The embedding matrix is 74 rows LARGER than the vocabulary. 100352 = 784 x 128, i.e.
   the vocab is padded to a multiple of 128 for tensor-core alignment. Consequences:
   - `src/utils/memory.py` correctly uses 100352 (the matrix), not 100278 (the tokenizer).
   - There are **74 unused embedding slots**, so a few chat special tokens can be added
     in M2 without `resize_token_embeddings()` growing the matrix at all.

2. **`bos_token == eos_token == '<|endoftext|>' (id 100257)`.** The same ID means "start"
   and "stop". You cannot distinguish them by token ID -- only by position. Label masking
   in M2 must not assume they are different.

3. **The tokenizer does NOT prepend BOS** (`starts with BOS? False`), yet the chat
   template *does* begin with `<|endoftext|>`. So BOS arrives from the template, not the
   tokenizer. Applying both would double it.

4. **`pad_token = '<|pad|>' (id 100277)`** exists and is distinct from EOS. Good -- many
   base models lack a pad token entirely and force you to invent one.

5. **Digits tokenize separately**: "2" after a space becomes TWO tokens, `220` (' ') then
   `17` ('2'), not a single ' 2'. Number handling is character-level here, which is worth
   remembering when we measure math ability in M13.

### Result 5 (measured) — logits are exactly the shape M0 predicted

    input_ids : (1, 8)            [batch, seq]
    logits    : (1, 8, 100352)    [batch, seq, VOCAB]

1.6 MB in bf16 for an 8-token prompt. Scaled to batch 1 / seq 2048 in fp32 this is the
1.53 GB "Logits + their grad" row from the M0 memory table -- confirmed by observation
rather than arithmetic. One score per vocabulary entry at EVERY position; position t
predicts token t+1.

### Also confirmed
- `embed_tokens` and `lm_head` have different `data_ptr()` -> untied, in the model object
  and not merely in the config. Cost: 205.5 M params = **3.29 GB** of training memory.
- Base checkpoint is fp32 (5.9 GB); the post-trained checkpoints are bf16 (~3 GB each).

### Result 6 (measured) — causality is now proven, not asserted

Took "The capital of France is Paris", changed ONLY the last token to " Berlin", and
re-ran the forward pass:

    logits at positions 0..n-2 identical?  True
    logits at the last position differ?    True

Bit-identical prefixes. That is the causal mask observed directly. It is also *why*
teacher forcing works: because no position can see the future, one forward pass scores
every next-token prediction in the sequence at once, instead of needing N passes.

### Result 7 (measured) — the pretraining objective, computed two ways

    text            : "The capital of France is Paris."
    model loss      : 2.8438  (mean cross-entropy, nats/token)
    perplexity      : 17.18
    manual recompute: 2.8438  <- identical

Manual version: shift logits `[:, :-1, :]` (drop the last position, nothing follows it)
against labels `[:, 1:]` (drop the first token, nothing predicts it), then
`F.cross_entropy`. Shapes `(1, 6, 100352)` vs `(1, 6)`.

    L = -(1/N) * sum_t log P(y_t | y_<t)

**Per-token loss is the most instructive output of M1:**

    ' capital'   8.419   ################################
    ' of'        0.947   ###
    ' France'    5.036   ####################
    ' is'        1.276   #####
    ' Paris'     0.287   #
    '.'          1.098   ####

Uncertainty collapses as context accumulates. After "The" almost anything could follow
(8.4 nats). By the time the model has "The capital of France is", predicting " Paris"
costs 0.287 nats -- it is nearly certain. The loss number we will watch for the rest of
the project is just the mean of a column like this.

**This is the bridge to M2/M3.** SFT uses this identical loss. The only difference is
that prompt-position labels are replaced by -100 so they contribute nothing to the mean.

### Result 8 (measured) — embeddings are a lookup table, and the space is meaningful

    embedding matrix : (100352, 2048)   [vocab, hidden]
    ' Paris' -> id 12366 -> vector (2048,), L2 norm 10.174

The embedding "lookup" is literally row-indexing this matrix -- no computation at all.

Nearest neighbours by cosine similarity:

    0.9017 'Paris'    0.6579 ' paris'   0.4343 ' France'   0.4090 ' London'
    0.4006 'France'   0.3641 ' Berlin'  0.3630 'London'    0.3516 ' French'

Casing variants first, then the country, then other European capitals. The geometry
carries meaning, learned purely from next-token prediction.

### Script hardening (2026-09-08)

`scripts/explore_base_model.py` originally defaulted `--device` to `"cuda"`, which means
`cuda:0` = the first VISIBLE device. On this shared box, running without pinning would
have silently taken physical GPU 0 from a colleague. The script now **refuses to start**
on CUDA unless `CUDA_VISIBLE_DEVICES` is set (override: `--allow-any-gpu`), and it prints
the visible-to-physical mapping so `cuda:0` is never mistaken for physical GPU 0.

Sections 5-7 were run on **CPU** (192 cores, fp32) because all 8 GPUs were at 100%
utilisation with colleagues' jobs. Inspection work of this size does not need a GPU.

Conclusion:
- M1 complete. "Base vs Instruct" is no longer an abstraction: it is 120 rambling tokens
  versus 9 tokens and a stop.
- The official `-SFT`, `-DPO` and `-Instruct` checkpoints are all downloaded locally, so
  every checkpoint we train in M4/M9 can be compared against the official equivalent.
- Open for M2: build the chat template + label masking pipeline, being careful about
  (a) the 74 spare embedding rows, (b) bos == eos, (c) not double-adding BOS.

### M1 materials (2026-09-11) — learning method changed

The blank-page-reimplementation approach was replaced at the user's request with
annotated runnable materials. M1 was regenerated as:

- `notebooks/m1_base_model_exploration.py` — 42 cells, `# %%` delimited, runs as a script
- `notebooks/m1_base_model_exploration.ipynb` — generated from it
- `scripts/to_ipynb.py` — stdlib-only converter (no jupyter dependency to BUILD a notebook)

Contents: tokenizer and the four M2 traps; embeddings as a lookup table with cosine
neighbours; forward pass and the `[batch, seq, vocab]` logits; softmax / entropy /
temperature; a causality proof; the CLM loss computed by hand and matched against the
model, with per-token breakdown; a live demonstration that `-100` positions contribute
nothing; greedy generation and why Base never emits EOS; Base vs Instruct; and the
boundary questions that open M2.

Environment additions to the `ptl` env: `ipykernel`, `jupyterlab`, `jupytext`. Kernel
registered as `ptl` ("Python 3 (ptl)").

---

## m2-data-pipeline
Date        : TODO
Result      : TODO
