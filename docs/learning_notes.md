# Learning Notes

What I learned, in my own terms, milestone by milestone. Written after the work, not
before. If an entry here is something I could not explain to another engineer, it is
not finished.

---

## M0 — Repository and environment

### The one idea that reorganized my mental model

I came in understanding what SFT *data* looks like but not how it becomes parameter
updates. The bridge turned out to be smaller than expected:

> Every post-training method — SFT, reward modelling, DPO, PPO, GRPO — runs the
> *identical* forward/backward/optimizer machinery. They differ only in what text is
> fed in and how the loss is computed from the logits.

So "learning post-training" is mostly: learn ONE training step properly (M3), then
learn five different loss functions. That reframes the whole project from
"twenty separate techniques" to "one engine, five objectives."

### Training memory is not about the model

The result that surprised me most, before running anything:

```
OLMo 2 1B, full fine-tune, bf16 mixed precision, AdamW, batch 1, seq 2048
  Model parameters        2.77 GB   10.2%
  Gradients               2.77 GB   10.2%
  Optimizer states       11.06 GB   40.8%   <- largest single term
  FP32 master weights     5.53 GB   20.4%
  Activations             2.25 GB    8.3%
  Logits + their grad     1.53 GB    5.6%
  CUDA overhead           1.20 GB    4.4%
  TOTAL                  27.11 GB
```
*(estimates from `scripts/memory_math.py`; to be validated against real
`max_memory_allocated()` in M6.)*

The weights are 10% of the bill. AdamW's two fp32 moments plus the fp32 master copy
are 61%. Consequences I can now derive rather than memorize:

- **ZeRO-1 shards optimizer states first** because that is the biggest term.
- **LoRA's saving is not about the weights** — the frozen base still sits in VRAM.
  It removes gradients, optimizer states and master weights for 99.5% of parameters.
- **Activations scale with batch × seq, not with parameter count**, which is why a
  run that fits at batch 1 OOMs at batch 8 even though the model didn't change.
- **The logits tensor is `[batch, seq, vocab]` in fp32.** At batch 4 / seq 2048 with a
  100k vocab that is 3.3 GB — larger than the model. I would never have guessed the
  final linear layer was a memory problem.

### The number to memorize: 16 bytes per parameter

Everything above collapses into one rule of thumb for mixed-precision AdamW:

| what | bytes/param |
|---|---|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam `m` (momentum) | 4 |
| Adam `v` (variance) | 4 |
| **total** | **16** |

1.485 B × 16 bytes = 22.1 GB — 82% of the measured 27.11 GB total, before a single
activation. Inference is 2 bytes/param, so **training costs 8× inference on the
parameter side.** "Will this fit?" is now one multiplication instead of a guess.

Corollary I can now reason about: switching AdamW → SGD drops 16 bytes/param to 8
(no `m`, no `v`) and the total from 27.11 → 14.06 GB. Optimizer *choice* is a memory
decision, not just a convergence decision.

### KV cache is an inference concept, not a training one

I assumed the KV cache was part of the training bill. It is not, and the reason matters.
Training uses **teacher forcing**: the entire target sequence is already known, so the
whole thing goes through in one parallel forward pass. There is no step-by-step decoding,
so there is nothing to cache between steps.

The KV cache exists only to avoid recomputing keys and values across *incremental decode
steps* — a generation-time optimization. It becomes relevant again in M13/M18, where RL
rollouts actually generate text and therefore do pay for it.

### BF16 vs FP16 finally clicked

Both are 16 bits, but they spend them differently. FP16 gives 5 bits to the exponent;
BF16 gives 8 — the same range as FP32 — and pays for it with mantissa bits. Gradients
are tiny numbers, so *range* matters more than *precision*: in FP16 they underflow to
zero (hence the loss-scaling machinery), while in BF16 they simply lose a little
precision. Trading accuracy for range is the right trade when the failure mode is
"silently becomes 0".

| | exponent bits | mantissa bits | range |
|---|---|---|---|
| FP32 | 8 | 23 | ~1e-38 … 3e38 |
| BF16 | 8 | 7 | **same as FP32** |
| FP16 | 5 | 10 | 6e-5 … 65504 |

**The extra machinery FP16 needs is a loss scaler** (`torch.amp.GradScaler`), and it is
a real moving part, not a formality. FP16's smallest normal value is ~6e-5 while actual
gradients run around ~1e-7, so they would flush straight to zero. So you multiply the
loss by a large constant (say 2^16) *before* `.backward()`, which lifts every gradient
into representable range, then unscale before `optimizer.step()`. Dynamic scaling
watches for inf/NaN, skips those steps, and re-tunes the constant.

BF16 keeps FP32's full exponent range, so gradients never underflow and **no scaler is
needed at all** — one fewer component that can diverge. That, not precision, is why
BF16 won. Its cost is 7 mantissa bits instead of 10, which leads directly to the next
point.

And "mixed precision" is not "everything in bf16". The fp32 master copy exists
because `weight + 1e-5 * grad == weight` in bf16 — the update rounds away and
training silently stalls without ever erroring.

Concretely: bf16's 7 mantissa bits give a relative resolution of 2^-8 ≈ **0.4%**, and
the next representable value above 1.0 is 1.0078. So `bfloat16(1.0) + 0.0001 == 1.0`.
With lr=1e-5 a typical update is ~1e-7 against a weight of ~0.02 — five orders of
magnitude below the rounding threshold. Every update would vanish. **No crash, no NaN,
just a loss curve that refuses to move**, which is a far nastier bug than an error.
FP32's 23 mantissa bits (~1e-7 relative) hold the update, which is why the optimizer
step happens on the master copy and only the forward/backward run in bf16.

### Gradient checkpointing: it worked perfectly and barely helped

Measured on `olmo2-1b`, batch 1, seq 2048:

```
activations   2.25 GB  ->  0.27 GB    (88% of the component gone)
TOTAL        27.11 GB  -> 25.12 GB    (only 7% of the bill gone)
```

**Cost:** roughly one extra forward pass — typically 20–40% slower training.
**Mechanism:** normally every intermediate tensor from the forward pass is kept because
backward needs it. Checkpointing stores only layer boundaries and *recomputes* the
interior during backward. It trades FLOPs for bytes.

**It shrinks exactly one of the seven rows: Activations.** It cannot touch weights,
gradients, optimizer states or master weights.

This is the most useful lesson in M0. The technique did its job almost perfectly — it
deleted 88% of what it targets — and moved the total by 7%, because activations are
only 8.3% of memory at this batch and sequence length. **Whether an optimization helps
depends entirely on which component currently dominates**, and that changes with batch
size, sequence length and model size. Activations grow with batch × seq; optimizer
state does not. At batch 16 / seq 4096 the same flag becomes the difference between
OOM and training.

So when I want to attack the 61% held by optimizer + master weights, checkpointing is
the wrong tool — ZeRO sharding (M20), 8-bit optimizers, or LoRA are the right ones.
This is Rule 9 in one experiment: the folklore ("checkpointing saves memory") is true
and still useless without knowing *which* memory.

### The CUDA version rule I had backwards

I assumed `torch.version.cuda` had to be the *higher* number. It is the opposite, and
both halves of my mental model were wrong:

- **`nvidia-smi`'s "CUDA Version"** is the maximum CUDA the *driver* can support. It is
  a property of the **driver**, not of the GPU.
- **`torch.version.cuda`** is the CUDA toolkit the torch **wheel was compiled against**.
- **The driver's number must be ≥ torch's.**

The model: the driver is the floor the whole stack stands on. NVIDIA guarantees a newer
driver runs older toolkit code (backward compatible), never the reverse. Violate it and
you get `CUDA error: no kernel image is available for execution on the device` — an
error that sounds like a code bug and is actually an install bug.

`src/utils/env.py` already encodes this check and emits a note when the wheel is newer
than the driver allows. Our box: driver 570.172.08 supports CUDA 12.8, and we install
the cu128 build. Matched.

### Two things I got wrong, and what they taught me

1. **My first OLMo 2 1B parameter estimate was 1.279B, but the published figure is
   1.48B.** The 206M gap is exactly `vocab × hidden` = one untied `lm_head`. So OLMo 2
   1B does *not* tie its input and output embeddings. The lesson is bigger than the
   fix: computing a parameter count from architecture is a cheap check on whether I
   actually understand a model's shape, and disagreeing with the model card means one
   of us is wrong — worth resolving, not rounding away.

2. **`learning_rate: 1e-5` in YAML loads as the string `"1e-5"`.** PyYAML implements
   YAML 1.1, whose float pattern requires both a decimal point and a signed exponent.
   No error is raised at load time. This is exactly the class of bug that silently
   invalidates an ablation, and it is why the config loader has a test.

### Why the environment is the first milestone at all

I would have skipped straight to training. But the failure modes here — a torch wheel
built for a newer CUDA than the driver supports, a `flash-attn` compiled against a
different torch ABI, a compute capability below 8.0 with no bf16 — all present as
confusing runtime errors, sometimes hours into a paid GPU rental. Probing the machine
first and recording that probe with each experiment is what makes a result traceable.

### The training box is shared, and that changes how I work

The real environment turned out to be a **shared company server**, not a rented private
box: 8 × NVIDIA L20X (~140 GB each, compute capability 8.9 → native bf16), 192 CPU cores,
2 TB RAM, 13 colleagues' home directories — and **other people's jobs running under the
same UID as mine**. A 4.5-hour benchmark job was occupying GPU 2 while I was setting up.

Operational rules this forces, which are part of the craft and not a distraction from it:

- **Never broad-match process kills.** `pkill -f "<pattern>"` also matches the shell that
  invoked it *and* any same-UID job of a colleague. Kill by exact PID after checking
  `ps -o user,pid,etime,cmd -p <pid>`, or by a tmux session name I created.
- **Check `nvidia-smi`, then pin `CUDA_VISIBLE_DEVICES`** to specific idle GPUs. A job
  that grabs all 8 is a job that ruins someone's afternoon.
- **Install only into the project-local `.venv`.** No `sudo`, no system Python, no global
  pip. My environment must not be able to break anyone else's.

### Long jobs must survive the network, not just the GPU

SSH to this box goes through a corporate proxy that **kills long-lived connections**,
surfacing as `FATAL: recv() failed, 10054` and **exit code 255**.

The trap: exit 255 means the *connection* died, not the remote command. The job keeps
running server-side — so a naive retry starts a **second concurrent copy** writing into
the same venv. I did exactly this and had two pip processes racing on one virtualenv.

The fix, which now applies to every training run in M4 onward:

```bash
ssh <host> 'tmux new-session -d -s <name> "bash <script> > <log> 2>&1"'
```

SSH returns instantly, the job survives every drop, and progress is read by short
reconnects that `tail` the log. A 6-hour SFT run launched as a plain foreground SSH
command would die around minute 20. **Checkpointing and resumability (M21) are not
academic exercises on infrastructure like this** — they are what makes a long run
finishable at all.

### Open loops to close in M1

- Verify `tied_embeddings=False` against the real `config.json`.
- See actual logits and confirm the `[batch, seq, vocab]` shape with my own eyes.
- Watch a base model *fail* to stop generating, so "post-training" stops being abstract.


M1:
tokenizer is trained to learn the word spliting rules. and then it will go to vocabulary table to find the id. the vocab table is like a dictionary
```
    vocab = {
        "apple": 1,
        ...
    }

    # reverse mapping
    vocab = {
        1: "apple",
        ...
    }
```

embedding matrix shape will be: (num of vocab, num of features), i.e. (100351, 2048) in this model

L2 norm: sqrt(x1^2 + x2^2 + ... + xn^2): it represent the distance of the point in 2048 dimension to the origin

En = F.normalize(E.float(), dim=-1) # normalize every embedding to 1
sims = En @ En[tid] # @ is the dot product of vectors, computing the cosine similarity

logits:
1       = one input sequence
4       = four input tokens: ["Who", " are", " you", "?"]
100352  = one score for every token row the model can output