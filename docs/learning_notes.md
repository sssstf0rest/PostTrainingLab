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

### BF16 vs FP16 finally clicked

Both are 16 bits, but they spend them differently. FP16 gives 5 bits to the exponent;
BF16 gives 8 — the same range as FP32 — and pays for it with mantissa bits. Gradients
are tiny numbers, so *range* matters more than *precision*: in FP16 they underflow to
zero (hence the loss-scaling machinery), while in BF16 they simply lose a little
precision. Trading accuracy for range is the right trade when the failure mode is
"silently becomes 0".

And "mixed precision" is not "everything in bf16". The fp32 master copy exists
because `weight + 1e-5 * grad == weight` in bf16 — the update rounds away and
training silently stalls without ever erroring.

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

### Open loops to close in M1

- Verify `tied_embeddings=False` against the real `config.json`.
- See actual logits and confirm the `[batch, seq, vocab]` shape with my own eyes.
- Watch a base model *fail* to stop generating, so "post-training" stops being abstract.
