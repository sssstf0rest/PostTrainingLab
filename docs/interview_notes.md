# Interview Notes

Questions and answers grounded in work I actually did in this repo. Each answer
should be speakable in 60–90 seconds. Follow-ups are the questions a strong
interviewer asks next.

---

## M0 — Environment, memory and precision

### Q: A 1B-parameter model takes ~3 GB to serve. Why does training it take ~27 GB?

**Answer.** Inference needs only the weights and a small activation working set. In
bf16 that is roughly 2 bytes × 1.5B ≈ 3 GB. Training adds four things:

1. **Gradients** — one value per trainable parameter, same size as the weights.
2. **Optimizer states** — AdamW keeps a momentum and a variance term per parameter,
   both in fp32: 8 bytes per parameter, four times the bf16 weights.
3. **FP32 master weights** — mixed precision keeps a full-precision copy so that
   small updates aren't rounded away, another 4 bytes per parameter.
4. **Activations** — every intermediate tensor the backward pass needs, scaling with
   `layers × batch × seq_len × hidden`.

That is about 16 bytes per parameter before any activations. For OLMo 2 1B the
optimizer states alone are the largest single line item — roughly 40% of peak memory,
while the weights are about 10%.

**Follow-up: which component would you attack first when you OOM?**
It depends which term dominates, which is why I built a calculator rather than
guessing. If activations dominate (large batch or long sequences), gradient
checkpointing or a smaller micro-batch with gradient accumulation. If optimizer state
dominates, ZeRO-1/2 sharding across GPUs, or a lower-state optimizer like Adafactor,
or LoRA if the task tolerates it. Notably, LoRA does *not* reduce the parameter term —
the frozen base still occupies VRAM; it removes gradients, optimizer state and master
weights for the frozen 99.5%.

**Follow-up: anything surprising in the breakdown?**
Yes — the logits tensor. It is `[batch, seq, vocab]` and HF upcasts it to fp32 for the
cross-entropy. With a 100k vocabulary at batch 4 and sequence 2048 that is 3.3 GB, plus
an equally large gradient tensor. The final projection can cost more memory than the
entire model, which is why large-vocabulary models use fused or chunked cross-entropy.

---

### Q: BF16 and FP16 are both 16 bits. Why does modern LLM training use BF16?

**Answer.** They allocate their bits differently. FP16 has a 5-bit exponent and
10-bit mantissa; BF16 has an 8-bit exponent — identical dynamic range to FP32 — and a
7-bit mantissa. Gradients are very small numbers, so range matters more than
precision. In FP16 small gradients underflow to exactly zero, which is why FP16
training needs a dynamic **loss scaler**: multiply the loss by a large factor before
backward, then unscale before the optimizer step, and back off whenever an inf or NaN
appears. BF16 has the same exponent range as FP32, so gradients simply lose a little
precision instead of vanishing, and no loss scaling is needed. The cost is that BF16
needs hardware support — Ampere (compute capability 8.0) or newer.

**Follow-up: so why keep FP32 anywhere?**
Because "mixed precision" is not "everything in bf16". The optimizer keeps an fp32
master copy of the weights. With a learning rate around 1e-5, `weight + lr * grad`
evaluates to `weight` in bf16 — the update rounds away entirely. Accumulating updates
in fp32 is what makes the tiny steps actually land. Pure-bf16 training doesn't crash;
it silently stops learning, which is a much worse failure mode.

---

### Q: You have a new GPU box. What do you check before launching a run?

**Answer.** I run an environment probe that records: GPU model and VRAM, driver
version, the CUDA version the driver supports, the CUDA version torch was *built*
against, compute capability, and whether bf16 is available. Two of those are the usual
failure: `torch.version.cuda` must be ≤ the driver's CUDA version, otherwise kernels
fail to launch; and compute capability must be ≥ 8.0 for bf16, otherwise I need fp16
plus loss scaling. I save that probe as JSON alongside the experiment, so months later
I can tell whether a difference between two runs came from the code or the machine.

**Follow-up: why record it per experiment rather than once?**
Because rented boxes differ between sessions, and "the same script gave a different
number" is otherwise unresolvable. A result without its environment isn't reproducible.

---

### Q: What is the difference between SFT, DPO and GRPO at the level of code?

**Answer.** Much less than the names suggest. All three run the same forward pass,
the same autograd backward, and the same optimizer step. They differ in exactly two
places: what data goes in, and how the loss is computed from the logits. SFT takes
conversations and applies cross-entropy over assistant tokens only. DPO takes
preference pairs and computes a loss on the log-probability *gap* between the chosen
and rejected responses, offset by a frozen reference model. GRPO samples a group of
completions from the current policy, scores them with a reward function, normalizes
those rewards within the group into advantages, and weights each token's
log-probability by its advantage. Once you can write one training step by hand, the
rest is loss functions.

*(This answer gets stronger with every milestone — revisit after M9 and M13 and replace
the description with what I actually implemented.)*
