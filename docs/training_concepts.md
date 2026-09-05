# Training Concepts

Concise technical reference. Grows one milestone at a time. Every entry should be
something I can explain out loud without notes.

---

## 0. The training loop (the whole project in one diagram)

```
Training Data → Tokenization → Batch → Forward → Logits → Loss
                                                            ↓
Updated Weights ← Optimizer ← Gradients ← Backpropagation ←─┘
        ↓
     Repeat
```

**The central insight of this project:** SFT, reward modelling, DPO, PPO and GRPO
differ ONLY in *what text is shown* (step 1) and *how wrongness is measured* (step 5).
Forward, backward, gradients and optimizer are byte-for-byte the same machinery in
all of them.

| Method | Data | Loss |
|---|---|---|
| Pretraining | Raw web text | Predict every next token |
| SFT | Conversations | Predict next token, assistant spans only |
| Reward model | Preference pairs | Rank chosen above rejected |
| DPO | Preference pairs | Raise chosen logprob, lower rejected, anchored to a frozen reference |
| PPO / GRPO | Model's own samples | Raise logprob of tokens in high-reward samples |

### Causal language modelling objective

$$\mathcal{L} = -\frac{1}{N}\sum_{t \in \mathcal{S}} \log P_\theta(y_t \mid y_{<t})$$

- $y_t$ — correct token at position $t$
- $y_{<t}$ — all preceding tokens. *Causal* = the model cannot attend forward.
- $P_\theta$ — softmax over the model's logits at position $t-1$
- $\mathcal{S}$ — the set of **scored** positions. In pretraining, all of them. In SFT,
  assistant tokens only. This one set is the entire difference between pretraining
  and SFT.
- $N = |\mathcal{S}|$

**Perplexity** $= e^{\mathcal{L}}$ — "how many tokens is the model effectively choosing
between". Loss 2.3 ≈ perplexity 10. Loss 0.7 ≈ perplexity 2.

A forward pass over a length-$L$ sequence produces $L$ predictions at once, which is
why training is throughput-bound rather than sequential.

---

## 1. CPU vs GPU, and what CUDA actually is

A **CPU** has ~8–64 fast, general cores optimized for branchy sequential work.
A **GPU** has ~10,000 slow, simple cores optimized for the same arithmetic applied
to enormous arrays. Transformer training is ~95% dense matrix multiplication, which
is exactly that shape, so a GPU is 10–100× faster here despite slower clocks.

**CUDA** is NVIDIA's programming model + driver + toolkit for running code on their
GPUs. Three CUDA version numbers exist and confusing them is the #1 setup failure:

| Thing | Reported by | Meaning |
|---|---|---|
| Driver CUDA version | `nvidia-smi` (top right) | Highest CUDA the driver supports — an *upper bound* |
| Toolkit torch was built with | `torch.version.cuda` | What the wheel expects. Must be ≤ driver version |
| Compute capability | `torch.cuda.get_device_properties(0)` | The GPU's *architecture* generation, e.g. 8.0 = Ampere |

Compute capability ≥ 8.0 (Ampere: A100, A6000, 3090, 4090, L40S, H100) is what gives
you native **bf16** and fast TF32 matmuls. On older cards (V100 = 7.0, T4 = 7.5) you
must fall back to fp16 with loss scaling.

**Apple MPS** is a different backend entirely. It can run forward/backward for small
models, but has no FlashAttention, no bitsandbytes, no DeepSpeed, and different
autocast semantics. Fine for inspection; useless for real training.

---

## 2. VRAM: where training memory actually goes

VRAM is the GPU's own memory. Everything the GPU touches must live there. The
question "will this fit?" decomposes into seven terms.

For a model with $\Psi$ parameters trained in **bf16 mixed precision with AdamW**:

| Component | Size | Why it exists |
|---|---|---|
| Parameters | $2\Psi$ | the weights, in bf16 |
| Gradients | $2\Psi$ | one per **trainable** parameter |
| AdamW state $m$ | $4\Psi$ | fp32 momentum (1st moment) |
| AdamW state $v$ | $4\Psi$ | fp32 variance (2nd moment) |
| FP32 master weights | $4\Psi$ | so tiny `lr × grad` updates don't round to zero in bf16 |
| Activations | see below | forward tensors the backward pass needs |
| Logits + their grad | $2 \cdot B \cdot L \cdot V \cdot 4$ | the sleeper term |

That is **≈16 bytes per parameter before a single activation**. A 1B model needs
~16 GB just to hold its own training state.

**Measured against our estimator** (`scripts/memory_math.py --preset olmo2-1b`,
batch 1, seq 2048, full fine-tune): the model weights are only **10%** of peak VRAM.
Optimizer + master weights + gradients are **71%**. This is the single most
counter-intuitive fact about training memory, and it explains:

- why inference of a 1B model needs ~3 GB but training needs ~27 GB;
- why **DeepSpeed ZeRO** shards *optimizer states first* (ZeRO-1) — it is the
  biggest single term, and sharding it is free of extra communication;
- why **LoRA** is so cheap: freezing the base collapses gradients + optimizer +
  master weights to near zero, leaving only parameters and activations.

### Activations

Every intermediate tensor saved during forward, because backward needs it to apply
the chain rule. Scales with `layers × batch × seq_len × hidden`, NOT with parameter
count. This is why doubling batch size can OOM a model that fit a moment ago.

**Gradient checkpointing** saves only each layer's *input* and recomputes the
interior during backward: ~linear activation memory instead of ~18× that, at the
cost of roughly +30% compute. Classic memory/time trade. (To be measured in M6.)

Without **FlashAttention**, an explicit $[B, \text{heads}, L, L]$ attention matrix is
materialized — quadratic in sequence length, and usually the reason long-context
runs OOM.

### The logits term (the one people forget)

Logits are $[B, L, V]$ and HF upcasts them to fp32 for the cross-entropy. With
OLMo 2's $V = 100{,}352$, batch 4, seq 2048: $4 \times 2048 \times 100352 \times 4$ B
= **3.3 GB**, and the backward pass allocates a same-sized gradient. That is more
memory than the entire 1B model's weights, spent on the last layer alone. It is why
large-vocabulary models use chunked/fused cross-entropy kernels.

---

## 3. Numeric precision

| dtype | bits | exponent | mantissa | notes |
|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | The safe baseline. 2× memory, slower matmuls. |
| TF32 | 32 storage | 8 | 10 | Ampere+ matmul mode. FP32 range, reduced precision. Nearly free speedup. |
| FP16 | 16 | 5 | 10 | Narrow range → gradients underflow to 0 → needs a **loss scaler**. |
| BF16 | 16 | 8 | 7 | **Same exponent range as FP32**, fewer mantissa bits. No loss scaling needed. Ampere+. |
| FP8 | 8 | 4/5 | 3/2 | Hopper+. Frontier-scale training. Not in scope here. |

**BF16 is the default for modern LLM training** because range matters more than
precision for gradients. Trading mantissa bits for exponent bits means values never
silently become zero or infinity — the failure mode that made FP16 training so
fragile.

"**Mixed precision**" ≠ "everything in bf16". It means: matmuls in bf16 for speed,
but a master copy of the weights and the optimizer moments in fp32, so that an
update of size `1e-5 × grad` is not rounded away. Pure-bf16 training silently stalls
because `weight + tiny_update == weight` in bf16.

---

## 4. Why deep-learning environments are so fragile

A working training environment is a chain where every link must match:

```
GPU architecture → driver → CUDA toolkit → torch wheel → flash-attn / bitsandbytes
                                                 ↓
                                    transformers ↔ trl ↔ peft ↔ accelerate
```

- Compiled extensions (`flash-attn`, `bitsandbytes`, `deepspeed`) are built against a
  specific torch + CUDA ABI. Upgrading torch silently breaks them.
- `transformers` and `trl` change APIs frequently; TRL in particular has renamed
  trainer arguments across minor versions.
- New model architectures require a *minimum* `transformers` version, so "just pin
  everything old" is not a solution either.

**Discipline for this project:**
1. One conda env per machine, never `base`.
2. Install torch FIRST, matched to the box's CUDA. Then everything else.
3. Add dependencies only at the milestone that needs them (see `requirements/gpu.txt`).
4. Freeze exact versions (`pip freeze`) into each experiment directory from M4 on,
   so a result can always be traced to the stack that produced it.

---

## 5. Dependency stack — why each package exists

**Tier 0 (M0):**
- `torch` — tensors, autograd, optimizers, CUDA/MPS bindings. The engine.
- `pyyaml` — config files, so experiments are files not shell history.

**Tier 1 (M1–M2):**
- `transformers` — architectures, `from_pretrained`, tokenizers, chat templates.
- `huggingface_hub` — weight/dataset download and cache management.
- `datasets` — memory-mapped loading and `.map()` preprocessing that scales past RAM.
- `tokenizers`, `sentencepiece` — fast tokenizer backends.
- `rich` — legible terminal tables. We will read a *lot* of tensors.

**Deferred on purpose** (`requirements/gpu.txt`): `accelerate`, `trl`, `peft`,
`wandb`, `deepspeed`, `bitsandbytes`, `flash-attn`, `lm-eval`, `vllm`. Each is
installed at the milestone that first needs it, so that when something breaks we
know which addition broke it.

---

## 6. Terms fixed now, used forever

- **Base model** — output of pretraining. A next-token predictor. Not an assistant.
- **Instruct model** — a base model after post-training. Follows instructions, stops.
- **Post-training** — everything after pretraining: SFT, preference optimization, RL.
- **Step** — one optimizer update. NOT one batch (see gradient accumulation, M4).
- **Effective batch size** — `micro_batch × grad_accum_steps × num_gpus`. The number
  that actually determines gradient quality and must be held constant across ablations.
- **Checkpoint** — model + optimizer + scheduler + RNG state. Weights alone are not
  a resumable checkpoint (M21).
