# %% [markdown]
# # M1 — Base-Model Exploration
#
# **Goal:** understand what a pretrained Base LLM *is*, by looking directly at the
# tensors rather than at a diagram.
#
# By the end you should be able to explain, from memory:
#
# | # | Concept | Where it appears below |
# |---|---|---|
# | 1 | What pretraining optimises | Part 6 |
# | 2 | Tokenizer, vocabulary, special tokens | Part 1 |
# | 3 | Embeddings as a lookup table | Part 2 |
# | 4 | Logits and their `[batch, seq, vocab]` shape | Part 3 |
# | 5 | Softmax, probabilities, entropy, temperature | Part 4 |
# | 6 | Causal masking / why teacher forcing works | Part 5 |
# | 7 | Cross-entropy loss, computed by hand | Part 6 |
# | 8 | Generation, EOS, why a Base model rambles | Part 7 |
# | 9 | What post-training actually changed | Part 8 |
#
# **How to use this:** run the cells top to bottom, then go back and change things.
# The `TRY THIS` notes suggest edits whose outcome is worth predicting before you run.
#
# Everything here runs fine on **CPU** — this is inspection, not training.

# %% [markdown]
# ## Part 0 — Setup
#
# Three environment variables matter on this machine, and they are not boilerplate:
#
# * `RAYON_NUM_THREADS` / `TOKENIZERS_PARALLELISM` — the Rust tokenizer builds a thread
#   pool sized to the machine (192 cores here). When colleagues' jobs already hold
#   ~16k threads under our shared UID, that allocation fails with
#   `PanicException: The global thread pool has not been initialized ... EAGAIN`.
#   Capping the pool avoids it.
# * `OMP_NUM_THREADS` — same idea for the CPU math kernels underneath torch.
#
# Set these **before** importing torch/transformers; several libraries read them once
# at import time.

# %%
import os

os.environ.setdefault("RAYON_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

print("torch", torch.__version__, "| cuda build", torch.version.cuda)

# %% [markdown]
# ### Choosing a device — this box is shared
#
# `"cuda"` means `cuda:0`, and **`cuda:0` is the first *visible* device, not physical
# GPU 0.** If `CUDA_VISIBLE_DEVICES` is unset, `cuda:0` *is* physical GPU 0 — which on
# this machine is probably a colleague's.
#
# So: check what is idle first, then pin. To run on physical GPU 3, launch with
# `CUDA_VISIBLE_DEVICES=3`; inside the process it appears as `cuda:0`.
#
# ```
# nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
# ```
#
# **CPU is the right default for this notebook.** A 1B model in fp32 is ~5.9 GB of RAM
# and every cell here is a handful of forward passes.

# %%
USE_GPU = False  # flip to True only after pinning CUDA_VISIBLE_DEVICES

if USE_GPU:
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Pin a GPU first — see the cell above."
    DEVICE, DTYPE = "cuda", torch.bfloat16
    print("visible devices:", os.environ["CUDA_VISIBLE_DEVICES"], "-> torch sees cuda:0")
else:
    DEVICE, DTYPE = "cpu", torch.float32

BASE = "allenai/OLMo-2-0425-1B"
INSTRUCT = "allenai/OLMo-2-0425-1B-Instruct"
print("device:", DEVICE, "| dtype:", DTYPE)

# %%
# Loading reads ~5.9 GB from the HF cache; roughly a minute on CPU the first time.
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE).to(DEVICE).eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,}  ({n_params / 1e9:.4f} B)")

# %% [markdown]
# ## Part 1 — The tokenizer
#
# A model never sees text. It sees **integers**. The tokenizer is the (lossless,
# reversible) map between them, and it is fixed at pretraining time — you cannot
# change it later without invalidating every learned embedding.

# %%
TEXT = "What is 2 + 2?"

enc = tok(TEXT)
ids = enc["input_ids"]

print("text          :", repr(TEXT))
print("input_ids     :", ids)
print("attention_mask:", enc["attention_mask"], " <- 1 = real token, 0 = padding")
print("n_tokens      :", len(ids))
print()
for i, tid in enumerate(ids):
    print(f"  {i:2d}  id={tid:<7d} {tok.decode([tid])!r}")

# %% [markdown]
# Note `' is'` carries its **leading space**. In byte-level BPE the space belongs to the
# token that follows it, which is why `"2"` and `" 2"` are different tokens.
#
# ### Four facts that will bite you in M2
#
# Read the output of the next cell carefully — each line is a trap that has cost people
# real debugging time.

# %%
print("1. vocab_size (tokenizer) :", tok.vocab_size)
print("   embedding rows (model) :", model.get_input_embeddings().weight.shape[0])
print("   -> the matrix is LARGER than the vocabulary.")
print("      100352 = 784 x 128: padded for tensor-core alignment.")
print("      Those spare rows can absorb new special tokens with no resize.")
print()
print("2. bos / eos / pad        :", repr(tok.bos_token), repr(tok.eos_token), repr(tok.pad_token))
print("   ids                    :", tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
print("   -> BOS and EOS are the SAME id. You cannot tell 'start' from 'stop'")
print("      by token id alone — only by position.")
print()
print("3. does the tokenizer prepend BOS?", ids[0] == tok.bos_token_id)
print("   -> No. But the chat template DOES start with it (Part 8).")
print("      Apply both and you double it.")
print()
digits = tok(" 2")["input_ids"]
print("4. ' 2' tokenizes to      :", digits, [tok.decode([d]) for d in digits])
print("   -> digits split character-wise. Relevant when we measure math in M13.")
print()
print("round-trip lossless?      :", tok.decode(ids) == TEXT)

# %% [markdown]
# **TRY THIS.** Before running: how many tokens is `"antidisestablishmentarianism"`?
# How many is `"      "` (six spaces)? Predict, then check with
# `len(tok(...)['input_ids'])`. Tokenizer cost is not proportional to character count,
# and that directly drives your training cost.

# %% [markdown]
# ## Part 2 — Embeddings
#
# The embedding "layer" is a `[vocab, hidden]` matrix, and the lookup is **literally
# row-indexing it**. No computation happens. That is the entire operation.

# %%
E = model.get_input_embeddings().weight
print("embedding matrix:", tuple(E.shape), " [vocab, hidden]")

word = " Paris"
tid = tok(word)["input_ids"][-1]
v = E[tid]
print(f"{word!r} -> id {tid} -> vector {tuple(v.shape)}")
print("first 8 values  :", [round(x, 4) for x in v[:8].float().tolist()])
print("L2 norm         :", round(v.float().norm().item(), 3))

# %% [markdown]
# Is that 2048-dim space meaningful? Cosine similarity against every other token says yes
# — and nothing taught it geography. This structure fell out of next-token prediction.

# %%
En = F.normalize(E.float(), dim=-1)
sims = En @ En[tid]
top = torch.topk(sims, 9)
print(f"nearest neighbours of {word!r}:")
for s, i in zip(top.values.tolist(), top.indices.tolist()):
    if i != tid:
        print(f"  {s:.4f}  {tok.decode([i])!r}")

# %% [markdown]
# **TRY THIS.** Swap `" Paris"` for `" king"`, `" Monday"`, or `" 7"`. Which kinds of
# words have tight neighbourhoods and which are diffuse? Casing variants usually rank
# first — worth asking yourself why that is unsurprising.

# %% [markdown]
# ## Part 3 — The forward pass
#
# One call. The key output is `logits`, shaped `[batch, seq, vocab]`:
# **one raw score for every vocabulary entry, at every position.**
#
# Position `t` holds the model's prediction for the token at position `t+1`.
# That single sentence is all of causal language modelling.

# %%
enc_t = tok(TEXT, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    out = model(**enc_t)

logits = out.logits
print("input_ids:", tuple(enc_t["input_ids"].shape), "  [batch, seq]")
print("logits   :", tuple(logits.shape), "  [batch, seq, VOCAB]")
print("dtype    :", logits.dtype)
print(f"size     : {logits.numel() * logits.element_size() / 1e6:.1f} MB for one short prompt")
print()
print("Scale that up: batch 1, seq 2048, fp32 ->",
      f"{1 * 2048 * logits.shape[-1] * 4 / 1e9:.2f} GB")
print("...and backward needs a gradient of the same size. That is the")
print("'Logits + their grad' row in the M0 memory table — bigger than the model.")

# %% [markdown]
# ## Part 4 — Logits to probabilities
#
# Logits are unbounded real numbers. `softmax` turns them into a distribution over the
# whole vocabulary. We want the prediction for the token *after* our prompt, which lives
# at the **last** position, `logits[0, -1]`.

# %%
last = logits[0, -1].float()
probs = torch.softmax(last, dim=-1)

top = torch.topk(probs, 10)
print(f"Top-10 continuations of {TEXT!r} (BASE model):\n")
for r, (p, i) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
    print(f"  {r:2d}  p={p:.4f}  {tok.decode([i])!r}")

ent = -(probs * probs.clamp_min(1e-12).log()).sum()
print(f"\nentropy: {ent:.3f} nats   (uniform over {len(probs)} would be "
      f"{torch.tensor(float(len(probs))).log():.3f})")

# %% [markdown]
# **This is the most important output in M1.**
#
# Not one candidate is `"4"`. They are all *discourse continuations* — `' It'`, `' The'`,
# `' Well'`. And entropy far below uniform means the model is **confident**.
#
# So the Base model is not bad at answering. It is confidently doing a different task:
# continuing a document that happens to contain a question. That is exactly what
# next-token pretraining asked it to do.
#
# **Post-training does not add knowledge. It installs the convention that a question is
# followed by an answer, and then a stop.**

# %%
# Temperature rescales logits before softmax: p = softmax(z / T).
# T < 1 sharpens (more deterministic), T > 1 flattens (more random).
for T in (0.5, 1.0, 2.0):
    p = torch.softmax(last / T, dim=-1)
    e = -(p * p.clamp_min(1e-12).log()).sum()
    tk = torch.topk(p, 3)
    shown = ", ".join(f"{tok.decode([i])!r}:{v:.3f}" for v, i in
                      zip(tk.values.tolist(), tk.indices.tolist()))
    print(f"T={T:<4} entropy={e:.3f}  top3: {shown}")

# %% [markdown]
# ## Part 5 — Causality, proven rather than asserted
#
# "Position `t` cannot see `t+1`" is a claim we can **test**. Take two sequences that
# share every position but the last. If attention is causal, the logits at all earlier
# positions must be *bit-identical*.

# %%
a = tok("The capital of France is Paris", return_tensors="pt").to(DEVICE)["input_ids"]
b = a.clone()
b[0, -1] = tok(" Berlin")["input_ids"][-1]

with torch.no_grad():
    la = model(input_ids=a).logits
    lb = model(input_ids=b).logits

print("changed only the last token: ' Paris' -> ' Berlin'")
print("logits at positions 0..n-2 identical?", torch.equal(la[0, :-1], lb[0, :-1]), " <- must be True")
print("logits at the last position differ? ", not torch.equal(la[0, -1], lb[0, -1]), " <- must be True")

# %% [markdown]
# That is the causal mask, observed.
#
# It is also **why teacher forcing works**: because no position can see the future, a
# single forward pass scores every next-token prediction in the sequence at once. Without
# causality you would need one pass per token. This is the difference between training
# being feasible and not.
#
# **TRY THIS.** Change a *middle* token instead of the last. Predict first: which
# positions' logits move? (Every position from that one onward — and none before it.)

# %% [markdown]
# ## Part 6 — The loss
#
# **This is the bridge to M2 and M3.** Everything above was observation; this is the
# quantity that training actually minimises.
#
# In causal-LM pretraining, `labels` *are* `input_ids`. The model shifts them internally.

# %%
LOSS_TEXT = "The capital of France is Paris."
enc_l = tok(LOSS_TEXT, return_tensors="pt").to(DEVICE)
ids_l = enc_l["input_ids"]

with torch.no_grad():
    out_l = model(input_ids=ids_l, labels=ids_l)

print("text       :", repr(LOSS_TEXT))
print("input_ids  :", tuple(ids_l.shape))
print("labels     : the same tensor")
print(f"loss       : {out_l.loss.item():.4f}  (mean cross-entropy, nats/token)")
print(f"perplexity : {out_l.loss.exp().item():.2f}  (= exp(loss); 'effective number of")
print("             equally-likely choices the model was deciding between')")

# %% [markdown]
# Now the same number **by hand**, so nothing is hidden.
#
# The shift is the only subtle part:
#
# * drop the **last** logit position — no token follows it, so it predicts nothing
# * drop the **first** label — nothing precedes it, so nothing predicts it
#
# leaving `seq-1` (prediction, target) pairs.

# %%
logits_l = out_l.logits.float()

shift_logits = logits_l[:, :-1, :]   # [batch, seq-1, vocab]
shift_labels = ids_l[:, 1:]          # [batch, seq-1]

manual = F.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
)

print("shift_logits:", tuple(shift_logits.shape), " [batch, seq-1, vocab]")
print("shift_labels:", tuple(shift_labels.shape), "      [batch, seq-1]")
print()
print(f"model  loss : {out_l.loss.item():.4f}")
print(f"manual loss : {manual.item():.4f}   <- identical")
print()
print("  L = -(1/N) * sum_t  log P(y_t | y_<t)")

# %% [markdown]
# The mean hides the story. Per-token loss shows what the model was actually surprised by.

# %%
per_tok = F.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
    reduction="none",
)

print("per-token loss (high = surprised):\n")
for tid_, l in zip(shift_labels[0].tolist(), per_tok.tolist()):
    print(f"  {tok.decode([tid_])!r:<12} {l:6.3f}  {'#' * min(int(l * 4), 40)}")

# %% [markdown]
# **Watch uncertainty collapse as context accumulates.** After `"The"` almost anything
# could follow, so the loss is high. By the time the model has `"The capital of France
# is"`, predicting `" Paris"` costs almost nothing — it is nearly certain.
#
# The single loss number you will stare at for the rest of this project is just the mean
# of a column like this.
#
# ### The one sentence that carries into M2 and M3
#
# > **SFT uses this exact loss.** The only change is that labels at prompt positions are
# > replaced with `-100`, so they contribute nothing to the mean.
#
# `-100` is `F.cross_entropy`'s default `ignore_index`. That is the whole mechanism of
# "assistant-only loss" — no special loss function, just a sentinel in the labels tensor.

# %%
# Proof that -100 is simply ignored: mask everything except the final two positions.
masked = shift_labels.clone()
masked[0, :-2] = -100

masked_loss = F.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    masked.reshape(-1),
)
kept = per_tok[-2:].mean()

print("labels with -100 applied:", masked[0].tolist())
print(f"masked loss                 : {masked_loss.item():.4f}")
print(f"mean of the 2 kept positions: {kept.item():.4f}   <- same number")
print("\nThe masked positions did not contribute at all. That is M2 in one cell.")

# %% [markdown]
# ## Part 7 — Generation
#
# Generation is just Part 4 in a loop: predict a distribution, pick a token, append it,
# repeat — until EOS or a length cap.
#
# `do_sample=False` is **greedy** decoding (always take the argmax), which makes runs
# reproducible.

# %%
gen = model.generate(
    **enc_t,
    max_new_tokens=120,
    do_sample=False,
    pad_token_id=tok.pad_token_id or tok.eos_token_id,
)
new = gen[0][enc_t["input_ids"].shape[1]:]

print("prompt:", repr(TEXT))
print(f"generated {len(new)} new tokens")
print("stopped on its own (EOS)?", new[-1].item() == tok.eos_token_id)
print("-" * 70)
print(tok.decode(new, skip_special_tokens=True))
print("-" * 70)

# %% [markdown]
# It hits the 120-token cap and **never emits EOS**. It would continue indefinitely.
#
# Why: in pretraining, `<|endoftext|>` appears between *documents*. The model learned it
# as "this document has ended", never as "I have finished answering you" — a notion that
# did not exist in the training data. Teaching it that is literally what M4's SFT does.

# %% [markdown]
# ## Part 8 — Base vs Instruct
#
# Same architecture, same parameter count, same tokenizer. **Only the weight values
# differ.** Load the officially post-trained checkpoint and ask the identical question.

# %%
del model
import gc; gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

tok_i = AutoTokenizer.from_pretrained(INSTRUCT)
model_i = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=DTYPE).to(DEVICE).eval()

print("Base     chat_template:", "None" if not getattr(tok, "chat_template", None) else "present")
print("Instruct chat_template:",
      f"present ({len(tok_i.chat_template)} chars)" if getattr(tok_i, "chat_template", None) else "None")

# %% [markdown]
# The **chat template** is the interface post-training installed: a convention for
# marking who is speaking. A Base model has no notion of roles at all.
#
# This exact string is the input to M2.

# %%
chat = tok_i.apply_chat_template(
    [{"role": "user", "content": TEXT}],
    tokenize=False,
    add_generation_prompt=True,
)
print("templated prompt:")
print(repr(chat))

enc_i = tok_i(chat, return_tensors="pt").to(DEVICE)
gen_i = model_i.generate(
    **enc_i,
    max_new_tokens=120,
    do_sample=False,
    pad_token_id=tok_i.pad_token_id or tok_i.eos_token_id,
)
new_i = gen_i[0][enc_i["input_ids"].shape[1]:]

print(f"\ngenerated {len(new_i)} new tokens")
print("stopped on its own (EOS)?", new_i[-1].item() == tok_i.eos_token_id)
print("-" * 70)
print(tok_i.decode(new_i, skip_special_tokens=True))
print("-" * 70)

# %% [markdown]
# | | Base | Instruct |
# |---|---|---|
# | tokens generated | ~120 (hit the cap) | ~9 |
# | stopped on its own | **No** | **Yes** |
# | `chat_template` | `None` | present |
#
# Identical capacity. Post-training redistributed it rather than adding to it.
#
# **TRY THIS.** Run the Base model on the *templated* prompt. It has never seen
# `<|user|>` in this role, so watch what it does with tokens it does not understand as
# control structure.

# %% [markdown]
# ## Part 9 — Where M2 begins
#
# M2 takes that templated string and answers one question: **which tokens should the
# model be trained to produce, and which must be masked to `-100`?**
#
# ```
# <|endoftext|><|user|>\nWhat is 2 + 2?\n<|assistant|>\n2 + 2 equals 4.<|endoftext|>
# └──────────────── mask with -100 ───────────────┘└──── learn these ────┘
# ```
#
# Think about the boundary cases before we build it — they are where everyone gets it
# wrong:
#
# 1. Should `<|assistant|>\n` itself be learned, or masked?
# 2. Should the final `<|endoftext|>` be learned? (What happens at inference if not?)
# 3. In a multi-turn conversation, are *earlier* assistant turns learned or masked?
#
# Write your answers down. We will check them against a real implementation in M2.
