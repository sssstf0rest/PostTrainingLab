# %% [markdown]
# # M2 — The Data Pipeline, From Scratch
#
# **Goal:** turn a conversation into the exact tensors a training step consumes, and
# understand every decision made along the way.
#
# M1 ended with a question. This notebook answers it:
#
# ```
# <|endoftext|><|user|>\nWhat is 2 + 2?\n<|assistant|>\n2 + 2 equals 4.<|endoftext|>
# └──────────────── mask with -100 ───────────────┘└──── learn these ────┘
# ```
#
# By the end you should be able to explain, from memory:
#
# | # | Concept | Where it appears below |
# |---|---|---|
# | 1 | What a chat template *is* (and that it is just Jinja) | Part 2 |
# | 2 | `input_ids` and `attention_mask` | Part 3 |
# | 3 | Why `labels` are not simply a copy of `input_ids` | Part 4 |
# | 4 | Assistant-only loss masking, and why `-100` | Parts 5–6 |
# | 5 | `offset_mapping` — the robust way to locate spans | Part 6 |
# | 6 | Reading a complete training example token by token | Part 7 |
# | 7 | Proof the mask changes the loss | Part 8 |
# | 8 | Multi-turn conversations | Part 9 |
# | 9 | Padding, `attention_mask`, and the collator | Part 10 |
# | 10 | Truncation and packing | Part 11 |
#
# **How to use this:** run top to bottom, then go back and break things. The `TRY THIS`
# notes suggest edits whose outcome is worth predicting *before* you run them.
#
# **No model and no GPU are needed for any of this.** M2 is entirely a tokenizer
# exercise — which is itself worth noticing: the data pipeline is independent of the
# weights, and bugs here are invisible until training silently learns the wrong thing.

# %% [markdown]
# ## Part 0 — Setup
#
# Same two environment concerns as M1. The Rust tokenizer builds a thread pool sized to
# the machine (192 cores here), and when colleagues' jobs already hold ~16k threads under
# our shared UID that allocation fails with
# `PanicException: The global thread pool has not been initialized ... EAGAIN`.
# Set these **before** importing — several libraries read them once at import time.
#
# We load only the **tokenizer**, so this costs a few megabytes rather than 5.9 GB.

# %%
import os

os.environ.setdefault("RAYON_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"  # the Base model has NO chat template
tok = AutoTokenizer.from_pretrained(MODEL_ID)

print("tokenizer      :", type(tok).__name__)
print("is_fast        :", tok.is_fast, "  <- offset_mapping in Part 6 needs this")
print("bos / eos / pad:", repr(tok.bos_token), repr(tok.eos_token), repr(tok.pad_token))
print("ids            :", tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)

# %% [markdown]
# Note again the M1 trap: **`bos_token == eos_token == '<|endoftext|>' (100257)`**.
# One ID means both "start" and "stop". Nothing downstream can tell them apart by ID —
# only by position. Any masking logic that keys off "is this token EOS?" will also
# match the BOS at position 0.

# %% [markdown]
# ## Part 1 — The raw material
#
# An SFT example is not a string. It is a **list of role/content dicts** — the same shape
# you already know from any chat API. This is the format essentially every instruction
# dataset ships in (Tulu 3, UltraChat, OpenHermes, ShareGPT variants).

# %%
CONVO = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "2 + 2 equals 4."},
]

print(json.dumps(CONVO, indent=2))

# %% [markdown]
# ## Part 2 — The chat template
#
# A chat template is **not** a model property in any deep sense. It is a **Jinja2 string**
# stored in `tokenizer_config.json`, and it is the entire convention post-training
# installed. Look at it directly rather than treating it as magic.

# %%
print(tok.chat_template)

# %% [markdown]
# Unpacked, that template says:
#
# * emit `bos_token` once at the very start
# * `system` → `<|system|>\n` + content + `\n`
# * `user` → `<|user|>\n` + content + `\n`
# * `assistant` → `<|assistant|>\n` + content + **`eos_token`** (+ `\n` if more turns follow)
# * if `add_generation_prompt`, append a bare `<|assistant|>\n` at the end
#
# **The single most important detail: every assistant turn ends with `eos_token`.**
# That is where "stop talking" is going to be learned.

# %%
text = tok.apply_chat_template(CONVO, tokenize=False)
print(repr(text))

# %% [markdown]
# Two modes, and confusing them is a classic bug:
#
# * **Training** (`add_generation_prompt=False`): the assistant's answer is present, and
#   ends with EOS. This is what we build labels from.
# * **Inference** (`add_generation_prompt=True`): the string *stops* at `<|assistant|>\n`
#   and the model continues from there.

# %%
prompt_only = tok.apply_chat_template(
    CONVO[:1], tokenize=False, add_generation_prompt=True
)
print("TRAINING :", repr(text))
print()
print("INFERENCE:", repr(prompt_only))

# %% [markdown]
# **TRY THIS.** Call `apply_chat_template(CONVO, add_generation_prompt=True)` on the
# *full* conversation (assistant turn included). What do you get, and why would training
# on that string teach the model something wrong?

# %% [markdown]
# ## Part 3 — Text → `input_ids` → `attention_mask`
#
# Now the tokenizer. One critical argument: **`add_special_tokens=False`**.
#
# The template *already* emitted `bos_token`. If we also let the tokenizer add its own
# special tokens we would get BOS twice — a real bug that shifts every position by one
# and is nearly invisible in a loss curve.
#
# (From M1: this tokenizer does not prepend BOS by default. Other families do. Never
# assume — check, as we do below.)

# %%
enc = tok(text, add_special_tokens=False, return_tensors="pt")
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]

print("input_ids     :", tuple(input_ids.shape), "[batch, seq]")
print("attention_mask:", tuple(attention_mask.shape))
print()
print("input_ids     :", input_ids[0].tolist())
print("attention_mask:", attention_mask[0].tolist())
print()
print("double-BOS check — how many BOS/EOS ids (100257) are present?",
      (input_ids[0] == tok.bos_token_id).sum().item())

# %% [markdown]
# `attention_mask` is all `1` here: every position is a real token. It only becomes
# interesting when we pad (Part 10). Its job is to tell attention *"ignore these
# positions entirely"*.
#
# **`attention_mask` and label masking are different mechanisms and solve different
# problems.** Mixing them up is the most common M2 misconception:
#
# | | `attention_mask = 0` | `labels = -100` |
# |---|---|---|
# | Effect | token is invisible to attention | token produces no loss |
# | Can other tokens see it? | **No** | **Yes** |
# | Used for | padding | prompt tokens |
#
# The prompt must remain **visible** (the model has to read the question!) while
# contributing **no loss**. That is exactly `attention_mask=1` with `labels=-100`.

# %% [markdown]
# ### Which of these are actually special tokens?
#
# Worth checking rather than assuming — the answer here is surprising.

# %%
for s in ("<|endoftext|>", "<|pad|>", "<|user|>", "<|assistant|>", "<|system|>"):
    enc_s = tok(s, add_special_tokens=False)["input_ids"]
    kind = "SINGLE special token" if len(enc_s) == 1 else "%d ordinary tokens" % len(enc_s)
    print("%-16s -> %-22s %s" % (s, kind, [tok.decode([i]) for i in enc_s]))

# %% [markdown]
# **Only `<|endoftext|>` and `<|pad|>` are real vocabulary entries.** The role markers are
# *plain text*: `<|user|>` is five tokens — `'<'`, `'|'`, `'user'`, `'|'`, `'>\n'`.
#
# Three consequences:
#
# 1. **Scaffolding is expensive.** In our 28-token example, 19 tokens are template
#    overhead and only 9 are the answer. Every training example pays that tax.
# 2. **The model must *learn* that this 5-token sequence marks a role boundary**, rather
#    than receiving it as one atomic symbol. It clearly does — but nothing made it easy.
# 3. **It justifies working in character space** (Part 6). Locating `<|assistant|>` by
#    token id would mean matching a 5-token pattern, and `'<'` and `'|'` appear in ordinary
#    text constantly.
#
# Recall from M1 that the embedding matrix is `[100352, 2048]` while the tokenizer holds
# 100278 entries — **74 unused rows**. Adding the role markers as genuine special tokens
# would fit comfortably, cost no `resize_token_embeddings()`, and cut ~8 tokens per turn.
#
# **TRY THIS.** Tempting, isn't it? Now explain why doing so would *break* this checkpoint.
# (Hint: those 74 rows are randomly initialised, and the model was post-trained with the
# markers split. What would `<|user|>` as a fresh single token mean to it?)

# %% [markdown]
# ## Part 4 — `labels`, and the naive mistake
#
# In M1 Part 6 we computed the pretraining loss by passing `labels=input_ids`. That is
# correct for *pretraining*: predict every token from the ones before it.
#
# For SFT it is **wrong**, and wrong in a way that still trains and still lowers the loss —
# the worst kind of bug. Let us be precise about what it would teach.

# %%
naive_labels = input_ids.clone()
ids_list = input_ids[0].tolist()

print("If labels == input_ids, the model is trained to produce ALL of this:")
print()
for i, tid in enumerate(ids_list):
    print("  %2d  %-10r <- trained to emit" % (i, tok.decode([tid])))

# %% [markdown]
# Read that list. We would be training the model to generate `<|user|>`, to generate the
# user's question, and to generate the `<|assistant|>` marker. None of those are ever the
# model's job — at inference the harness supplies all of them.
#
# The practical damage:
#
# 1. **Wasted capacity** on modelling text it will never need to produce.
# 2. **Learning to ask itself questions**, which surfaces as the model continuing past its
#   answer and inventing a new `<|user|>` turn.
# 3. **A diluted loss signal.** Here the prompt is most of the sequence, so most of the
#   gradient is about the wrong thing.

# %% [markdown]
# ## Part 5 — Assistant-only masking, and the three boundary questions
#
# The rule: **train on the assistant's tokens; mask everything else with `-100`.**
#
# Why `-100` specifically? It is the default `ignore_index` of
# `torch.nn.CrossEntropyLoss`. Positions whose label is `ignore_index` contribute
# **nothing** to the numerator *and* are excluded from the denominator of the mean — so
# they do not merely count as zero loss, they are not counted at all. Any negative
# sentinel would do; `-100` is simply the agreed-upon one.
#
# Now the three questions M1 left you with.
#
# **1. Should `<|assistant|>\n` itself be learned, or masked?**
# **Masked.** At inference the template supplies it (`add_generation_prompt=True`) before
# the model ever runs. It is a *cue to start*, not something to produce. Training on it
# spends gradient predicting a token that is always given for free.
#
# **2. Should the final `<|endoftext|>` be learned?**
# **Yes — and this is the one that matters most.** EOS is how the model learns to *stop*.
# Mask it and you get precisely the M1 Base-model behaviour: a fluent answer that runs on
# forever because nothing ever taught it to terminate. In M1 the Base model hit the
# 120-token cap and never emitted EOS; Instruct answered in 9 tokens and stopped. **That
# difference is largely this one label.**
#
# **3. In multi-turn, are earlier assistant turns learned or masked?**
# **Learned** — all of them. Every assistant turn is a valid demonstration, and masking
# all but the last throws away supervision you already paid for. (Some recipes do mask
# them; it is a deliberate trade, not the default.) We verify this in Part 9.

# %% [markdown]
# ## Part 6 — Building the mask robustly, with `offset_mapping`
#
# A tempting approach is to search the token list for the `<|assistant|>` id and slice.
# That breaks the moment content contains the marker text, and it silently mis-aligns when
# a template changes.
#
# The robust tool is **`offset_mapping`**: for each token, the fast tokenizer reports the
# `(char_start, char_end)` it came from. So we can work in **character space** — where the
# template structure is obvious — and map back to tokens.

# %%
ASSISTANT_MARK = "<|assistant|>\n"
EOS_STR = "<|endoftext|>"


def assistant_char_spans(s: str) -> list[tuple[int, int]]:
    """Character spans the model should learn: content + its terminating EOS.

    Everything after `<|assistant|>\\n` up to and including the next EOS.
    A trailing generation prompt (marker with no content) is skipped.
    """
    spans, i = [], 0
    while True:
        j = s.find(ASSISTANT_MARK, i)
        if j == -1:
            break
        start = j + len(ASSISTANT_MARK)
        k = s.find(EOS_STR, start)
        if k == -1:            # generation prompt: nothing generated yet
            break
        spans.append((start, k + len(EOS_STR)))   # include EOS -> learn to stop
        i = k + len(EOS_STR)
    return spans


spans = assistant_char_spans(text)
print("assistant spans (chars):", spans)
for a, b in spans:
    print("  ->", repr(text[a:b]))

# %% [markdown]
# Notice the span **includes** the trailing `<|endoftext|>`. That is question 2, made
# concrete in one slice index.

# %%
def build_labels(s: str, tokenizer) -> tuple[list[int], list[int]]:
    """Tokenize `s` and return (input_ids, labels) with assistant-only supervision."""
    e = tokenizer(s, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = e["input_ids"], e["offset_mapping"]
    keep = assistant_char_spans(s)

    labels = [-100] * len(ids)
    for i, (a, b) in enumerate(offs):
        if a == b:                      # zero-width: a special token with no source text
            continue
        if any(a >= s0 and b <= s1 for s0, s1 in keep):
            labels[i] = ids[i]
    return ids, labels


ids, labels = build_labels(text, tok)
n_train = sum(1 for x in labels if x != -100)
print("tokens total    :", len(ids))
print("tokens trained  :", n_train)
print("tokens masked   :", len(ids) - n_train)
print("supervision rate: %.1f%%" % (100 * n_train / len(ids)))

# %% [markdown]
# **TRY THIS.** That supervision rate is low — most of this example is prompt. Predict
# what happens to it as the answer gets longer, then test with a 200-word assistant reply.
# This ratio is exactly why SFT datasets favour substantial responses: you pay attention
# cost for the whole sequence but only learn from part of it.

# %% [markdown]
# ## Part 7 — One complete training example, token by token
#
# This is the table to stare at. It is the whole of M2 on one screen.

# %%
print("%3s  %-8s %-14s %-8s %s" % ("pos", "id", "token", "label", "trained?"))
print("-" * 62)
for i, (tid, lab) in enumerate(zip(ids, labels)):
    trained = "" if lab == -100 else "  <== LEARN"
    lab_s = "-100" if lab == -100 else str(lab)
    print("%3d  %-8d %-14r %-8s %s" % (i, tid, tok.decode([tid]), lab_s, trained))

# %% [markdown]
# Read down the `trained?` column. The model sees the whole sequence, but is graded only
# on the answer and the stop token.
#
# One subtlety worth internalising: **the labels are not shifted here.** In M1 Part 6 we
# shifted by hand (`logits[:, :-1]` against `labels[:, 1:]`). Hugging Face models do that
# shift *internally* when you pass `labels=`. So `labels[i]` is the target **at** position
# `i`, and the model's prediction for it comes from position `i-1`. Shifting yourself
# before passing to the model is a classic off-by-one.

# %% [markdown]
# ## Part 8 — Proof the mask actually changes the loss
#
# Talk is cheap. Compute both losses on the same logits and watch them differ.
# We use random logits so this runs instantly without a model — the arithmetic of
# `ignore_index` is what is on trial, not any particular model's predictions.

# %%
torch.manual_seed(0)
V = len(tok)
fake_logits = torch.randn(1, len(ids), V)

ids_t = torch.tensor([ids])
lab_full = ids_t.clone()                 # pretraining-style: learn everything
lab_mask = torch.tensor([labels])        # SFT-style: assistant only


def causal_loss(logits, labels):
    """Exactly what HF does internally: shift, then cross-entropy with ignore_index."""
    sl = logits[:, :-1, :].reshape(-1, logits.size(-1))
    tl = labels[:, 1:].reshape(-1)
    return F.cross_entropy(sl, tl, ignore_index=-100)


print("loss, all tokens      : %.4f   (over %d positions)"
      % (causal_loss(fake_logits, lab_full), len(ids) - 1))
print("loss, assistant only  : %.4f   (over %d positions)"
      % (causal_loss(fake_logits, lab_mask), sum(1 for x in labels[1:] if x != -100)))

# %% [markdown]
# Different numbers, from identical logits. The mask is not cosmetic: it changes what the
# gradient points at.
#
# And the denominator claim from Part 5, verified directly — `-100` positions are dropped
# from the mean, not averaged in as zeros:

# %%
per_tok = F.cross_entropy(
    fake_logits[:, :-1, :].reshape(-1, V),
    lab_mask[:, 1:].reshape(-1),
    ignore_index=-100,
    reduction="none",
)
kept = per_tok[per_tok != 0]
print("per-token losses at masked positions :", per_tok[per_tok == 0].tolist()[:5], "...")
print("mean over KEPT positions only        : %.4f" % kept.mean())
print("mean over ALL positions (wrong)      : %.4f" % per_tok.mean())
print()
print("reduction='mean' with ignore_index   : %.4f  <- matches KEPT, not ALL"
      % causal_loss(fake_logits, lab_mask))

# %% [markdown]
# ## Part 9 — Multi-turn
#
# Now the third boundary question, with a real two-turn conversation plus a system prompt.

# %%
MULTI = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4."},
    {"role": "user", "content": "And times 3?"},
    {"role": "assistant", "content": "12."},
]

mtext = tok.apply_chat_template(MULTI, tokenize=False)
print(repr(mtext))
print()
mids, mlabels = build_labels(mtext, tok)
print("assistant spans:", assistant_char_spans(mtext))
print()
print("%3s  %-14s %s" % ("pos", "token", "label"))
print("-" * 40)
for i, (tid, lab) in enumerate(zip(mids, mlabels)):
    print("%3d  %-14r %s" % (i, tok.decode([tid]), "-100" if lab == -100 else "LEARN"))

# %% [markdown]
# **Both** assistant turns are supervised, each including its own EOS. The system prompt
# and both user turns are masked. One conversation, two demonstrations — which is why
# multi-turn data is efficient training material.
#
# **TRY THIS.** Change `assistant_char_spans` to keep only the *last* span, and compare
# supervision rates. That is the "mask all but the final turn" recipe. When might throwing
# away that signal actually be the right call? (Hint: what if the earlier turns came from
# a weaker model?)

# %% [markdown]
# ## Part 10 — Padding, and why the collator exists
#
# Real batches hold sequences of different lengths, but a tensor is rectangular. So short
# sequences get padded — and **padding must be neutralised in two independent places**.

# %%
SHORT = [{"role": "user", "content": "Hi"},
         {"role": "assistant", "content": "Hello!"}]

examples = []
for c in (CONVO, MULTI, SHORT):
    t_ = tok.apply_chat_template(c, tokenize=False)
    i_, l_ = build_labels(t_, tok)
    examples.append({"input_ids": i_, "labels": l_})

for k, ex in enumerate(examples):
    print("example %d: %d tokens" % (k, len(ex["input_ids"])))

# %%
def collate(batch, pad_id, pad_to=None):
    """Pad a list of examples into rectangular tensors.

    THREE parallel paddings, and each uses a DIFFERENT filler:
      input_ids      <- pad_token_id   (must be a real id; the embedding is looked up)
      attention_mask <- 0              (invisible to attention)
      labels         <- -100           (contributes no loss)
    """
    L = pad_to or max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        n = L - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * n)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * n)
        out["labels"].append(b["labels"] + [-100] * n)
    return {k: torch.tensor(v) for k, v in out.items()}


batch = collate(examples, tok.pad_token_id)
for k, v in batch.items():
    print("%-15s %s" % (k, tuple(v.shape)))
print()
print("row 2 (shortest example), tail 12 positions:")
print("  input_ids     :", batch["input_ids"][2][-12:].tolist())
print("  attention_mask:", batch["attention_mask"][2][-12:].tolist())
print("  labels        :", batch["labels"][2][-12:].tolist())

# %% [markdown]
# Three different fillers, because they answer three different questions. Use `0` in
# `labels` instead of `-100` and you train the model to emit token id 0 after every
# answer. Forget `attention_mask` and real tokens attend to padding.
#
# **Dynamic vs static padding.** Above we padded to the longest sequence *in the batch*
# (dynamic). The alternative is padding every batch to a fixed `max_seq_len` (static).

# %%
dyn = collate(examples, tok.pad_token_id)
sta = collate(examples, tok.pad_token_id, pad_to=2048)
for name, b in (("dynamic", dyn), ("static-2048", sta)):
    total = b["input_ids"].numel()
    real = int(b["attention_mask"].sum())
    print("%-12s shape %-12s  real %4d / %5d tokens  (%.1f%% wasted)"
          % (name, str(tuple(b["input_ids"].shape)), real, total,
             100 * (1 - real / total)))

# %% [markdown]
# Dynamic padding wastes far less compute, which is why it is the default — but it makes
# every batch a different shape, which can defeat kernel caching and `torch.compile`.
# Static padding is predictable and wasteful. Most recipes use dynamic padding plus
# *length-grouped* sampling, so similar-length sequences batch together and the padding
# is small anyway.
#
# **TRY THIS.** Sort `examples` by length before collating and recompute the waste.

# %% [markdown]
# ## Part 11 — Truncation and packing
#
# **Truncation** caps sequence length. It is lossy, and *where* you cut matters: cut the
# tail and you may delete the entire answer, leaving an example whose labels are all
# `-100` — a training example that teaches nothing while still costing a forward pass.

# %%
MAXLEN = 12
trunc_ids = ids[:MAXLEN]
trunc_labels = labels[:MAXLEN]
kept_n = sum(1 for x in trunc_labels if x != -100)
print("truncated to %d tokens; supervised tokens remaining: %d" % (MAXLEN, kept_n))
print("is this example now useless?", kept_n == 0)

# %% [markdown]
# Worth checking for in any real pipeline: **drop examples whose labels are entirely
# `-100`**. They contribute nothing but consume batch slots.
#
# **Packing** attacks the same waste from the other side: concatenate several short
# examples into one full-length sequence so almost no padding is needed.

# %%
packed_ids, packed_labels = [], []
for ex in examples:
    packed_ids += ex["input_ids"]
    packed_labels += ex["labels"]

print("packed length          :", len(packed_ids))
print("sum of separate lengths:", sum(len(e["input_ids"]) for e in examples))
print("padding needed         : 0")
print()
print("vs dynamic batch above : %d slots, %d real (%.1f%% wasted)"
      % (dyn["input_ids"].numel(), int(dyn["attention_mask"].sum()),
         100 * (1 - int(dyn["attention_mask"].sum()) / dyn["input_ids"].numel())))

# %% [markdown]
# Packing is nearly free efficiency, with one catch: **by default, attention lets a packed
# sequence attend across the boundary into the previous example.** Example 2 can "see"
# example 1. Fixing this needs either a block-diagonal attention mask or FlashAttention's
# variable-length API.
#
# Whether that cross-contamination actually hurts is an empirical question — and a good
# candidate for an M6 ablation, rather than something to take on faith.

# %% [markdown]
# ## Part 12 — Where M3 begins
#
# You now have the exact tensors a training step consumes:
#
# ```
# conversation -> chat template -> input_ids    [batch, seq]
#                               -> attention_mask [batch, seq]
#                               -> labels       [batch, seq]   (-100 = ignore)
# ```
#
# M3 feeds these to a model and closes the loop that has been open since M0:
#
# ```
# input_ids -> forward -> logits -> loss -> backward -> gradients -> optimizer -> new weights
# ```
#
# Predict these before M3, and write the answers down:
#
# 1. `logits` will be `[batch, seq, vocab]` and `labels` is `[batch, seq]`. What exactly
#    does cross-entropy do to reconcile those shapes?
# 2. If `loss.backward()` runs on a batch where every label is `-100`, what is the
#    gradient? Does the optimizer step still change the weights?
# 3. Our supervision rate in Part 6 was low. Does the backward pass cost less for a
#    heavily-masked batch than for an unmasked one, or the same?
#
# **Before moving on, rebuild `build_labels` from an empty file.** You have read it; that
# is not the same as being able to write it. If it comes out wrong, that is the signal —
# reread Part 6 and try again. This is the function every later milestone depends on.
