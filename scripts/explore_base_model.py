#!/usr/bin/env python3
"""M1 — Base-model exploration.

WHY THIS FILE EXISTS
--------------------
Before training anything we must be able to answer, from direct observation rather
than from a diagram:

  * What does a tokenizer actually turn text into?
  * What shape are the logits, and what do they mean?
  * What does a BASE model do when you ask it a question?
  * How is that different from an INSTRUCT model?

Every later milestone manipulates these same objects. This script prints them.

    python scripts/explore_base_model.py --model allenai/OLMo-2-0425-1B
    python scripts/explore_base_model.py --model allenai/OLMo-2-0425-1B \
        --compare allenai/OLMo-2-0425-1B-Instruct

Pin GPUs before running, e.g.  CUDA_VISIBLE_DEVICES=1 python scripts/...
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

W = 78
PROMPT = "Who are you?"


def rule(title: str = "") -> None:
    print("\n" + "=" * W)
    if title:
        print("  " + title)
        print("=" * W)


def load(model_id: str, device: str, dtype: torch.dtype):
    """Load tokenizer + model. Returns (tok, model)."""
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:  # older transformers spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device).eval()
    return tok, model


# ---------------------------------------------------------------------------
# 1. Tokenizer
# ---------------------------------------------------------------------------
def inspect_tokenizer(tok, model_id: str) -> None:
    rule(f"1. TOKENIZER — {model_id}")
    print(f"  class            : {type(tok).__name__}")
    print(f"  vocab_size       : {tok.vocab_size}")
    print(f"  len(tokenizer)   : {len(tok)}   <- includes added special tokens")
    print(f"  bos / eos / pad  : {tok.bos_token!r} / {tok.eos_token!r} / {tok.pad_token!r}")
    print(f"  bos/eos/pad ids  : {tok.bos_token_id} / {tok.eos_token_id} / {tok.pad_token_id}")

    # THE central object: text -> integers.
    enc = tok(PROMPT)
    ids = enc["input_ids"]
    print(f"\n  raw text         : {PROMPT!r}")
    print(f"  input_ids        : {ids}")
    print(f"  attention_mask   : {enc['attention_mask']}   <- 1 = real token, 0 = padding")
    print(f"  n_tokens         : {len(ids)}")
    print("\n  token-by-token (note the leading spaces — they are part of the token):")
    for i, tid in enumerate(ids):
        print("    %2d  id=%-7d %r" % (i, tid, tok.decode([tid])))

    # Does the tokenizer prepend BOS automatically? Easy to get wrong, matters in M2.
    print(f"\n  starts with BOS? : {ids[0] == tok.bos_token_id if tok.bos_token_id is not None else 'no bos token'}")
    print(f"  round-trip ok?   : {tok.decode(ids) == PROMPT}   decode -> {tok.decode(ids)!r}")

    # Chat template: present on Instruct models, absent on Base. This is the
    # single clearest signal of what post-training added. M2 depends on it.
    tmpl = getattr(tok, "chat_template", None)
    print(f"\n  chat_template    : {'PRESENT (' + str(len(tmpl)) + ' chars)' if tmpl else 'None  <- a BASE model has no notion of roles'}")


# ---------------------------------------------------------------------------
# 2. Model structure
# ---------------------------------------------------------------------------
def inspect_model(model, model_id: str) -> None:
    rule(f"2. MODEL STRUCTURE — {model_id}")
    n = sum(p.numel() for p in model.parameters())
    print(f"  parameters       : {n / 1e9:.4f} B  ({n:,})")
    print(f"  dtype            : {next(model.parameters()).dtype}")
    print(f"  device           : {next(model.parameters()).device}")

    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    print(f"\n  embed_tokens     : {tuple(inp.weight.shape)}")
    print(f"  lm_head          : {tuple(out.weight.shape)}")
    # data_ptr is the ground truth for weight tying: same address = same tensor.
    # The config flag states intent; this states reality.
    tied = inp.weight.data_ptr() == out.weight.data_ptr()
    print(f"  same storage?    : {tied}   <- True would mean TIED embeddings")
    if not tied:
        extra = inp.weight.numel()
        print(f"  cost of untying  : {extra / 1e6:.1f} M params = {extra * 16 / 1e9:.2f} GB "
              f"of training memory at 16 bytes/param (mixed-precision AdamW)")


# ---------------------------------------------------------------------------
# 3. Forward pass -> logits
# ---------------------------------------------------------------------------
@torch.no_grad()
def inspect_forward(tok, model, device: str) -> None:
    rule("3. FORWARD PASS — what the model actually outputs")
    enc = tok(PROMPT, return_tensors="pt").to(device)
    ids = enc["input_ids"]
    out = model(**enc)
    logits = out.logits

    print(f"  input_ids shape  : {tuple(ids.shape)}                 [batch, seq]")
    print(f"  logits shape     : {tuple(logits.shape)}   [batch, seq, VOCAB]")
    print(f"  logits dtype     : {logits.dtype}")
    print(f"  logits MB        : {logits.numel() * logits.element_size() / 1e6:.1f} MB for ONE short prompt")
    print("\n  One score per vocabulary entry, at EVERY position. Position t predicts")
    print("  token t+1 — that is the whole of causal language modelling.")

    # Distribution for the NEXT token = last position.
    last = logits[0, -1].float()
    probs = torch.softmax(last, dim=-1)
    top = torch.topk(probs, 10)
    print(f"\n  Top-10 predictions to follow {PROMPT!r}:")
    for rank, (p, i) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        print("    %2d  p=%.4f  id=%-7d %r" % (rank, p, i, tok.decode([i])))

    ent = -(probs * probs.clamp_min(1e-12).log()).sum()
    print(f"\n  entropy          : {ent:.3f} nats  (uniform over {len(probs)} would be {torch.tensor(float(len(probs))).log():.3f})")


# ---------------------------------------------------------------------------
# 4. Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def inspect_generation(tok, model, device: str, label: str, use_chat_template: bool) -> str:
    rule(f"4. GENERATION — {label}")
    if use_chat_template and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
        )
        print("  prompt was wrapped in the model's chat template:")
        print("  " + repr(text))
    else:
        text = PROMPT
        print(f"  raw prompt (no template): {text!r}")

    enc = tok(text, return_tensors="pt").to(device)
    gen = model.generate(
        **enc,
        max_new_tokens=120,
        do_sample=False,  # greedy => reproducible
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    new = gen[0][enc["input_ids"].shape[1]:]
    completion = tok.decode(new, skip_special_tokens=True)
    stopped = new[-1].item() == tok.eos_token_id
    print(f"\n  generated {len(new)} new tokens; hit EOS on its own? {stopped}")
    print("  " + "-" * (W - 4))
    for line in completion.splitlines() or [""]:
        print("  | " + line[:W - 6])
    print("  " + "-" * (W - 4))
    return completion


# ---------------------------------------------------------------------------
# 5. Causality — proof, not assertion
# ---------------------------------------------------------------------------
@torch.no_grad()
def inspect_causality(tok, model, device: str) -> None:
    rule("5. CAUSALITY — proving position t cannot see position t+1")
    a = tok("The capital of France is Paris", return_tensors="pt").to(device)
    ids = a["input_ids"].clone()

    # Same prefix, different FINAL token.
    b = ids.clone()
    b[0, -1] = tok(" Berlin")["input_ids"][-1]

    la = model(input_ids=ids).logits
    lb = model(input_ids=b).logits

    # Every position except the last shares an identical prefix, so if attention
    # is causal those logits must be bit-identical.
    same_prefix = torch.equal(la[0, :-1], lb[0, :-1])
    last_differs = not torch.equal(la[0, -1], lb[0, -1])
    print(f"  changed only the LAST token: {tok.decode(ids[0, -1:])!r} -> {tok.decode(b[0, -1:])!r}")
    print(f"  logits at positions 0..n-2 identical? {same_prefix}   <- must be True")
    print(f"  logits at the last position differ?   {last_differs}   <- must be True")
    print("\n  That is the causal mask, observed. Each position attends only to itself")
    print("  and everything BEFORE it, which is why one forward pass can score every")
    print("  next-token prediction in the sequence simultaneously (teacher forcing).")


# ---------------------------------------------------------------------------
# 6. The pretraining objective — the bridge to M2/M3
# ---------------------------------------------------------------------------
@torch.no_grad()
def inspect_loss(tok, model, device: str) -> None:
    rule("6. THE PRETRAINING LOSS — what next-token prediction actually optimises")
    text = "The capital of France is Paris."
    enc = tok(text, return_tensors="pt").to(device)
    ids = enc["input_ids"]

    # In causal LM pretraining, labels ARE the inputs. The model shifts internally.
    out = model(input_ids=ids, labels=ids)
    print(f"  text            : {text!r}")
    print(f"  input_ids       : {tuple(ids.shape)}")
    print(f"  labels          : same tensor as input_ids  <- pretraining predicts the input")
    print(f"  model loss      : {out.loss.item():.4f}   (mean cross-entropy, nats/token)")
    print(f"  perplexity      : {out.loss.exp().item():.2f}")

    # Recompute by hand to remove the magic.
    logits = out.logits.float()
    shift_logits = logits[:, :-1, :]   # drop the LAST position: nothing follows it
    shift_labels = ids[:, 1:]          # drop the FIRST token: nothing predicts it
    manual = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )
    print(f"\n  manual recompute: {manual.item():.4f}   <- must match the model loss")
    print(f"  shift_logits    : {tuple(shift_logits.shape)}  [batch, seq-1, vocab]")
    print(f"  shift_labels    : {tuple(shift_labels.shape)}       [batch, seq-1]")
    print("\n  L = -(1/N) * sum_t log P(y_t | y_<t)")

    # Per-token detail: which words was the model surprised by?
    per_tok = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    )
    print("\n  per-token loss (high = surprised):")
    for i, (tid, l) in enumerate(zip(shift_labels[0].tolist(), per_tok.tolist())):
        bar = "#" * min(int(l * 4), 40)
        print("    %-12r %6.3f  %s" % (tok.decode([tid]), l, bar))
    print("\n  M2/M3 PREVIEW: SFT uses this EXACT loss. The only change is that labels")
    print("  for the prompt are replaced with -100 so they contribute nothing.")


# ---------------------------------------------------------------------------
# 7. Embeddings — the vectors themselves
# ---------------------------------------------------------------------------
@torch.no_grad()
def inspect_embeddings(tok, model) -> None:
    rule("7. EMBEDDINGS — what a token ID becomes")
    E = model.get_input_embeddings().weight
    word = " Paris"
    tid = tok(word)["input_ids"][-1]
    v = E[tid]
    print(f"  embedding matrix : {tuple(E.shape)}   [vocab, hidden]")
    print(f"  token {word!r} -> id {tid} -> vector of shape {tuple(v.shape)}")
    print(f"  first 8 values   : {[round(x, 4) for x in v[:8].float().tolist()]}")
    print(f"  L2 norm          : {v.float().norm().item():.3f}")
    print("\n  The lookup is literally row-indexing this matrix. No computation.")

    # Is the space meaningful? Nearest neighbours by cosine similarity.
    En = F.normalize(E.float(), dim=-1)
    sims = En @ En[tid]
    top = torch.topk(sims, 9)
    print(f"\n  nearest neighbours of {word!r} by cosine similarity:")
    for s, i in zip(top.values.tolist(), top.indices.tolist()):
        if i == tid:
            continue
        print("    %.4f  %r" % (s, tok.decode([i])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--compare", default=None, help="second model id, e.g. the Instruct checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--allow-any-gpu", action="store_true",
                    help="skip the CUDA_VISIBLE_DEVICES guard (only if you own the machine)")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)

    # --- shared-box safety -------------------------------------------------
    # "cuda" means cuda:0 = the first VISIBLE device. On a machine shared with
    # colleagues, running without pinning silently lands on physical GPU 0,
    # which is almost certainly someone else's. Refuse rather than intrude.
    if args.device.startswith("cuda"):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible and not args.allow_any_gpu:
            raise SystemExit(
                "\nREFUSING TO RUN: CUDA_VISIBLE_DEVICES is not set.\n"
                "This is a shared machine; 'cuda' would grab physical GPU 0.\n"
                "Pick an idle GPU first:\n"
                "    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv\n"
                "then re-run, e.g.:\n"
                "    CUDA_VISIBLE_DEVICES=5 python scripts/explore_base_model.py ...\n"
                "(override with --allow-any-gpu only if you own the whole box.)\n"
            )
        print(f"device={args.device}  dtype={dtype}")
        print(f"CUDA_VISIBLE_DEVICES={visible or 'UNSET'}  -> torch cuda:0 is physical GPU "
              f"{visible.split(',')[0] if visible else '0'}")
        print(f"GPU: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}")
    else:
        print(f"device={args.device}  dtype={dtype}")

    tok, model = load(args.model, args.device, dtype)
    inspect_tokenizer(tok, args.model)
    inspect_model(model, args.model)
    inspect_forward(tok, model, args.device)
    inspect_causality(tok, model, args.device)
    inspect_loss(tok, model, args.device)
    inspect_embeddings(tok, model)
    inspect_generation(tok, model, args.device, f"BASE — {args.model}", use_chat_template=False)

    if args.compare:
        del model
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None
        tok2, model2 = load(args.compare, args.device, dtype)
        inspect_tokenizer(tok2, args.compare)
        inspect_model(model2, args.compare)
        inspect_generation(tok2, model2, args.device, f"INSTRUCT — {args.compare}", use_chat_template=True)

    rule("DONE")


if __name__ == "__main__":
    main()
