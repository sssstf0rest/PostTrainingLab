#!/usr/bin/env python3
"""M1 EXERCISE — rebuild the core of M1 from a blank page.

HOW TO USE THIS FILE
--------------------
1. Do NOT open scripts/explore_base_model.py. That is the answer key.
2. Fill in the four functions below. Each is 3-15 lines.
3. Run it. Every function is checked against ground truth computed a different way,
   so you get PASS/FAIL, not a feeling.

    python exercises/m1_rebuild.py --device cpu --dtype float32

If you get stuck for more than ~10 minutes on one, that is a signal about which
concept is still thin -- note WHICH one, then look it up. The stuck point is the
information, not the failure.

Everything here runs fine on CPU. No GPU needed.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXT = "The capital of France is Paris."


# ===========================================================================
# EXERCISE 1 — the pretraining loss.  THE important one.
# ===========================================================================
def clm_loss(model, tok, text: str) -> float:
    """Return the mean causal-LM cross-entropy loss for `text`, in nats/token.

    You must compute this YOURSELF from logits. Do not pass `labels=` to the
    model and read `.loss` -- that is the thing being checked against.

    Hints about shape, not about code:
      - the model gives you logits of shape [batch, seq, vocab]
      - position t is the model's prediction for the token at position t+1
      - so one end of the logits and the other end of the labels do not
        participate; work out which
      - F.cross_entropy wants [N, vocab] and [N]

    Returns: a Python float.
    """
    raise NotImplementedError("EXERCISE 1")


# ===========================================================================
# EXERCISE 2 — per-token loss
# ===========================================================================
def per_token_loss(model, tok, text: str) -> list[tuple[str, float]]:
    """Return [(decoded_token, loss_in_nats), ...] for each PREDICTED token.

    Same shift as exercise 1, but do not average. The list is shorter than the
    number of input tokens -- by exactly how many, and why?

    The mean of the losses you return must equal exercise 1's answer.
    """
    raise NotImplementedError("EXERCISE 2")


# ===========================================================================
# EXERCISE 3 — next-token distribution
# ===========================================================================
def top_k_next(model, tok, text: str, k: int = 5) -> list[tuple[str, float]]:
    """Return the k most likely NEXT tokens after `text`, as (token, probability).

    Probabilities, not logits -- they must sum to <= 1 across the whole vocab.
    Which position of the logits tensor holds the prediction for the token that
    comes after the end of `text`?
    """
    raise NotImplementedError("EXERCISE 3")


# ===========================================================================
# EXERCISE 4 — prove causality
# ===========================================================================
def causality_holds(model, tok, text: str) -> bool:
    """Return True iff editing the LAST token leaves all earlier logits unchanged.

    Design the experiment yourself: build two input_ids that share every position
    but the last, run both, and compare the right slices.

    Think first: if you instead changed a MIDDLE token, which positions' logits
    would move? Write your prediction in a comment before you code it.
    """
    raise NotImplementedError("EXERCISE 4")


# ===========================================================================
# CHECKS — ground truth computed independently of your code
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()

    ids = tok(TEXT, return_tensors="pt").to(args.device)["input_ids"]
    passed = failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nChecking against {args.model} on {args.device}\n" + "-" * 66)

    # --- truth: let the model compute its own loss --------------------------
    with torch.no_grad():
        ref_loss = model(input_ids=ids, labels=ids).loss.item()

    # Exercise 1
    try:
        got = clm_loss(model, tok, TEXT)
        check("1 clm_loss", abs(got - ref_loss) < 1e-3, f"yours={got:.4f} ref={ref_loss:.4f}")
    except NotImplementedError:
        check("1 clm_loss", False, "not implemented")

    # Exercise 2
    try:
        rows = per_token_loss(model, tok, TEXT)
        n_ok = len(rows) == ids.shape[1] - 1
        mean_ok = abs(sum(v for _, v in rows) / len(rows) - ref_loss) < 1e-3
        check("2 per_token_loss length", n_ok, f"got {len(rows)}, expected {ids.shape[1] - 1}")
        check("2 per_token_loss mean", mean_ok, f"mean={sum(v for _, v in rows) / len(rows):.4f}")
        if n_ok and mean_ok:
            print("      your per-token breakdown:")
            for t_, v in rows:
                print("        %-12r %6.3f %s" % (t_, v, "#" * min(int(v * 4), 40)))
    except NotImplementedError:
        check("2 per_token_loss", False, "not implemented")

    # Exercise 3
    try:
        top = top_k_next(model, tok, TEXT, k=5)
        ok = len(top) == 5 and all(0.0 <= p <= 1.0 for _, p in top)
        desc = all(top[i][1] >= top[i + 1][1] for i in range(len(top) - 1))
        check("3 top_k_next", ok and desc, str([(t_, round(p, 4)) for t_, p in top]))
    except NotImplementedError:
        check("3 top_k_next", False, "not implemented")

    # Exercise 4
    try:
        check("4 causality_holds", causality_holds(model, tok, TEXT) is True)
    except NotImplementedError:
        check("4 causality_holds", False, "not implemented")

    print("-" * 66)
    print(f"  {passed} passed, {failed} failed\n")


if __name__ == "__main__":
    main()
