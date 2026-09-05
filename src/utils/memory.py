"""Training-memory accounting.

WHY THIS FILE EXISTS
--------------------
"Will this fit in my GPU?" is the question you will ask before every single run in
this project, and CUDA OOM is the error you will hit most often. Guessing is
expensive: you rent a GPU by the hour, launch a job, and it dies 90 seconds in.

This module makes the accounting explicit so that OOM stops being mysterious. It
answers: for a model of this size, at this precision, with this optimizer, batch
size and sequence length, WHERE does the VRAM go?

IMPORTANT HONESTY NOTE
----------------------
Everything here is an *estimate from first principles*. The parameter/gradient/
optimizer terms are near-exact (they are simple arithmetic). The activation term is
an approximation with a documented constant. In M6 and M20 we will MEASURE real
`torch.cuda.max_memory_allocated()` and correct the constants against reality.
Never present these numbers as measurements. See Rule 9 / Rule 10.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1024**3

#: Bytes per element for each dtype we care about.
DTYPE_BYTES: dict[str, int] = {"fp32": 4, "tf32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}  # type: ignore[dict-item]

#: Rough number of hidden-size-elements saved per token per layer for the backward
#: pass, assuming a modern SwiGLU + GQA block with FlashAttention (so no materialized
#: [seq, seq] attention matrix). Derived in docs/training_concepts.md. TO BE MEASURED.
ACT_ELEMS_PER_HIDDEN = 18


@dataclass
class ModelShape:
    """Architecture, in the only terms that affect memory."""

    name: str
    hidden: int
    layers: int
    vocab: int
    intermediate: int
    n_heads: int
    n_kv_heads: int  # < n_heads means Grouped-Query Attention: smaller K/V, less memory
    tied_embeddings: bool = True

    def n_params(self) -> int:
        """Total parameter count, derived from architecture (not read off a model card).

        Doing this by hand once is worth it: it shows you exactly where a model's
        parameters live, which is the same thing as knowing where your VRAM goes.
        """
        head_dim = self.hidden // self.n_heads
        kv_dim = head_dim * self.n_kv_heads

        embed = self.vocab * self.hidden
        lm_head = 0 if self.tied_embeddings else self.vocab * self.hidden

        attn = (
            self.hidden * self.hidden          # q_proj
            + self.hidden * kv_dim             # k_proj  (shrunk by GQA)
            + self.hidden * kv_dim             # v_proj  (shrunk by GQA)
            + self.hidden * self.hidden        # o_proj
        )
        mlp = 3 * self.hidden * self.intermediate  # SwiGLU: gate, up, down
        norms = 2 * self.hidden                    # RMSNorm weights, negligible but real

        return embed + lm_head + self.layers * (attn + mlp + norms) + self.hidden


#: Presets. VERIFY these against the real HF config in M1 — do not trust them blindly.
#: `scripts/memory_math.py --from-hf <repo_id>` reads the true config instead.
PRESETS: dict[str, ModelShape] = {
    "olmo2-1b": ModelShape("OLMo-2-0425-1B", hidden=2048, layers=16, vocab=100352,
                           intermediate=8192, n_heads=16, n_kv_heads=16, tied_embeddings=False),
    "olmo2-7b": ModelShape("OLMo-2-1124-7B", hidden=4096, layers=32, vocab=100352,
                           intermediate=11008, n_heads=32, n_kv_heads=32, tied_embeddings=False),
    "qwen3-1.7b": ModelShape("Qwen3-1.7B", hidden=2048, layers=28, vocab=151936,
                             intermediate=6144, n_heads=16, n_kv_heads=8, tied_embeddings=True),
}


@dataclass
class MemoryBudget:
    """Per-component VRAM estimate, in bytes."""

    params: float
    gradients: float
    optimizer: float
    master_weights: float
    activations: float
    logits: float
    overhead: float

    @property
    def total(self) -> float:
        return (self.params + self.gradients + self.optimizer + self.master_weights
                + self.activations + self.logits + self.overhead)

    def rows(self) -> list[tuple[str, float, str]]:
        t = self.total
        items = [
            ("Model parameters", self.params, "the weights themselves"),
            ("Gradients", self.gradients, "one number per TRAINABLE param"),
            ("Optimizer states", self.optimizer, "AdamW keeps 2 per trainable param"),
            ("FP32 master weights", self.master_weights, "mixed-precision only"),
            ("Activations", self.activations, "saved forward tensors, needed by backward"),
            ("Logits + their grad", self.logits, "[batch, seq, VOCAB] in fp32 — often the surprise"),
            ("CUDA ctx + fragmentation", self.overhead, "kernels, allocator slack"),
        ]
        return [(n, v, f"{100 * v / t:4.1f}%  {d}") for n, v, d in items]


def estimate(
    shape: ModelShape,
    *,
    batch_size: int = 1,
    seq_len: int = 2048,
    param_dtype: str = "bf16",
    mixed_precision: bool = True,
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
    flash_attention: bool = True,
    trainable_fraction: float = 1.0,
    inference_only: bool = False,
) -> MemoryBudget:
    """Estimate peak training VRAM.

    Args:
        param_dtype: storage dtype of the weights ("bf16" or "fp32").
        mixed_precision: if True, an FP32 copy of the weights is kept for the optimizer
            update. This is what "bf16 mixed precision" actually means — compute in
            bf16 for speed, but accumulate updates in fp32 so tiny LR*grad steps are
            not rounded to zero.
        trainable_fraction: 1.0 = full fine-tuning. LoRA on a 1B model is ~0.001-0.01.
            Gradients and optimizer states scale with this; parameters do NOT.
        inference_only: drop gradients/optimizer/master weights entirely.

    Returns:
        MemoryBudget in bytes.
    """
    p = shape.n_params()
    trainable = p * trainable_fraction
    pb = DTYPE_BYTES[param_dtype]

    params = p * pb

    if inference_only:
        # Inference also needs a KV cache, which we ignore here; it is a separate
        # topic we will cover in M25 (serving).
        act = batch_size * seq_len * shape.hidden * pb * 2
        logits = batch_size * 1 * shape.vocab * 4  # only the last position matters when decoding
        return MemoryBudget(params, 0, 0, 0, act, logits, 0.8 * GB)

    gradients = trainable * pb
    opt_slots = {"adamw": 2, "adam": 2, "sgd_momentum": 1, "sgd": 0, "adafactor": 0.1}[optimizer]
    optimizer_mem = trainable * opt_slots * 4  # AdamW states are kept in fp32
    master = trainable * 4 if (mixed_precision and pb < 4) else 0

    # --- activations ---
    per_layer_tokens = batch_size * seq_len * shape.hidden * pb
    if gradient_checkpointing:
        # Store only each layer's INPUT; recompute the interior during backward.
        # Cost: ~30% more compute. Benefit: activation memory becomes ~linear in layers
        # with a tiny constant instead of a ~18x one.
        activations = shape.layers * per_layer_tokens + ACT_ELEMS_PER_HIDDEN * per_layer_tokens
    else:
        activations = shape.layers * ACT_ELEMS_PER_HIDDEN * per_layer_tokens

    if not flash_attention:
        # The materialized attention probability matrix: [batch, heads, seq, seq].
        # This is the term that makes long context quadratic and why FlashAttention exists.
        activations += shape.layers * batch_size * shape.n_heads * seq_len * seq_len * pb

    # --- logits ---
    # HF upcasts logits to fp32 before cross-entropy for numerical stability.
    # [batch, seq, vocab] fp32, plus a same-sized gradient tensor in the backward pass.
    logits = 2 * batch_size * seq_len * shape.vocab * 4

    overhead = 1.2 * GB  # CUDA context + cuBLAS workspaces + allocator fragmentation

    return MemoryBudget(params, gradients, optimizer_mem, master, activations, logits, overhead)
