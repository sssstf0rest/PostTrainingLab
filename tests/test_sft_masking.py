"""Contract tests for the M2 label-masking pipeline.

WHY THIS FILE EXISTS
--------------------
`notebooks/m2_data_pipeline.py` *explains* assistant-only masking. Reading an explanation
is not the same as being able to write the code. These tests define the contract so you
can rebuild the implementation from an empty file and get a hard pass/fail instead of a
feeling.

TO MAKE THEM PASS, create `src/data/sft.py` exporting:

    build_labels(text: str, tokenizer) -> tuple[list[int], list[int]]
        Tokenize an already-templated conversation string and return
        (input_ids, labels) where labels are -100 everywhere except the
        assistant's own tokens, INCLUDING each turn's terminating EOS.
        Use add_special_tokens=False: the template already emitted BOS.

    collate(batch: list[dict], pad_id: int, pad_to: int | None = None) -> dict
        Pad a list of {"input_ids", "labels"} into rectangular torch tensors
        with keys input_ids / attention_mask / labels. Three different fillers:
        pad_id, 0, and -100 respectively.

Run:  RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false pytest tests/test_sft_masking.py -v

Do NOT copy the notebook cell. Write it, run these, fix what breaks.
"""

from __future__ import annotations

import os

os.environ.setdefault("RAYON_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sft = pytest.importorskip(
    "src.data.sft",
    reason="M2 exercise: create src/data/sft.py with build_labels() and collate(). "
           "See this file's docstring for the contract.",
)

from transformers import AutoTokenizer  # noqa: E402

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
IGNORE = -100


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def single(tok):
    convo = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 equals 4."},
    ]
    text = tok.apply_chat_template(convo, tokenize=False)
    ids, labels = sft.build_labels(text, tok)
    return text, ids, labels


@pytest.fixture(scope="module")
def multi(tok):
    convo = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "And times 3?"},
        {"role": "assistant", "content": "12."},
    ]
    text = tok.apply_chat_template(convo, tokenize=False)
    ids, labels = sft.build_labels(text, tok)
    return text, ids, labels


# --- shape / sanity ---------------------------------------------------------

def test_lengths_match(single):
    _, ids, labels = single
    assert len(ids) == len(labels), "labels must be parallel to input_ids"


def test_no_double_bos(single, tok):
    _, ids, _ = single
    # The template emits BOS itself. add_special_tokens=False must be used, or
    # position 0 gets duplicated and every position shifts by one.
    assert ids[0] == tok.bos_token_id
    assert ids[1] != tok.bos_token_id, "double BOS: pass add_special_tokens=False"


def test_something_is_masked_and_something_is_not(single):
    _, _, labels = single
    assert any(x == IGNORE for x in labels), "nothing was masked"
    assert any(x != IGNORE for x in labels), "everything was masked"


def test_unmasked_labels_equal_their_input_ids(single):
    _, ids, labels = single
    for i, lab in enumerate(labels):
        if lab != IGNORE:
            assert lab == ids[i], f"label at {i} must copy input_ids, not be shifted"


# --- the actual masking rule ------------------------------------------------

def test_assistant_content_is_learned(single, tok):
    _, ids, labels = single
    learned = tok.decode([i for i, l in zip(ids, labels) if l != IGNORE])
    assert "equals" in learned, f"assistant content not supervised; got {learned!r}"


def test_prompt_is_masked(single, tok):
    _, ids, labels = single
    learned = tok.decode([i for i, l in zip(ids, labels) if l != IGNORE])
    assert "What is" not in learned, "the user's question must not be supervised"
    assert "user" not in learned, "the <|user|> marker must not be supervised"


def test_assistant_marker_itself_is_masked(single, tok):
    """Boundary question 1: <|assistant|>\\n is a cue supplied at inference, not output."""
    _, ids, labels = single
    learned = tok.decode([i for i, l in zip(ids, labels) if l != IGNORE])
    assert "assistant" not in learned, "the <|assistant|> marker must be masked"


def test_final_eos_is_learned(single, tok):
    """Boundary question 2 — the most important test in this file.

    If EOS is masked the model never learns to stop, which is exactly the M1
    Base-model failure: a fluent answer that runs until the token cap.
    """
    _, ids, labels = single
    assert ids[-1] == tok.eos_token_id, "templated training text should end with EOS"
    assert labels[-1] != IGNORE, "EOS must be SUPERVISED or the model never stops"


def test_bos_is_not_learned(single):
    """bos_token_id == eos_token_id here, so naive 'supervise every EOS id' is wrong."""
    _, _, labels = single
    assert labels[0] == IGNORE, "position 0 is BOS, not a stop token; must be masked"


# --- multi-turn -------------------------------------------------------------

def test_all_assistant_turns_are_learned(multi, tok):
    """Boundary question 3: earlier assistant turns are demonstrations too."""
    _, ids, labels = multi
    learned = tok.decode([i for i, l in zip(ids, labels) if l != IGNORE])
    assert "4" in learned and "12" in learned, (
        f"both assistant turns must be supervised; got {learned!r}"
    )


def test_system_and_all_user_turns_masked(multi, tok):
    _, ids, labels = multi
    learned = tok.decode([i for i, l in zip(ids, labels) if l != IGNORE])
    assert "terse" not in learned, "system prompt must be masked"
    assert "times" not in learned, "later user turns must be masked"


def test_each_turn_keeps_its_own_eos(multi, tok):
    _, ids, labels = multi
    eos_positions = [i for i, t in enumerate(ids) if t == tok.eos_token_id]
    # position 0 is BOS; the rest terminate assistant turns
    supervised_eos = [i for i in eos_positions if i != 0 and labels[i] != IGNORE]
    assert len(supervised_eos) == 2, (
        f"expected 2 supervised EOS (one per assistant turn), got {len(supervised_eos)}"
    )


# --- generation prompt ------------------------------------------------------

def test_generation_prompt_supervises_nothing(tok):
    """An inference-shaped string has no assistant content, so nothing is learnable."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids, labels = sft.build_labels(text, tok)
    assert len(ids) == len(labels)
    assert all(x == IGNORE for x in labels), (
        "a generation prompt has no assistant tokens; everything must be masked"
    )


# --- collation --------------------------------------------------------------

def test_collate_pads_three_ways(tok):
    a = {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]}
    b = {"input_ids": [7, 8], "labels": [-100, 8]}
    out = sft.collate([a, b], pad_id=tok.pad_token_id)

    assert set(out) >= {"input_ids", "attention_mask", "labels"}
    assert out["input_ids"].shape == (2, 5)
    assert out["attention_mask"].shape == (2, 5)
    assert out["labels"].shape == (2, 5)

    # row 1 is the short one: 2 real tokens then 3 pads
    assert out["input_ids"][1].tolist() == [7, 8] + [tok.pad_token_id] * 3
    assert out["attention_mask"][1].tolist() == [1, 1, 0, 0, 0]
    assert out["labels"][1].tolist() == [-100, 8, -100, -100, -100], (
        "padded label positions must be -100, NOT 0 and NOT pad_token_id"
    )


def test_collate_static_padding(tok):
    a = {"input_ids": [1, 2], "labels": [-100, 2]}
    out = sft.collate([a], pad_id=tok.pad_token_id, pad_to=16)
    assert out["input_ids"].shape == (1, 16)
    assert int(out["attention_mask"].sum()) == 2


# --- the loss actually behaves as claimed -----------------------------------

def test_ignore_index_excludes_from_the_mean(single):
    """-100 positions are dropped from the denominator, not averaged in as zeros."""
    import torch.nn.functional as F

    _, ids, labels = single
    V = 1000
    torch.manual_seed(0)
    logits = torch.randn(1, len(ids), V)
    lab = torch.tensor([[min(x, V - 1) if x != IGNORE else IGNORE for x in labels]])

    shifted_logits = logits[:, :-1, :].reshape(-1, V)
    shifted_labels = lab[:, 1:].reshape(-1)

    mean_loss = F.cross_entropy(shifted_logits, shifted_labels, ignore_index=IGNORE)
    per_tok = F.cross_entropy(
        shifted_logits, shifted_labels, ignore_index=IGNORE, reduction="none"
    )
    kept = per_tok[shifted_labels != IGNORE]

    assert torch.allclose(mean_loss, kept.mean(), atol=1e-5), (
        "mean must be over supervised positions only"
    )
    assert not torch.allclose(mean_loss, per_tok.mean(), atol=1e-3), (
        "if these match, masked positions are being averaged in as zeros"
    )
