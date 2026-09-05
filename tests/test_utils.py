"""Tests for M0 infrastructure.

We test the boring plumbing because a silent bug here (a mis-parsed learning rate,
an unseeded RNG) corrupts every downstream experiment without ever raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, parse_dotted_overrides
from src.utils.memory import PRESETS, estimate
from src.utils.seed import set_seed


def test_yaml_parses_scientific_notation_as_float(tmp_path: Path) -> None:
    """The YAML 1.1 trap: `1e-5` must not load as the string '1e-5'."""
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  learning_rate: 1e-5\n  weight_decay: 0.01\n")
    cfg = load_config(p)
    assert isinstance(cfg["train"]["learning_rate"], float)
    assert cfg["train"]["learning_rate"] == 1e-5


def test_config_extends_deep_merges(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("train:\n  lr: 1e-5\n  epochs: 3\nmodel:\n  name: base\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\ntrain:\n  lr: 5e-6\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["train"]["lr"] == 5e-6      # overridden
    assert cfg["train"]["epochs"] == 3     # inherited, not clobbered
    assert cfg["model"]["name"] == "base"


def test_dotted_overrides_preserve_types() -> None:
    o = parse_dotted_overrides(["train.lr=1e-5", "train.bf16=true", "data.n=1000"])
    assert isinstance(o["train"]["lr"], float)
    assert o["train"]["bf16"] is True
    assert o["data"]["n"] == 1000


def test_seed_makes_torch_reproducible() -> None:
    import torch

    set_seed(0)
    a = torch.randn(8)
    set_seed(0)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_param_count_matches_published_size() -> None:
    """OLMo 2 1B is published at ~1.48B params. Our architecture arithmetic must agree,
    otherwise every memory estimate built on it is wrong."""
    n = PRESETS["olmo2-1b"].n_params()
    assert 1.45e9 < n < 1.52e9


def test_gradient_checkpointing_reduces_activation_memory() -> None:
    shape = PRESETS["olmo2-1b"]
    off = estimate(shape, batch_size=8, seq_len=2048, gradient_checkpointing=False)
    on = estimate(shape, batch_size=8, seq_len=2048, gradient_checkpointing=True)
    assert on.activations < off.activations
    assert on.params == off.params  # checkpointing does not touch weights


def test_lora_shrinks_optimizer_not_parameters() -> None:
    shape = PRESETS["olmo2-1b"]
    full = estimate(shape, trainable_fraction=1.0)
    lora = estimate(shape, trainable_fraction=0.005)
    assert lora.optimizer < full.optimizer / 100
    assert lora.params == full.params  # the frozen base still occupies VRAM
