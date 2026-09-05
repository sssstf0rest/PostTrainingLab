"""Config loading.

WHY THIS FILE EXISTS
--------------------
Rule: an experiment is defined by a config FILE, never by flags typed into a shell.
Reasons:
  - The config is committed alongside the result, so the result is reproducible.
  - Diffing two configs shows exactly what an ablation changed (and nothing else).
  - Six months later, `experiments/m4-sft-1k/config.yaml` still explains the run.

Kept deliberately tiny. We are not building a config framework; we are building a
record-keeping habit. If this ever needs to grow, we adopt Hydra — not before.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml


class SciFloatLoader(yaml.SafeLoader):
    """SafeLoader that parses `1e-5` as a float instead of the string "1e-5".

    PyYAML implements YAML 1.1, whose float resolver demands a decimal point AND a
    signed exponent -- so plain `learning_rate: 1e-5` silently loads as a str. It
    then either crashes deep inside the optimizer or, worse, gets coerced somewhere
    and trains with a wrong LR. YAML 1.2 fixed this; we backport the 1.2 resolver.
    """


SciFloatLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                   |[-+]?\.[0-9][0-9_]*(?:[eE][-+]?[0-9]+)?
                   |[-+]?[0-9][0-9_]*(?:[eE][-+]?[0-9]+)
                   |[-+]?\.(?:inf|Inf|INF)
                   |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."),
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving a single optional `extends:` parent.

    `extends` lets configs/sft_base.yaml hold the shared settings while
    configs/m6_lr1e5.yaml contains only `train: {learning_rate: 1.0e-5}` — so the
    diff between two ablations is literally the one line that differs.
    """
    path = Path(path)
    with path.open() as f:
        cfg = yaml.load(f, Loader=SciFloatLoader) or {}

    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)

    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def parse_dotted_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ["train.learning_rate=1e-5", "data.n=1000"] into a nested dict.

    For quick sweeps only. Anything you intend to KEEP belongs in a committed file.
    """
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.load(raw, Loader=SciFloatLoader)
    return out
