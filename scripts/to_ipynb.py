#!/usr/bin/env python3
"""Convert a `# %%` cell-delimited .py into a .ipynb. Pure stdlib — no jupyter needed.

    python to_ipynb.py source.py out.ipynb

Cell markers:
    # %%              -> code cell
    # %% [markdown]   -> markdown cell (subsequent '# ' prefixes are stripped)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def split_cells(text: str) -> list[tuple[str, list[str]]]:
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    buf: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if buf:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
            continue
        buf.append(line)
    if buf:
        cells.append((kind, buf))
    return cells


def clean(kind: str, lines: list[str]) -> list[str]:
    if kind == "markdown":
        out = []
        for ln in lines:
            if ln.startswith("# "):
                out.append(ln[2:])
            elif ln.strip() == "#":
                out.append("")
            else:
                out.append(ln)
        lines = out
    # trim leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def to_source(lines: list[str]) -> list[str]:
    """nbformat wants a list of strings, each ending in \\n except the last."""
    if not lines:
        return []
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    cells_raw = split_cells(src.read_text(encoding="utf-8"))

    cells = []
    n_code = n_md = 0
    for kind, lines in cells_raw:
        lines = clean(kind, lines)
        if not lines:
            continue
        if kind == "markdown":
            n_md += 1
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": to_source(lines),
            })
        else:
            n_code += 1
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": to_source(lines),
            })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ptl)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.15"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dst}  ({n_md} markdown + {n_code} code = {len(cells)} cells)")


if __name__ == "__main__":
    main()
