# Repository Guidelines

## Project Structure & Module Organization

OpenPostTrain Lab is a Python/PyTorch post-training laboratory. Implement algorithms in plain PyTorch before adopting production frameworks.

- `src/`: data processing, training (`sft`, `reward_model`, `dpo`, `ppo`, `grpo`), rewards, agents, evaluation, and shared utilities. Most algorithm modules are scaffolds.
- `scripts/`: environment reports, memory estimates, model exploration, and notebook conversion.
- `tests/`: utility tests; `notebooks/` and `exercises/`: learning materials.
- `configs/`: experiment YAML; `experiments/`: reproducibility records; `analysis/`: plots and ablations; `docs/`: concepts, learning notes, and experiment log.
- `data/raw/` and `data/processed/`: ignored datasets; `data/scripts/`: preparation scripts.

## Build, Test, and Development Commands

Use Python 3.10+ in an isolated environment. Run from the repository root; no separate build step is required.

```bash
python -m pip install -r requirements/base.txt
python -m pip install pytest ruff  # development tools, not in base requirements
python -m pytest                 # full test suite
python -m ruff check .            # configured lint rules
python scripts/env_report.py      # inspect hardware and environment
python scripts/memory_math.py --preset olmo2-1b --compare
```

The memory command produces estimates. Add `requirements/gpu.txt` dependencies progressively on Linux/CUDA only.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, type hints, and explanatory docstrings. Ruff targets Python 3.10 with a 100-character line length; E501 is disabled. Follow existing formatting.

Edit `notebooks/mN_*.py` sources, then regenerate notebooks with `python scripts/to_ipynb.py <source.py> <output.ipynb>`.

## Testing Guidelines

Use pytest and `tests/test_*.py` files with `test_*` functions. Add focused checks for config parsing, seeding, memory accounting, and new preprocessing/reward behavior. Prefer tiny deterministic inputs and `tmp_path`. Run a focused file with `python -m pytest tests/test_utils.py`. No coverage threshold is configured.

## Commit & Pull Request Guidelines

History uses short milestone-oriented subjects, such as `M0: repository scaffold, environment probe, memory accounting, docs`; no enforced Conventional Commits scheme exists. Prefer descriptive milestone-prefixed commits.

PRs should explain the change, relevant milestone/issues, validation commands, and measured results. Update affected documentation.

## Experiment & Agent Workflow

Read `PROJECT_STATUS.md`, `docs/learning_notes.md`, and `docs/experiment_log.md` before changes. For complex tasks, follow the planning-with-files skill and maintain `task_plan.md`, `findings.md`, and `progress.md`.

Name experiments `m<milestone>-<method>-<setting>`. Commit configs and reproducibility records described in `experiments/README.md`; exclude weights, datasets, and raw logs. Label estimates explicitly. On shared servers, select idle GPUs, set `CUDA_VISIBLE_DEVICES`, and run long jobs under `tmux`.
