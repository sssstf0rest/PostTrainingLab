# OpenPostTrain Lab

**From Pretrained Base Model to Instruct, Reasoning, and Agentic LLM**

A hands-on laboratory for modern LLM post-training. The goal is not a fine-tuned
model — it is to implement, run, debug, evaluate and compare every major stage of the
post-training pipeline, from a raw next-token predictor to an instruction-following,
reasoning, tool-using model.

Every algorithm is hand-implemented in plain PyTorch **before** the production
framework equivalent is used. Every claim is backed by a measured experiment.

## Status

**M0 complete (local half).** No model trained yet. See
[PROJECT_STATUS.md](PROJECT_STATUS.md) for the live state and
[docs/experiment_log.md](docs/experiment_log.md) for results.

Benchmark tables appear here only once they are measured. Nothing is estimated,
projected, or filled in for illustration.

## Two model tracks

- **Track A — laboratory:** `allenai/OLMo-2-0425-1B`. Fully open (data, code,
  intermediate checkpoints), small enough for many cheap controlled experiments, and
  comparable against an official open post-training pipeline.
- **Track B — portfolio:** a modern ~1.5–4B base model, chosen at M24 after the
  recipe is understood.

## Four tracks of work

1. **General post-training** — Base → SFT → Reward Model → DPO → PPO
2. **Reasoning** — Reasoning-trace SFT → GRPO / RLVR
3. **Agents** — Tool-use SFT → agent preference optimization → agentic RLVR
4. **Training systems** — memory, precision, checkpointing, DDP, DeepSpeed ZeRO, throughput

## Quick start

```bash
conda create -n optl python=3.11 -y && conda activate optl
pip install -r requirements/base.txt
python scripts/env_report.py
```

On a rented CUDA box, install torch matched to that box's CUDA first, then
`requirements/gpu.txt` (packages are uncommented per milestone, not all at once).

### Two useful M0 tools

```bash
python scripts/env_report.py                              # what hardware is this, what fits?
python scripts/memory_math.py --preset olmo2-1b --compare # where does the VRAM go?
```

## Layout

```
configs/      one YAML per experiment (an experiment is a file, not a shell command)
data/         raw / processed datasets (gitignored) + preparation scripts
src/
  data/       tokenization, chat templates, label masking, collators
  training/   sft/ reward_model/ dpo/ ppo/ grpo/  — hand-rolled, then production
  rewards/    verifiable reward functions (math, format, tool-call correctness)
  agent/      tool schemas, executable environment, trajectory handling
  evaluation/ one shared harness so every checkpoint is judged identically
  utils/      env probe, memory accounting, seeding, config, logging
experiments/  per-run: config, git commit, env.json, metrics, notes
analysis/     ablation write-ups, reward-hacking case studies, plots
docs/         learning notes, concepts, experiment log, interview notes
tests/        preprocessing, label masking, reward calculation, metrics
```

## Documentation

| File | Purpose |
|---|---|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Live state: current milestone, next step, decisions, open questions |
| [docs/training_concepts.md](docs/training_concepts.md) | Technical reference — the loop, memory, precision, CUDA |
| [docs/learning_notes.md](docs/learning_notes.md) | What was learned, per milestone |
| [docs/experiment_log.md](docs/experiment_log.md) | Append-only record of every run |
| [docs/interview_notes.md](docs/interview_notes.md) | Q&A grounded in work actually done here |
