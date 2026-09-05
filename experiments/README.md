# Experiments

One directory per run, named by experiment id (`m4-sft-20k-lr5e6`).

Each must contain enough to reproduce or explain the run a year later:

```
experiments/<id>/
  config.yaml     the exact config used
  commit.txt      git commit the run was launched from
  env.json        scripts/env_report.py --save output (which machine)
  requirements.txt  pip freeze at launch time
  metrics.json    final measured numbers
  notes.md        what happened, what surprised me, what to change next
```

Checkpoints and raw logs are gitignored; everything above is committed.

Naming convention: `m<milestone>-<method>-<key hyperparameter>`, e.g.
`m3-manual-sft`, `m6-sft-lora`, `m9-dpo-beta01`, `m13-grpo-math`.
