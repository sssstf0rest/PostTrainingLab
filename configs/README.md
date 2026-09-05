# Configs

Every experiment is defined by a file here, never by flags typed into a shell.

- One config = one experiment. Committed alongside its result.
- Use `extends: <parent>.yaml` so an ablation config contains ONLY the lines that
  differ. The diff between two configs should be the diff between two experiments.
- Naming matches the experiment id: `m4-sft-1k.yaml`, `m6-sft-lr1e5.yaml`.

Loaded by `src/utils/config.py`, which backports the YAML 1.2 float resolver — so
`learning_rate: 1e-5` is a float here, unlike with plain `yaml.safe_load`.

Empty until M4; there is nothing to configure before there is something to train.
