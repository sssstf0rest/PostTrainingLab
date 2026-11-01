# Repository audit for the Mac learning track

Inspected on **2026-10-31 (Asia/Singapore)**. Branch: **`main`**. Upstream commit:
**`5b21b9575e14116ec2be2fe9cd16840f430a02b6`**. The supplied workspace was empty, so
the requested GitHub repository was fetched into it. There was no pre-existing
uncommitted project work; the new planning/course files are session additions.
This audit describes that upstream snapshot, before the Mac course additions.

## What exists

| Category | Evidence in the checkout | What it establishes |
|---|---|---|
| Implemented infrastructure (M0) | `src/utils/config.py`: `SciFloatLoader`, `load_config`, `parse_dotted_overrides`; `seed.py`: `set_seed`; `env.py`: `collect_env`; `memory.py`: `ModelShape.n_params`, `estimate`; `logging.py`: `get_logger`; `scripts/env_report.py`, `scripts/memory_math.py` | Config loading, RNG setup, environment reporting, an explicitly approximate memory calculator, and logging exist. The calculator is not a measured peak-memory profiler. |
| Implemented model inspection (M1) | `scripts/explore_base_model.py`: `load`, `inspect_tokenizer`, `inspect_model`, `inspect_forward`, `inspect_causality`, `inspect_loss`, `inspect_embeddings`, `inspect_generation` | A real pretrained-model inspection program exists. It can download large checkpoints and is outside the required offline Mac course. |
| Existing teaching material | Paired `notebooks/m1_base_model_exploration.py` / `.ipynb`; paired `notebooks/m2_data_pipeline.py` / `.ipynb`; four `docs/` files; optional `exercises/m1_rebuild.py` | M1 teaches tokenization, embeddings, logits, softmax, causality, loss and generation. M2 teaches templates, assistant labels, padding, truncation and packing. The exercise has intentional unfinished functions; it is not the new default lesson path. |
| Notebook generation | `scripts/to_ipynb.py`: `split_cells`, `clean`, `to_source`, `main` | Editable `# %%` Python sources are an established convention. The converter hardcodes the old `ptl` kernel/Python version and omits cell IDs; Mac generation must replace metadata/add IDs rather than copy those environment claims. |
| Recorded experiments | `docs/experiment_log.md`: `m0-env`, `m0-env-gpu`, `m0-close-tied-embeddings`, `m1-base-exploration`, `m2-data-pipeline`; `experiments/m0-env/env.json` | Historical local/server probes and small inspection results are recorded. They are not measurements repeated by this audit. Most entries lack the complete per-run directory promised in `experiments/README.md`. |
| Existing checks | `tests/test_utils.py` contains seven utility tests; `tests/test_sft_masking.py` contains eighteen M2 contract tests | Tests are present. The M2 module intentionally skips when `transformers` or `src.data.sft` is unavailable; `src/data/sft.py` does not exist in this snapshot. A green suite with skips would not prove the M2 pipeline is implemented. |
| Planned functionality | `src/data`, `src/training/{sft,reward_model,dpo,ppo,grpo}`, `src/rewards`, `src/agent`, `src/evaluation` contain empty `__init__.py` files | These are package scaffolds, not implemented trainers, reward functions, tools, or evaluation harnesses. |
| Planned configs/data/results | `configs/README.md`, `experiments/README.md`, `.gitkeep` files in `data/`; `docs/dpo_vs_ppo_vs_grpo.md` explicitly says placeholder | No training YAMLs, prepared datasets, trained checkpoints, shared benchmark table, or measured DPO/PPO/GRPO comparison are committed. `m3-manual-sft-loop` is a TODO log entry. |

Persisted upstream notebook outputs contain no saved error output, but they are not
a fresh-kernel verification: M1 has 44 cells (17 code cells with execution counts
out of 18 code cells), M2 has 50 cells (18 out of 19). The old experiment log's
42-cell M1 description is older than the checked-in notebook. The current session's
new execution evidence belongs in [VERIFICATION.md](VERIFICATION.md).

## Recorded results and their limits

- **M0:** The log records an Apple M3/16 GB environment probe, then a separate shared
  Linux CUDA probe. `experiments/m0-env/env.json` is the latter, despite the directory
  name. It records six visible GPUs named `NVIDIA L20X`, 139.8 GiB and capability
  `9.0`; the notes call them H200s based on an owner report and other properties.
  This checkout cannot independently establish the physical GPU identity. Treat
  that as historical evidence with a recorded conflict, not a universal rule to
  trust one tool over another.
- **M0 architecture:** The log records configuration/checkpoint inspection supporting
  untied embeddings and approximately 1.485 billion parameters for the selected
  OLMo checkpoint. The memory totals around 27.11/25.12 GiB are **estimator outputs**,
  not measured training peaks. A reported small matrix-multiplication benchmark is
  not end-to-end training throughput.
- **M1:** The log records a particular greedy Base versus Instruct comparison:
  120 generated tokens hitting a cap versus nine tokens ending with EOS, tokenizer
  facts, logits shapes, a causality check, and a manually matched loss of 2.8438
  nats/token on one sentence. These demonstrate inspection mechanics. They do not
  establish population-level quality, that all base models fail to answer, or that
  one EOS label explains the entire post-training difference.
- **M2:** The log records 28 tokens/9 supervised tokens for one templated example,
  role-marker tokenization for that tokenizer, mean-loss masking arithmetic, and
  padding comparisons. The notebook has local functions, including
  `assistant_char_spans`, `build_labels`, `causal_loss`, and `collate`; the reusable
  `src.data.sft` module is still an exercise. No trained SFT model follows from these
  examples.
- **Training outcomes:** There is no committed evidence of project-trained SFT,
  LoRA, reward-model, DPO, PPO, GRPO, or agent quality gains. Cached official
  post-trained checkpoints mentioned in server notes are pretrained artifacts,
  not models trained by this project or guaranteed to exist on this Mac.

## Discrepancies to repair in the teaching track

Historical records are preserved. The new lessons use the following corrections.

| Historical statement or shortcut | Teaching correction |
|---|---|
| README says “M0 complete (local half)”; status header says “M1 next” | Later code/log entries support M0 infrastructure, M1 inspection and M2 materials; M3 training is not implemented. Use evidence per artifact rather than the header. |
| README draws `SFT → Reward Model → DPO → PPO` as one track | DPO and reward-model-plus-PPO are alternative routes. GRPO is an optimizer; RLVR describes verifiable reward sources. Tool training can reuse several routes. |
| “All methods differ only in text and loss; byte-for-byte same machinery” | Autograd and optimizers are shared primitives. Online methods also require sampling, rollout storage, old-policy snapshots, reward computation and sometimes value estimation; reward heads and trainable/frozen components differ. |
| MPS is “useless for real training” / blanket backend exclusions | That is an outdated project assumption, not an impossibility result. Test exact operations and versions. The Mac course requires CPU and validates its tiny MPS path separately. Quantized extensions need method/backend/OS-specific verification before inclusion. |
| “16 bytes per parameter” as a universal mixed-precision rule | It is one assumed layout. FP32 weights/gradients and two FP32 Adam moments also total 16 bytes per trained scalar without a separate master copy. Actual dtypes, trainability, optimizer, autocast strategy, activations and allocator state determine memory. `GB = 1024**3` in the old code means GiB. |
| Estimator outputs later called “measured” in learning notes | Keep arithmetic estimates separate from measurements. Small shape observations do not measure the peak of a long training run. Checkpointing savings/slowdowns depend on implementation and workload. |
| BF16 “never underflows” / pure BF16 “always stalls” | Every finite dtype has limits. BF16 improves exponent range relative to FP16 but has coarse precision; stability and update loss depend on magnitudes and optimizer implementation. AMP commonly keeps FP32 parameters; a distinct FP32 master copy is not mandatory in every recipe. |
| RNG seed must not be fixed for RL | Seeding a pseudorandom stream allows diverse successive samples and repeatable experiments. Re-seeding before every sample can accidentally repeat samples. Exact reproducibility across devices/releases is a separate question. Setting `PYTHONHASHSEED` after interpreter launch does not retroactively set hash randomization. |
| Post-training only installs conventions / never adds knowledge | The selected experiment compares behavior on a small prompt set. Fine-tuning can also learn domain information and can forget prior behavior; evaluate both task quality and retention. |
| A tokenizer is always lossless and unchangeable | Exact round trips depend on normalization, special-token handling and decoding options. Vocabulary/template changes are possible but require careful alignment and training; they do not automatically invalidate every existing embedding. |
| No saved chat template means a base model has no notion of roles | Template metadata and learned behavior are different. Missing metadata does not establish the absence of role-related knowledge from pretraining. |
| Spare embedding rows are unused/random and adding tokens never resizes | Matrix padding alone does not prove row history or API resize behavior. Added-token IDs, `len(tokenizer)`, model matrix size and output semantics must be checked together. |
| Assistant-only masking is the sole valid SFT recipe; EOS masking guarantees endless output | Assistant-only loss is the course's explicit choice, not a universal definition of SFT. Stopping behavior depends on existing weights, data, token conventions and decoding configuration as well as EOS supervision. |
| String search plus tokenizer offsets is a robust role parser | `assistant_char_spans` searches literal marker/EOS text and can mistake content for structure. Offsets map spans to tokens; they do not establish trustworthy message boundaries. Lesson 02 must derive spans from structured messages/template generation and test marker-like content and special-token offsets. |
| Ignored tokens can be detected as per-token loss `!= 0` | Use the supervision mask itself. A valid token can have a zero loss numerically. Reject zero-supervision batches before a mean reduction. |

The MPS and reproducibility corrections are consistent with official PyTorch
[MPS guidance](https://docs.pytorch.org/docs/2.15/notes/mps.html),
[AMP examples](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html), and
[reproducibility notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html).
These are reference documents; the installed teaching environment and the exact
operations tested are recorded separately. DPO's separate route follows its
[original paper](https://arxiv.org/abs/2305.18290).

## Interview gaps addressed by the roadmap

The existing materials give a useful inspection/masking foundation. The additional
course must supply actual small training loops and controlled experiments; LoRA
and quantization mechanics; reward/preference/RL derivations; entity/time-aware
data design; inference and uncertainty; model comparison and retention; local
tools/verifiers; reproducible systems accounting; and a structured-plus-text
consumer-intelligence capstone. No Mobilewalla-specific proprietary data or
workflow is assumed. See [COURSE.md](COURSE.md) for the file-by-file roadmap and
[PROGRESS.md](PROGRESS.md) for learner evidence, kept distinct from execution.
