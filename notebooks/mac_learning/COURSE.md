# PostTrainingLab: Mac learning course

For a MacBook Air M3 with 16 GB unified memory. This is a **progressive course**:
lessons 00 and 01 are delivered in the first session; 02–13 are planned, not
completed project milestones. Every lesson will have a runnable notebook plus an
editable `# %%` Python source. [README.md](README.md) covers setup,
[AUDIT.md](AUDIT.md) establishes the upstream evidence, [VERIFICATION.md](VERIFICATION.md)
records execution, and [PROGRESS.md](PROGRESS.md) tracks understanding separately.

## How to study

Use **intuition → mathematics → worked example → implementation → controlled
experiment → interpretation → interview explanation**. At each “predict” prompt,
write a sentence before running the answer. Modify one thing, explain the result,
then attempt the final three exercises before reading the separated solutions.
Passing assertions proves properties of the code; it does not establish mastery.

**P0** means highest interview priority, **P1** means important supporting knowledge,
and **P2** means advanced interview depth. All 14 tiny local lessons belong to the
course. Optional pretrained-model/backend extensions are separately labeled and
disabled by default. We introduce only the mathematical prerequisites used in
each lesson, with symbols, reductions, shapes, masks and detached values explicit.

Recommended first-pass interview order: **00 → 01 → 02 → 03 → 04 → 05 → 06 → 11 →
12 → 13**, then deepen the online/reasoning/agent branches through **07 → 08 → 09 →
10** and revisit the capstone defense. Numerical order also works. Begin designing
evaluation/splits in lesson 02; lesson 12 develops the inference, not a license to
postpone evaluation until the end.

## Index and repository mapping

All filenames below are in `notebooks/mac_learning/`; each `.ipynb` has the same
basename `.py`. “Scaffold” means the upstream path has no implementation. Milestone
numbers are taken from actual files, not inferred from gaps in numbering.

| Lesson / notebook | Priority | Prerequisites | Upstream files/functions and milestone | Delivery |
|---|---|---|---|---|
| `00_local_setup_and_post_training_map.ipynb` | P0 | Python basics | M0: `src/utils/env.py::collect_env`, `src/utils/memory.py::estimate`, `scripts/env_report.py`; routes in `README.md` | First session |
| `01_tokens_probabilities_losses_and_gradients.ipynb` | P0 | 00; vectors/logs introduced here | M1: `notebooks/m1_base_model_exploration.py`, `scripts/explore_base_model.py::{inspect_forward,inspect_loss,inspect_causality}`; M3 loop planned | First session |
| `02_data_supervision_and_leakage.ipynb` | P0 | 01 | M2: `notebooks/m2_data_pipeline.py::{assistant_char_spans,build_labels,causal_loss,collate}`, `tests/test_sft_masking.py`; missing `src/data/sft.py` | Planned |
| `03_supervised_fine_tuning.ipynb` | P0 | 01–02 | M3/M4 planned: `src/training/sft/__init__.py` scaffold; `docs/experiment_log.md#m3-manual-sft-loop`; M4 dependencies in `requirements/gpu.txt` | Planned |
| `04_full_finetuning_lora_and_qlora.ipynb` | P0 | 03; matrix multiplication | M6 planned: `requirements/gpu.txt` PEFT/bitsandbytes comments; `src/utils/memory.py::estimate(trainable_fraction=...)`, `tests/test_utils.py::test_lora_shrinks_optimizer_not_parameters` | Planned |
| `05_reward_modeling.ipynb` | P1 | 01–03; sigmoid introduced here | Reward-model stage planned: `src/training/reward_model/__init__.py`, `src/rewards/__init__.py` scaffolds; no exact reward-model milestone number established | Planned |
| `06_direct_preference_optimization.ipynb` | P0 | 04–05; conditional probability | M9 planned: `src/training/dpo/__init__.py` scaffold; `experiments/README.md` DPO naming; `docs/dpo_vs_ppo_vs_grpo.md` placeholder for M19 | Planned |
| `07_ppo_based_rlhf.ipynb` | P2 | 03, 05–06; expectation/variance introduced here | M10 planned: `src/training/ppo/__init__.py` scaffold; comparison plan in `docs/dpo_vs_ppo_vs_grpo.md` | Planned |
| `08_reasoning_sft_and_verifiable_rewards.ipynb` | P1 | 02–03; evaluation basics | Reasoning route in `README.md`; `src/rewards/__init__.py` and `src/training/sft/__init__.py` scaffolds; `math-verify` comment in `requirements/gpu.txt`; reasoning-SFT milestone unnumbered | Planned |
| `09_grpo_and_rlvr.ipynb` | P2 | 07–08; group means/standard deviations | M13 planned: `src/training/grpo/__init__.py`, `src/rewards/__init__.py` scaffolds; M19 comparison placeholder | Planned |
| `10_tool_use_and_agent_post_training.ipynb` | P2 | 02–03, 05–06, 08–09 | Agent route in `README.md`; `src/agent/__init__.py`, `src/rewards/__init__.py` scaffolds; no exact agent milestone number established | Planned |
| `11_training_efficiency_and_reproducibility.ipynb` | P1 | 03–04; revisit after 07/09 | M0 utilities: `src/utils/{memory,seed,config,env}.py`; M6 memory ablations, M20 DDP/ZeRO, M21 checkpoint/resume, M25 serving are planned in docs/requirements | Planned |
| `12_statistical_evaluation_and_failure_diagnosis.ipynb` | P0 | 02–06; probability/inference introduced here | M5 eval dependencies in `requirements/gpu.txt`; `src/evaluation/__init__.py` scaffold; M6 ablation and M19 comparison plans; `experiments/README.md` evidence format | Planned |
| `13_consumer_intelligence_capstone_and_interview.ipynb` | P0 | 02–06, 11–12; optional branch evidence from 07–10 | New educational case, mapped to `src/data`, `src/evaluation`, `configs/README.md`, `experiments/README.md`; informs deferred M24 portfolio selection; not an upstream completed milestone | Planned |

## Lesson contracts and experiments

### 00 — Local setup and the post-training landscape

Identify the actual OS, Python/kernel, RAM, PyTorch and CPU/MPS availability. A tiny
local optimization makes the autograd/optimizer boundary visible and checks finite
gradients. Count tensor storage under stated assumptions; distinguish it from
process memory and a whole-device peak. Default to CPU/float32, bounded work and
`QUICK_MODE`; numerical double precision stays on CPU. The engineering upper target
of roughly 6–8 GB is a ceiling with headroom, not a fit guarantee or an intended
allocation.

Map branching routes: raw data → pretraining/continued pretraining; demonstrations
→ instruction/reasoning/tool SFT; preference pairs → DPO **or** reward modeling plus
PPO; sampled outputs + verifiers → RLVR, which may use PPO or GRPO. Retrieval supplies
fresh external context at inference. Clarify inputs, outputs, trainable/frozen
components and when each route is appropriate. Vary sequence length with batch,
heads, vocabulary and dtype fixed; compare actual attention/logit tensor payloads
and state why these do not measure training peaks or runtime. Exit check: choose a
route for fresh facts, behavior change, and ranked answers without treating the
methods as one mandatory chain.

### 01 — Tokens, probabilities, losses and gradients

Build a local vocabulary and tiny causal model from scratch. Inspect integer IDs
`[B,T]`, embedding lookups `[B,T,D]`, causal attention scores `[B,H,T,T]`, logits
`[B,T,V]`, stable log-softmax, autoregressive sequence probability, teacher forcing,
shifted cross-entropy and perplexity. Work a three-token probability/gradient
example by hand, then match it to autograd and a CPU float64 finite-difference
check. Show chain-rule effects through actual parameters.

Make gradient descent, AdamW moments/bias correction, learning rate, decoupled
weight decay, gradient reset and clipping concrete. Trace one full update and a
short bounded training experiment with a clearly stated baseline, held-constant
initialization/data/steps and a learning-rate comparison. Use bounded generation
under inference mode, distinguish teacher-forced loss from free generation, and
demonstrate `.eval()` versus disabling gradient tracking. Test causal isolation,
finite losses and manual/framework loss agreement. Exit check: explain which
logit predicts a particular target and how its gradient changes the parameters.

### 02 — Data preparation, supervision and leakage

Adapt M2's useful tables while replacing marker-string parsing with boundaries
derived from structured messages/template construction. Demonstrate chat roles,
assistant labels, attention masks versus loss masks, one causal shift, EOS,
padding, truncation, collation and packing. Test double-shifting, pad/EOS ID
collisions, genuine EOS retention, zero-supervision batches, role-marker text
inside content, cross-example packed attention, and token-mean versus
sequence-mean loss weighting for unequal response lengths.

Create synthetic entities, related records, timestamps and near duplicates.
Specify whether deployment predicts future records for known entities or unseen
entities before splitting. Compare a leaky row split to the appropriate group/time
split with a simple baseline. Hold modeling choices constant; attribute apparent
quality to leakage only within this constructed example. Add representative
sampling, deduplication, noisy labels and point-in-time features. Exit check:
construct a mask and split that answers a stated deployment question, and explain
the changed denominator after masking.

### 03 — Supervised fine-tuning

Derive the masked empirical next-token objective and implement forward → loss →
backward → clipping → optimizer step → reset. Inputs are locally generated
demonstrations; outputs are a tiny updated model and a self-contained checkpoint.
All tiny model parameters train initially. Validate without retaining graphs and
save optimizer/RNG/config state needed for a reproducible resume.

Compare an untrained or clearly defined pretrained-on-toy-corpus baseline with
SFT. Deliberately overfit a small demonstration set, investigate one factor such
as data size or learning rate, select early stopping on validation, and retain a
second toy task for forgetting checks. Demonstrate token-weighted gradient
accumulation against a true larger batch, then explain dropout/numerics and
variable-supervised-token differences. Contrast continued pretraining on raw
domain text with instruction SFT and with RAG for fresh facts. Exit check: debug
a decreasing training loss with worsening validation and retention.

### 04 — Full fine-tuning, LoRA and QLoRA

Implement `W = W0 + (alpha/r) B A`, with `W0:[d_out,d_in]`,
`A:[r,d_in]`, `B:[d_out,r]`, and `r(d_in+d_out)` adapter parameters.
Freeze `W0`; derive why zero `B` preserves the initial function and initially
blocks the gradient to `A` while allowing a gradient to `B`. Compare matched
small full-tuning and adapter runs; vary rank/scaling in a controlled experiment.
Count parameters, gradients and optimizer state separately from unchanged base
weights/activations; test frozen-parameter invariants and adapter merge behavior.

Use a tiny quantize/dequantize illustration to show frozen quantized weights plus
trainable floating-point adapters. Clearly separate that illustration from
QLoRA's NF4, double quantization and paged optimizers. Any optional real extension
must first record model/download size, license, exact framework/backend/OS and
verified training support; it remains disabled. No CUDA-only QLoRA assumptions or
blanket Apple-Silicon impossibility claim. Exit check: derive memory savings and
explain which allocations LoRA does not remove.

### 05 — Reward modeling

Derive Bradley–Terry preference likelihood
`P(chosen > rejected | x) = sigmoid(r(x,chosen)-r(x,rejected))` and its stable
negative log-likelihood. A tiny scorer trains on synthetic pairs; policy models
need not train in this lesson. Explain score identifiability, preference
probabilities, disagreement and why a score is not calibrated cardinal utility.

Hold the pair generator/splits/model fixed while varying label noise or a
spurious length cue. Evaluate on held-out prompts and a counterexample segment.
Intentionally build a length-biased reward and optimize/search against it to
show undesirable behavior that receives a high predicted reward. Distinguish
preference accuracy from calibration and task utility. Exit check: explain why
adding a constant to all rewards leaves pair probabilities unchanged, and why
reward-model overoptimization can harm real quality.

### 06 — Direct Preference Optimization

Derive the optimal policy for expected reward minus `beta * KL(policy || reference)`
under stated support assumptions, eliminate the prompt-only partition function in
reward differences, and substitute into Bradley–Terry. Implement response-token
sequence log-probabilities and the stable loss
`-logsigmoid(beta * ((logpi_ch-logref_ch)-(logpi_rj-logref_rj)))`.
Train only the tiny policy; freeze/detach reference values. Start with a numerical
pair, prove equal policy/reference gives `log(2)`, and check gradient directions.

Vary beta while holding data/reference/initialization/steps fixed; report margins,
absolute chosen/rejected log-probabilities, held-out preferences and drift.
Distinguish beta's theoretical KL role from its local gradient role; do not claim a
universal monotonic update-size relation. Compare sequence sums with length
normalization as different objectives, discuss length and reference-policy
choices, poor preferences and distribution mismatch. Exit check: explain why no
separate reward model is needed and why a better margin does not guarantee an
absolute probability increase for every chosen answer.

### 07 — PPO-based RLHF

Begin with an exact tiny policy-gradient expectation, then sample actual bounded
trajectories from a small policy. Implement returns, value estimates, detached
advantages, GAE, terminal masks, importance ratios, clipped policy loss, value
regression and KL regularization. Name the **current policy**, **old rollout
policy**, and **reference policy** separately: the ratio uses old-policy
log-probabilities; reference KL regularizes a different comparison. A tiny reward
scorer can be frozen; the value function trains alongside the policy.

State the simplification (small discrete environment/short trajectories rather
than several LLM copies). Map states/actions to prompts/tokens. Compare one
clipping or KL setting with matched seeds/budgets, track actual sampled success
and drift, demonstrate reward hacking, and show why clipping is not a hard KL
bound. Exit check: identify stale/undetached quantities in a faulty update and
explain the memory roles of all components.

### 08 — Reasoning SFT and verifiable rewards

Generate small arithmetic/symbolic tasks with independent answer checking and
train/test separation by problem structure or operands, not duplicate wording.
Compare answer-only and demonstration-trace SFT under explicitly matched budgets,
with a rule/exact solver and untrained model as appropriate baselines. Define
outcome versus process supervision and which output tokens receive loss.

Implement strict parsing and bounded exact evaluation. Build a flawed verifier
(for example, rewarding an answer merely appearing anywhere) and an exploit;
repair it and check the independent ground truth. Evaluate final-answer accuracy
and explanation consistency separately. A correct final answer does not show
that the visible explanation faithfully reports internal reasoning. Exit check:
distinguish reward validity, benchmark accuracy and claims about reasoning.

### 09 — GRPO and RLVR

State the precise implemented GRPO variant: group-relative reward normalization,
chosen variance convention, clipping, reference-KL estimator/placement, and
token/sequence reductions. Sample a small group of completions per prompt from
the rollout policy, compute detached relative advantages and perform bounded
updates of the current tiny policy. Distinguish the optimizer GRPO from the
verifiable reward setup RLVR; omit a learned value critic explicitly.

Start with four rewards and manually compute centered/scaled advantages. Test
all-equal groups: epsilon makes division finite but supplies no useful reward
signal. Compare one normalization/group-size factor at a fixed sampling budget,
tracking reward variance, success, clipping fraction and KL. Compare DPO/PPO/GRPO
by required data, policy/reference/reward/value components, sampling, memory,
stability and task fit; report only the toy measurements actually run. Exit check:
explain lost signal in equal-reward groups and sensitivity to reward scaling.

### 10 — Tool-use and agent post-training

Build a tiny calculator/lookup environment with an allowlist, strict JSON/schema
validation, bounded calls and no arbitrary `eval`, network or shell execution.
Generate locally supervised tool trajectories, preference pairs and verifiable
whole-task outcomes. Start from tool-call SFT; illustrate how preference or RLVR
objectives could optimize the same environment without claiming a missing
production agent is implemented upstream.

Measure syntax validity, tool-selection/argument correctness and final task
success separately. Deliberately produce valid-but-wrong calls and unnecessary
calls. Compare intermediate format rewards with final task rewards under a fixed
task set; explain credit assignment, tool-result visibility, intermediate versus
terminal rewards, trajectory-level evaluation and environment leakage. Exit check:
defend why high valid-JSON rates can coexist with poor business task success.

### 11 — Training efficiency and reproducibility

Account separately for weight storage, trainable gradients, actual optimizer
state dtypes, optional master weights, saved activations, logits, rollout/KV
buffers and extra reference/reward/value components. Explain assumptions and
units; instrument what is measurable on CPU/MPS and label allocator/process/peak
limitations. Measure bounded runtime with synchronization/warmup when required.

Demonstrate precision tradeoffs, token-weighted accumulation, checkpointing,
configuration and exact resume with RNG state. Use small calculations/simulations
for DDP replicated data parallelism and ZeRO optimizer/gradient/parameter sharding,
including communication and temporary allocations; never require those server
packages locally. Compare one efficiency knob with matching work/quality and
report the tradeoff. Distinguish deterministic pseudorandom sequences from
repeated identical samples and cross-device bitwise reproducibility. Exit check:
estimate a workload using stated assumptions and explain what still requires
measurement.

### 12 — Statistical evaluation and failure diagnosis

Predefine a primary metric and practical effect threshold. Keep validation for
tuning and the held-out test for final assessment; compare base models, rules,
logistic regression/trees, prompting/retrieval and post-trained models when
appropriate. Use paired comparisons, effect sizes, suitable tests and confidence
intervals. Resample independent **entities/prompts**, not correlated rows/tokens;
distinguish training-seed variation from evaluation-sampling uncertainty.

Run small learning curves/ablations and segment error tables. Address imbalance,
precision/recall, calibration, distribution shift, hallucinations and abstention,
second-task retention, basic memorization/privacy probes and data provenance.
Create a controlled example in which naive token/row resampling yields
overconfident uncertainty. Explain statistical significance versus useful impact,
multiple tuning choices and small-sample limitations. Exit check: defend the unit
of analysis and distinguish “no evidence of improvement” from “proved equal.”

### 13 — Consumer-intelligence capstone and interview defense

Use **synthetic educational data, not Mobilewalla data or an asserted Mobilewalla
workflow**: consumer/entity records, time-stamped observations and short text.
Define deployment, point-in-time availability, target, independent units, primary
metric, latency budget and an illustrative cost model before modeling. Include a
local SQLite exercise joining only features observed before each prediction.

Compare rules/statistical and logistic-regression/tree baselines with structured
and text features. Evaluate local prompting/retrieval/tiny SFT/adapters only where
the task supports them; add preference optimization only for an explicitly
defined ranking/preference subproblem. Avoid implying tiny random-model prompting
matches a real pretrained LLM. Compare freshness, labels, quality, latency and
measured/estimated cost separately; propose monitoring, shift triggers and
retraining. Distinguish propensity from causal uplift: incremental impact needs
randomized intervention evidence or justified causal identification assumptions.
Finish a defensible proposal and interview narrative: problem, data, method,
evaluation, actual findings, limitations and next steps. Exit check: answer which
claims are implemented/measured/studied/proposed and defend the simplest useful
solution.

## References and delivery standard

Each delivered notebook contains its own short primary-source list and
version-specific API references. Later lessons will use the original
[LoRA](https://arxiv.org/abs/2106.09685),
[QLoRA](https://arxiv.org/abs/2305.14314),
[DPO](https://arxiv.org/abs/2305.18290),
[PPO](https://arxiv.org/abs/1707.06347), and
[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300) papers; those links are a
reading roadmap, not claims that their experiments have been reproduced here.

Every default path uses synthetic data and tiny local models, needs no checkpoint
or dataset download after environment installation, has CPU support, finite-loss
and relevant invariant checks, bounded steps/generation and fresh-kernel
execution where available. Optional extensions must independently establish
license, download size, resource fit and backend support. Exercises have complete
separated solutions and never break Run All. Runtime and resource values come
from execution or are explicitly marked estimates/illustrations. Actual evidence
and learner explanations, including unresolved misconceptions, are recorded in
the progress files before advancing.
