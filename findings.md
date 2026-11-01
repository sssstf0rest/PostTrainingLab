# Findings

## Initial context
- User request was read from the attached text in full.
- Workspace initially contained no files or Git checkout.
- Host reports Darwin arm64, MacBook Air; default python3 is 3.15.0. Hardware and compatible Python/PyTorch versions still need inspection.
- User requires progressive course construction: only notebooks 00 and 01 in this session; roadmap covers all 14 lessons.
- Existing records must remain historical; no claims of mastery from notebook execution.

## Audit and references
Pending repository acquisition.

## Repository acquired
- Fetched requested GitHub repository into the initially empty workspace; branch `main`, commit `5b21b9575e14116ec2be2fe9cd16840f430a02b6`.
- No preexisting user modifications were present. Only session planning files existed before checkout.
- Real implementation: environment/config/seed/memory/logging utilities and base-model inspection script. Most training/data/reward/agent/evaluation packages are empty scaffolds.
- Existing M1/M2 percent-format notebooks and converter can be reused as teaching references; M2 production implementation `src/data/sft.py` is missing.
- Audit found stale README/status statements and overgeneralized memory, BF16, MPS, EOS, and knowledge-acquisition explanations. Preserve historical records and qualify them in the new track.
- Actual target hardware confirmed: Apple M3 MacBook Air, 16 GiB unified memory; arm64 Darwin. Avoid storing machine identifiers.
- `uv` and Homebrew Python 3.12 are available; use a separate `.venv` with only teaching dependencies, not server requirements.

## Source verification and teaching design
- Primary DPO, PPO, DeepSeekMath/GRPO, LoRA, QLoRA, InstructGPT, RAG, and domain-adaptive pretraining sources opened. Official PyTorch stable MPS documentation redirects to version 2.15.
- Installed teaching environment resolves Python 3.12.15, torch 2.15.0, NumPy 2.5.4, Matplotlib 3.11.2, JupyterLab 4.6.5. Full transitive snapshot stored separately from historical server requirements.
- Lesson 00 authored: branching route map, evidence selection, explicit two-logit gradient example, storage-scaling controlled experiment, local backend update check, interview practice.
- Lesson 01 delegated: local character tokenizer, explicit causal LM, gradients and optimizer mechanics, paired-seed learning-rate experiment with train-fitted bigram baseline.

## Review and first execution
- MPS is built/available in torch2.15.0 on this actual macOS27.0.1 M3 Air; a direct float32 forward/backward probe succeeded.
- Lesson00 fresh-kernel CPU execution passed (6 code cells). Source-hash guard caught a concurrent final author edit before lesson01 execution; rebuilt final source after handoff.
- Independent review found no lesson01 math/code blocker; clarified lesson00 token-vs-sequence normalization and switched diagram arrows to box-boundary clipping.
- Visually inspected the executed branching-map PNG; nodes and arrows render clearly after the fix.
- Lesson01 uses 5,592 parameters, a train-fitted bigram baseline plus a grammar-aware rule baseline, and a paired-seed learning-rate comparison. Lower average NLL alone does not imply cue-task success; no production claim is made.
- Added a finite dataset-generation attempt budget and split-capacity validation so optional filler-length edits cannot hang.

## Final execution evidence
- Both required notebooks executed completely in fresh kernels on CPU and MPS on the actual M3/16GiB Mac (Python3.12.15, torch2.15.0, macOS27.0.1). Numerical float64 diagnostics intentionally remain on CPU.
- Notebook00 has 24 cells/6 code cells; notebook01 has 45 cells/20 code cells. All assertions passed, all code cells executed, no error outputs.
- Visually inspected lesson01 learning curves: high LR lowers NLL but cue-recall plateaus near chance, while the grammar rule gives perfect recall. Curves clearly display both metrics and baselines.
- CPU01 validation NLL: lr.0001 mean2.0650 vs lr.01 mean1.1136 across two paired seeds. Cue accuracy for high LR .500/.469, rule1.0. Bounded generation copies the wrong cue. These are toy results, not business/production gains.
- Existing utility tests: 7 passed; M2 test module skipped because transformers is absent. Independently confirmed src/data/sft.py is absent. Do not interpret skip as M2 success.
- Final small wording correction to notebook00 trainable components triggered rebuild/re-execution of that notebook only; no unrelated reruns planned.
- Final source/notebook/evidence consistency checks and dependency check passed; original tracked README/status/docs/src/tests/scripts/requirements remain unchanged.
