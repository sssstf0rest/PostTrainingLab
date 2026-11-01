# PostTrainingLab on your Mac

Start with [00 — Local setup and the map](00_local_setup_and_post_training_map.ipynb),
then [01 — Tokens, probabilities, losses and gradients](01_tokens_probabilities_losses_and_gradients.ipynb).
The notebook is the study artifact; edit its matching `# %%` Python source and
regenerate it. The complete [course index](COURSE.md), [repository audit](AUDIT.md),
[execution evidence](VERIFICATION.md), and [learner record](PROGRESS.md) are separate
so running code never becomes a claim of understanding.

## Install once

Run these commands **from the repository root**. This separate environment avoids
`requirements/base.txt` and the server/GPU stack. Python 3.12 is the tested minor
version; the exact interpreter used was 3.12.15. Native arm64 Python is recommended
on Apple Silicon. `uv` is used below; it can obtain Python 3.12 if it is absent.

```bash
uv venv --python 3.12.15 .venv
uv pip install --python .venv/bin/python -r notebooks/mac_learning/requirements.lock.txt
.venv/bin/python -m ipykernel install --prefix .venv --name ptl-mac --display-name "PostTrainingLab Mac"
.venv/bin/jupyter lab notebooks/mac_learning
```

Select **PostTrainingLab Mac**. The notebook's first code cell prints the interpreter,
PyTorch version, device and quick-mode status; verify you selected the intended
kernel. The kernel registration stays inside `.venv`, rather than changing your
user-wide Jupyter kernels. The environment was already installed in this workspace
for the recorded runs. Re-running `uv venv` recreates it; skip the installation
commands when continuing with this existing environment.

Without `uv`, use an installed `python3.12 -m venv .venv`, then
`.venv/bin/python -m pip install -r notebooks/mac_learning/requirements.lock.txt`.
Use the same kernel/Jupyter commands afterwards. No editable install of the full
repository is necessary. Installation requires internet; required lessons use only
locally generated data and models afterwards. No pretrained weights, paid services,
Hugging Face downloads, cloud GPUs, or MLX dependencies are required.

`requirements.txt` lists direct dependencies with tested exact versions;
`requirements.lock.txt` snapshots their full resolved environment on macOS arm64.
The lock contains version pins, not wheel hashes or an all-platform solver result.
For another platform, use the direct requirements and record its resolved packages;
MPS is Apple-specific and CPU is the required portable path. Do not treat newer
package releases as verified automatically.

## Workload and devices

Both notebooks default to `QUICK_MODE=True`, CPU and float32. They are independent:
no saved checkpoint from lesson 00 is needed by lesson 01. Learning exercises are
complete; there are no unfinished functions in the Run All path. Default work uses
short sequences, small batches, tiny random models and bounded update/generation
counts. Quick mode controls demonstration length, not which core concepts exist.

The engineering upper target is approximately **6–8 GiB** for the lesson workload,
leaving substantial room on a 16 GiB unified-memory Mac. This is not a guarantee of
safety for arbitrary workloads. These two lessons use much smaller allocations;
consult the actual kernel-RSS samples in [VERIFICATION.md](VERIFICATION.md), and
Activity Monitor for whole-system memory pressure. Restart kernels to release
retained models. Do not raise allocator limits to fit larger models.

CPU is usually convenient for these tiny workloads. For optional MPS execution,
stop the existing Jupyter server, then launch with the device selected:

```bash
PTL_DEVICE=mps .venv/bin/jupyter lab notebooks/mac_learning
```

Restart the notebook kernel and Run All. Alternatively change the requested device
in the notebook's setup cell. `torch.backends.mps.is_available()` is checked and unsupported operations
fail visibly. CPU float64 numerical checks remain on CPU in either run. Do not set
`PYTORCH_ENABLE_MPS_FALLBACK=1` when validating MPS: it could hide CPU fallback. The
exact tiny MPS path was tested as recorded; that does not validate arbitrary models,
quantized training, or every operation supported by an evolving backend.

## Rebuild and verify

The builder reuses `scripts/to_ipynb.py` parsing functions and adds stable cell IDs,
the teaching kernel, current Python metadata and source hashes. It leaves the old
converter and historical notebook metadata unchanged.

```bash
.venv/bin/python notebooks/mac_learning/manage.py build
.venv/bin/python notebooks/mac_learning/manage.py run --device cpu
.venv/bin/python notebooks/mac_learning/manage.py run --device mps
```

Each run creates a new kernel using the Python that invokes the runner. CPU outputs
are saved in the study notebooks. MPS outputs are saved under ignored `.mps_runs/`,
so they cannot silently replace CPU outputs. `verification/*_cpu.json` and
`verification/*_mps.json` contain source hashes, wall time, cell counts and sampled
kernel RSS. The RSS sample is not a whole-system or GPU peak. The external wall time
includes kernel startup/shutdown; in-notebook timing excludes some setup/imports.
Runs set Hugging Face offline flags and use no remote resources, but do not globally
block network access. `--lessons 00` limits a build/run to that lesson.

A normal script path is also available, with plots suppressed from opening windows:

```bash
MPLBACKEND=Agg .venv/bin/python notebooks/mac_learning/00_local_setup_and_post_training_map.py
MPLBACKEND=Agg .venv/bin/python notebooks/mac_learning/01_tokens_probabilities_losses_and_gradients.py
```

The original utility tests can be checked separately with a temporary `pytest`
runner; they are not a dependency of the course. The historical M2 masking module
is absent and its contract tests skip, which must not be reported as implementation
success. New lesson assertions check relevant invariants directly.

## Continue learning

The current lesson is **00, section 1**. Read the first scenario, give your prediction,
then run the first setup cell. Later study the worked two-answer update, where the
loss changes logits. Use [PROGRESS.md](PROGRESS.md) to record predictions, explanations,
questions and corrections. No concept is marked mastered in this first delivery.
