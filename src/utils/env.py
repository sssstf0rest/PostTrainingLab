"""Environment probing.

WHY THIS FILE EXISTS
--------------------
Deep-learning bugs are overwhelmingly *environment* bugs, not logic bugs: a CUDA
version that does not match the torch wheel, a GPU with less VRAM than you assumed,
a driver that is too old, a bf16 kernel that silently is not available. Every
experiment we run must record exactly what machine it ran on, otherwise a result is
not reproducible and not trustworthy.

So this module does two jobs:
  1. Human-readable report  -> `scripts/env_report.py` (run it after every new box).
  2. Machine-readable dict  -> embedded into every experiment's metadata (M4+).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GPUInfo:
    """One accelerator device.

    total_memory_gb is the number that actually constrains us: model weights,
    gradients, optimizer states and activations must all fit inside it.
    """

    index: int
    name: str
    total_memory_gb: float
    capability: str | None = None  # CUDA compute capability, e.g. "8.9". >=8.0 => native bf16.


@dataclass
class EnvInfo:
    python_version: str
    platform: str
    machine: str
    cpu_count: int | None
    torch_version: str | None = None
    accelerator: str = "cpu"  # "cuda" | "mps" | "cpu"
    cuda_available: bool = False
    torch_cuda_version: str | None = None  # CUDA version torch was COMPILED against
    driver_cuda_version: str | None = None  # CUDA version the DRIVER supports (nvidia-smi)
    driver_version: str | None = None
    mps_available: bool = False
    gpus: list[GPUInfo] = field(default_factory=list)
    bf16_supported: bool = False
    disk_free_gb: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nvidia_smi() -> tuple[str | None, str | None]:
    """Return (driver_version, max CUDA version the driver supports).

    Note the distinction that trips everyone up:
      - `nvidia-smi` reports the CUDA version the DRIVER can support (an upper bound).
      - `torch.version.cuda` reports the CUDA toolkit the torch WHEEL was built with.
    The wheel's version must be <= the driver's. Mismatch here is the #1 cause of
    "CUDA error: no kernel image is available for execution on the device".
    """
    if shutil.which("nvidia-smi") is None:
        return None, None
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0].strip()
        header = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15, check=True
        ).stdout
        cuda = None
        for line in header.splitlines():
            if "CUDA Version" in line:
                cuda = line.split("CUDA Version:")[1].split("|")[0].strip()
                break
        return drv, cuda
    except Exception:
        return None, None


def collect_env() -> EnvInfo:
    """Probe the current machine. Never raises; missing pieces become None + a note."""
    info = EnvInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=None,
    )
    try:
        import os

        info.cpu_count = os.cpu_count()
        info.disk_free_gb = round(shutil.disk_usage(".").free / 1024**3, 1)
    except Exception:
        pass

    try:
        import torch
    except ImportError:
        info.notes.append("torch is not installed — nothing can run yet.")
        return info

    info.torch_version = torch.__version__
    info.cuda_available = torch.cuda.is_available()
    info.torch_cuda_version = torch.version.cuda
    info.mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if info.cuda_available:
        info.accelerator = "cuda"
        info.driver_version, info.driver_cuda_version = _nvidia_smi()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info.gpus.append(
                GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory_gb=round(props.total_memory / 1024**3, 1),
                    capability=f"{props.major}.{props.minor}",
                )
            )
        # bf16 needs Ampere (SM 8.0) or newer. On older cards (T4, V100) you must use fp16,
        # which has a much narrower exponent range and therefore needs loss scaling.
        info.bf16_supported = torch.cuda.is_bf16_supported()
        if info.driver_cuda_version and info.torch_cuda_version:
            try:
                if float(info.torch_cuda_version) > float(info.driver_cuda_version):
                    info.notes.append(
                        f"torch was built for CUDA {info.torch_cuda_version} but the driver only "
                        f"supports up to {info.driver_cuda_version}. Expect kernel launch failures."
                    )
            except ValueError:
                pass
    elif info.mps_available:
        info.accelerator = "mps"
        info.notes.append(
            "Apple MPS backend. Fine for tokenization, inspection and tiny-model demos "
            "(M1-M3). NOT suitable for real training: no bf16 autocast parity, no "
            "FlashAttention, no DeepSpeed, no bitsandbytes. Rent a CUDA box for M4+."
        )
        info.bf16_supported = False
    else:
        info.notes.append("CPU only. Everything will work but slowly.")

    return info
