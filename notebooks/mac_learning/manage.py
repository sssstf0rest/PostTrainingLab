#!/usr/bin/env python3
"""Build paired notebooks using the existing converter; execute in fresh kernels.

Run from any directory using the teaching environment's Python. No model downloads.
CPU run outputs are saved to the course notebooks. MPS outputs go to a separate
ignored directory unless --output-dir is supplied. JSON records are execution evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nbformat
import psutil
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def lesson_sources(prefixes):
    paths = sorted(HERE.glob('[0-9][0-9]_*.py'))
    if prefixes:
        paths = [p for p in paths if p.name[:2] in prefixes]
    if not paths:
        raise ValueError('No matching lessons.')
    return paths


def build(source):
    spec = importlib.util.spec_from_file_location('original_converter', ROOT / 'scripts/to_ipynb.py')
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    cells = []
    for kind, lines in converter.split_cells(source.read_text(encoding='utf-8')):
        cleaned = converter.clean(kind, lines)
        if not cleaned:
            continue
        text = '\n'.join(cleaned)
        constructor = nbformat.v4.new_markdown_cell if kind == 'markdown' else nbformat.v4.new_code_cell
        cell = constructor(text)
        cell.id = hashlib.sha256(f'{len(cells)}:{text}'.encode()).hexdigest()[:12]
        cells.append(cell)
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'PostTrainingLab Mac', 'language': 'python', 'name': 'ptl-mac'},
        'language_info': {'name': 'python', 'version': platform.python_version()},
        'mac_learning': {'source': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
    })
    nbformat.validate(nb)
    nbformat.write(nb, source.with_suffix('.ipynb'))
    print(f'Built {source.with_suffix(".ipynb").name}: {len(cells)} cells', flush=True)


def run(source, device, output_dir):
    path = source.with_suffix('.ipynb')
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if nb.metadata.get('mac_learning', {}).get('source_sha256') != digest:
        raise ValueError(f'{path.name} is stale: run manage.py build before executing.')
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.execution_count = None
            cell.outputs = []
    env = {**os.environ, 'PTL_DEVICE': device, 'PTL_QUICK_MODE': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'OMP_NUM_THREADS': '2', 'MPLBACKEND': 'module://matplotlib_inline.backend_inline'}
    # An MPS run is only meaningful if unsupported operations cannot silently move to CPU.
    env.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)
    rss_samples = []
    finished = threading.Event()
    with tempfile.TemporaryDirectory(prefix='ptl-kernel-') as tmp:
        kernel_dir = Path(tmp) / 'ptl-mac'
        kernel_dir.mkdir()
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'PostTrainingLab Mac', 'language': 'python',
        }))
        ksm = KernelSpecManager(kernel_dirs=[tmp])
        km = KernelManager(kernel_name='ptl-mac', kernel_spec_manager=ksm)
        client = NotebookClient(nb, km=km, timeout=180,
                                resources={'metadata': {'path': str(HERE)}})

        def sample_rss():
            while not finished.is_set():
                try:
                    pid = getattr(getattr(km, 'provisioner', None), 'pid', None)
                    if pid:
                        rss_samples.append(psutil.Process(pid).memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                finished.wait(0.05)

        monitor = threading.Thread(target=sample_rss, daemon=True)
        monitor.start()
        started = time.perf_counter()
        try:
            # Explicit cleanup is needed because this client receives an external manager.
            with client.setup_kernel(env=env, cwd=str(HERE), cleanup_kc=True):
                client.execute(env=env, cwd=str(HERE))
        finally:
            elapsed = time.perf_counter() - started
            finished.set()
            monitor.join(timeout=2)
    errors = [o for c in nb.cells if c.cell_type == 'code'
              for o in c.outputs if o.output_type == 'error']
    assert not errors
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code_cells)
    nbformat.validate(nb)
    record = {
        'lesson': path.name, 'device': device, 'quick_mode': True, 'status': 'passed',
        'executed_at': datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
        'source_sha256': digest, 'base_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'python': platform.python_version(), 'platform': platform.system(),
        'macOS': platform.mac_ver()[0], 'machine': platform.machine(),
        'wall_seconds_including_kernel_start_and_shutdown': round(elapsed, 3),
        'sampled_kernel_max_rss_bytes': max(rss_samples) if rss_samples else None,
        'rss_sample_count': len(rss_samples), 'rss_poll_seconds': 0.05,
        'memory_limitation': 'Observed maximum kernel RSS, sampled every 50 ms; excludes runner, other apps and some driver/shared memory. Not a guaranteed process/device peak.',
        'code_cells': len(code_cells), 'errors': len(errors),
        'network_status': 'No remote assets or APIs used; HF offline flags set. Network was not globally blocked.',
    }
    nb.metadata.mac_learning['execution'] = record
    output_dir.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output_dir / path.name)
    evidence_dir = HERE / 'verification'
    evidence_dir.mkdir(exist_ok=True)
    (evidence_dir / f'{source.stem}_{device}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'run'])
    parser.add_argument('--lessons', nargs='*', help='Two-digit lesson prefixes, e.g. 00 01')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    for source in lesson_sources(args.lessons):
        if args.action == 'build':
            build(source)
        else:
            destination = args.output_dir or (HERE if args.device == 'cpu' else HERE / '.mps_runs')
            run(source, args.device, destination.resolve())


if __name__ == '__main__':
    main()
