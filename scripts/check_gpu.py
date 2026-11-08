"""Inspect NVIDIA readiness without downloading weights, changing packages or running inference.

This file can be copied to the NVIDIA machine and run on its own:
    python3 check_gpu.py
    python3 check_gpu.py --json --checkpoint /path/to/OLMoE-1B-7B-0125-Instruct
Exit 0 means PyTorch can access CUDA; exit 2 means access/setup still needs work.
A successful check does not establish model memory feasibility or benchmark performance.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess


def inspect(checkpoint: Path | None = None) -> dict:
    report = {
        'python': platform.python_version(), 'platform': platform.system(),
        'nvidia_smi_available': bool(shutil.which('nvidia-smi')),
        'gpus': [], 'cuda_available': False, 'cuda_device_count': 0,
        'packages': {}, 'notes': [],
    }
    for package in ('torch', 'transformers', 'accelerate', 'safetensors'):
        try:
            report['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][package] = None
    if report['nvidia_smi_available']:
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version',
                '--format=csv,noheader,nounits',
            ], capture_output=True, text=True, timeout=20, check=True)
            for row in csv.reader(result.stdout.splitlines()):
                idx, name, total, free, driver = [part.strip() for part in row]
                report['gpus'].append({'index': int(idx), 'name': name, 'total_memory_mib': float(total),
                                       'free_memory_mib': float(free), 'driver': driver})
        except (subprocess.SubprocessError, ValueError) as exc:
            report['notes'].append(f'nvidia-smi could not report GPU details: {exc}')
    else:
        report['notes'].append('nvidia-smi was not found in this environment.')
    try:
        import torch  # optional: uses the installed environment, never installs anything
        report['torch_cuda_build'] = torch.version.cuda
        report['cuda_available'] = bool(torch.cuda.is_available())
        report['cuda_device_count'] = int(torch.cuda.device_count())
        report['cuda_devices'] = []
        for idx in range(report['cuda_device_count']):
            props = torch.cuda.get_device_properties(idx)
            report['cuda_devices'].append({'index': idx, 'name': props.name,
                                           'compute_capability': f'{props.major}.{props.minor}',
                                           'total_memory_bytes': int(props.total_memory)})
    except Exception as exc:
        report['notes'].append(f'PyTorch CUDA inspection failed: {exc}')
    if checkpoint is not None:
        checkpoint = checkpoint.expanduser().resolve()
        shards = list(checkpoint.glob('*.safetensors')) if checkpoint.is_dir() else []
        report['checkpoint'] = {
            'path': str(checkpoint), 'config_present': (checkpoint / 'config.json').is_file(),
            'weight_shards': len(shards), 'stored_weight_bytes': sum(p.stat().st_size for p in shards),
        }
    report['notes'].append('No model was loaded and no inference benchmark was run. Memory availability is a snapshot.')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print machine-readable hardware details')
    parser.add_argument('--checkpoint', type=Path, help='Optional local checkpoint folder to inspect (no download)')
    args = parser.parse_args()
    report = inspect(args.checkpoint)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Python {report['python']} on {report['platform']}")
        print(f"PyTorch CUDA access: {'READY' if report['cuda_available'] else 'NOT READY'}")
        print(f"CUDA devices visible to PyTorch: {report['cuda_device_count']}")
        for gpu in report['gpus']:
            print(f"GPU {gpu['index']}: {gpu['name']} | {gpu['total_memory_mib'] / 1024:.1f} GiB total, "
                  f"{gpu['free_memory_mib'] / 1024:.1f} GiB free | driver {gpu['driver']}")
        for package, version in report['packages'].items():
            print(f"{package}: {version or 'not installed'}")
        for note in report['notes']:
            print(note)
    return 0 if report['cuda_available'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
