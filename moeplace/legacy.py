"""Helpers that keep the original phase scripts runnable but safe.

prepare() makes the script run from the project root (so its relative
"outputs/..." paths resolve from any cwd) and refuses to overwrite existing
result files unless --force is passed.
"""
from __future__ import annotations

import os
import sys

from .paths import PROJECT_ROOT


def prepare(outputs: list[str], measured: bool = False) -> None:
    os.chdir(PROJECT_ROOT)
    existing = [p for p in outputs if (PROJECT_ROOT / p).exists()]
    if not existing or "--force" in sys.argv:
        return
    script = os.path.basename(sys.argv[0])
    lines = [f"{script}: refusing to overwrite existing results:"]
    lines += [f"  - {p}" for p in existing]
    if measured:
        lines += ["These are REAL OLMoE measurements. To collect new measurements without touching them, use:",
                  "  python scripts/profile_routing.py --help"]
    else:
        lines += ["To run new simulations into outputs/runs/ instead, use:",
                  "  python scripts/run_experiment.py --help"]
    lines += [f"Re-run with --force to overwrite (back the files up first)."]
    sys.exit("\n".join(lines))
