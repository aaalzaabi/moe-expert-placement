"""Project-root resolution so scripts, tests and the dashboard work from any cwd."""
from __future__ import annotations

import os
from pathlib import Path

# moeplace/paths.py -> project root is one level above the package.
PROJECT_ROOT = Path(os.environ.get("MOE_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

MODEL_DIR = PROJECT_ROOT / "models" / "OLMoE-1B-7B-0125-Instruct"
OUTPUTS = PROJECT_ROOT / "outputs"
PHASE4F = OUTPUTS / "phase4f"
RUNS = OUTPUTS / "runs"                  # new simulation runs (never the original files)
MEASUREMENTS = OUTPUTS / "measurements"  # new real OLMoE profiling runs
DOCS = PROJECT_ROOT / "docs"


def output(name: str | Path) -> Path:
    """Absolute path of a file under outputs/."""
    return OUTPUTS / name


def require(path: Path, hint: str = "") -> Path:
    """Return path if it exists, else raise a FileNotFoundError with a useful hint."""
    if not path.exists():
        rel = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
        msg = f"Required file not found: {rel}"
        if hint:
            msg += f"\n  -> {hint}"
        raise FileNotFoundError(msg)
    return path
