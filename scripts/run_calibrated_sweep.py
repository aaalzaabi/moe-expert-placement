
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare(['outputs/phase4f/calibrated/bandwidth_3.csv', 'outputs/phase4f/calibrated/bandwidth_12.csv', 'outputs/phase4f/calibrated/bandwidth_25.csv', 'outputs/phase4f/calibrated/bandwidth_100.csv'])
# ---------------------------------------------------------------------------------------

from pathlib import Path
import subprocess
import sys
import shutil
import re

script = Path("scripts/simulate_integrated.py")
output = Path("outputs/integrated_comparison.csv")
destination = Path("outputs/phase4f/calibrated")

destination.mkdir(parents=True, exist_ok=True)

original = script.read_text()

pattern = r"^BANDWIDTH_GBPS\s*=\s*[0-9.]+"
expert_pattern = r"^EXPERT_BYTES\s*=\s*([\d_]+)"

match = re.search(expert_pattern, original, re.MULTILINE)
assert match, "EXPERT_BYTES setting not found"

assert int(match.group(1).replace("_", "")) == 12_582_912, (
    "Expert size must be calibrated before the sweep."
)

try:
    for bandwidth in [3, 12, 25, 100]:
        print(f"\nRunning {bandwidth} GB/s...", flush=True)

        modified, count = re.subn(
            pattern,
            f"BANDWIDTH_GBPS = {float(bandwidth)}",
            original,
            count=1,
            flags=re.MULTILINE,
        )

        assert count == 1, "Bandwidth setting not found"
        script.write_text(modified)

        subprocess.run(
            [sys.executable, str(script), "--force"],
            check=True,
        )

        shutil.copy2(
            output,
            destination / f"bandwidth_{bandwidth}.csv",
        )

        print(f"Saved bandwidth_{bandwidth}.csv", flush=True)

finally:
    # Restore the original simulator even if an experiment fails.
    script.write_text(original)

print("\nAll calibrated experiments saved.")


