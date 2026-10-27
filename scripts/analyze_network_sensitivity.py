
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare(['outputs/phase4f/network/network_results.csv', 'outputs/phase4f/network/network_improvement.csv'])
# ---------------------------------------------------------------------------------------

from pathlib import Path
import pandas as pd

FOLDER = Path("outputs/phase4f/network")

speeds = [3, 12, 25, 100]
results = []

for speed in speeds:
    file = FOLDER / f"bandwidth_{speed}.csv"

    df = pd.read_csv(file)

    required = {
        "policy",
        "p95_latency_ms",
        "makespan_ms",
        "migration_mb",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{file}: missing columns {missing}"
        )

    for row in df.to_dict("records"):
        results.append({
            "bandwidth_gb_per_s": speed,
            **row,
        })

comparison = pd.DataFrame(results)

# Separate static and migration policies.
static = comparison[
    comparison.policy == "static"
]

migrate = comparison[
    comparison.policy == "migrate"
]

merged = static.merge(
    migrate,
    on="bandwidth_gb_per_s",
    suffixes=("_static", "_migrate"),
    validate="one_to_one",
)

merged["p95_improvement_pct"] = (
    (
        merged["p95_latency_ms_static"]
        - merged["p95_latency_ms_migrate"]
    )
    / merged["p95_latency_ms_static"]
    * 100
)

merged["makespan_improvement_pct"] = (
    (
        merged["makespan_ms_static"]
        - merged["makespan_ms_migrate"]
    )
    / merged["makespan_ms_static"]
    * 100
)

comparison.to_csv(
    FOLDER / "network_results.csv",
    index=False,
)

merged.to_csv(
    FOLDER / "network_improvement.csv",
    index=False,
)

print("\nNETWORK SENSITIVITY RESULTS")
print(
    merged[
        [
            "bandwidth_gb_per_s",
            "p95_latency_ms_static",
            "p95_latency_ms_migrate",
            "p95_improvement_pct",
            "makespan_improvement_pct",
        ]
    ].round(3).to_string(index=False)
)

