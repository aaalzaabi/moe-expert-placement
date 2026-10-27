
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare(['outputs/gpu_placements.csv', 'outputs/placement_comparison.csv'])
# ---------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("outputs/workload_routing.csv")
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

# Simulated GPU configuration.
# These values are assumptions, NOT hardware measurements.
gpu_speeds = np.array([1.0, 1.0, 1.0, 1.0])
gpu_capacity = np.array([256, 256, 256, 256])

df = pd.read_csv(INPUT)

required = {
    "request_id", "layer_id",
    "expert_id", "token_assignments"
}
assert required.issubset(df.columns)

# One demand value for every layer-specific expert.
experts = (
    df.groupby(["layer_id", "expert_id"], as_index=False)
      ["token_assignments"].sum()
      .rename(columns={"token_assignments": "demand"})
      .sort_values(["layer_id", "expert_id"])
      .reset_index(drop=True)
)

assert len(experts) == 1024
assert sum(gpu_capacity) >= len(experts)

def assign(strategy):
    loads = np.zeros(len(gpu_speeds), dtype=float)
    counts = np.zeros(len(gpu_speeds), dtype=int)
    placements = {}

    if strategy == "static":
        ordering = list(experts.index)
    else:
        ordering = sorted(
            experts.index,
            key=lambda i: -experts.loc[i, "demand"]
        )

    for idx in ordering:
        demand = float(experts.loc[idx, "demand"])

        feasible = [
            g for g in range(len(gpu_speeds))
            if counts[g] < gpu_capacity[g]
        ]

        if not feasible:
            raise RuntimeError("Insufficient capacity")

        if strategy == "static":
            # Fixed round-robin assignment.
            gpu = feasible[0]
            for candidate in range(len(gpu_speeds)):
                g = (int(idx) + candidate) % len(gpu_speeds)
                if g in feasible:
                    gpu = g
                    break

        elif strategy == "balanced":
            # Greedy load balancing for identical GPUs.
            gpu = min(feasible, key=lambda g: (loads[g], g))

        elif strategy == "demand_aware":
            # Compare projected normalized GPU loads.
            gpu = min(
                feasible,
                key=lambda g: (
                    (loads[g] + demand) / gpu_speeds[g],
                    g
                )
            )

        else:
            raise ValueError(strategy)

        placements[idx] = gpu
        loads[gpu] += demand
        counts[gpu] += 1

    result = experts.copy()
    result["gpu_id"] = result.index.map(placements)
    result["strategy"] = strategy

    return result

strategies = [
    "static",
    "balanced",
    "demand_aware"
]

all_placements = []
metrics = []

for strategy in strategies:
    placement = assign(strategy)
    all_placements.append(placement)

    gpu_load = (
        placement.groupby("gpu_id")["demand"]
                 .sum()
                 .reindex(range(len(gpu_speeds)), fill_value=0)
                 .to_numpy()
    )

    normalized_load = gpu_load / gpu_speeds

    metrics.append({
        "strategy": strategy,
        "max_normalized_load": normalized_load.max(),
        "mean_normalized_load": normalized_load.mean(),
        "load_imbalance": (
            normalized_load.max() /
            normalized_load.mean()
            if normalized_load.mean() > 0 else 1.0
        ),
    })

placements = pd.concat(all_placements, ignore_index=True)
results = pd.DataFrame(metrics)

placements.to_csv(
    OUTPUT / "gpu_placements.csv",
    index=False
)
results.to_csv(
    OUTPUT / "placement_comparison.csv",
    index=False
)

print("\nPLACEMENT COMPARISON")
print(results.to_string(index=False))
print("\nSaved:")
print("outputs/gpu_placements.csv")
print("outputs/placement_comparison.csv")

