
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare(['outputs/heterogeneous_placements.csv', 'outputs/heldout_comparison.csv'])
# ---------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

df = pd.read_csv("outputs/workload_routing.csv")

# Three training requests and three held-out requests.
train_ids = [0, 2, 4]
test_ids = [1, 3, 5]

train = df[df.request_id.isin(train_ids)]
test = df[df.request_id.isin(test_ids)]

# Hypothetical heterogeneous GPU speeds.
# These are NOT measured hardware speeds.
speeds = np.array([1.0, 1.3, 0.8, 1.6])
capacities = np.array([256, 256, 256, 256])

keys = ["layer_id", "expert_id"]

def aggregate(data, name):
    return (
        data.groupby(keys)["token_assignments"]
        .sum()
        .rename(name)
    )

experts = pd.concat(
    [aggregate(train, "train_demand"),
     aggregate(test, "test_demand")],
    axis=1,
).fillna(0).reset_index()

experts = experts.sort_values(keys).reset_index(drop=True)

assert len(experts) == 1024
assert set(train_ids).isdisjoint(test_ids)
assert sum(capacities) >= len(experts)

def place(strategy):
    loads = np.zeros(4, dtype=float)
    counts = np.zeros(4, dtype=int)
    assigned = {}

    if strategy == "static":
        order = list(experts.index)
    else:
        order = sorted(
            experts.index,
            key=lambda i: -experts.loc[i, "train_demand"]
        )

    for i in order:
        demand = float(experts.loc[i, "train_demand"])

        feasible = [
            g for g in range(4)
            if counts[g] < capacities[g]
        ]

        if strategy == "static":
            gpu = i % 4

        elif strategy == "balanced":
            gpu = min(feasible, key=lambda g: (loads[g], g))

        elif strategy == "demand_aware":
            gpu = min(
                feasible,
                key=lambda g: (
                    (loads[g] + demand) / speeds[g], g
                )
            )

        else:
            raise ValueError(strategy)

        assigned[i] = gpu
        counts[gpu] += 1
        loads[gpu] += demand

    result = experts.copy()
    result["gpu_id"] = result.index.map(assigned)
    result["strategy"] = strategy
    return result

placements = []
metrics = []

for strategy in ["static", "balanced", "demand_aware"]:
    result = place(strategy)
    placements.append(result)

    # Evaluate only on unseen requests.
    gpu_demand = (
        result.groupby("gpu_id")["test_demand"]
        .sum()
        .reindex(range(4), fill_value=0)
        .to_numpy()
    )

    normalized = gpu_demand / speeds

    metrics.append({
        "strategy": strategy,
        "peak_test_load": normalized.max(),
        "mean_test_load": normalized.mean(),
        "test_imbalance": (
            normalized.max() / normalized.mean()
            if normalized.mean() > 0 else 1.0
        ),
    })

pd.concat(placements).to_csv(
    "outputs/heterogeneous_placements.csv",
    index=False
)

summary = pd.DataFrame(metrics)
summary.to_csv(
    "outputs/heldout_comparison.csv",
    index=False
)

print("\nHELD-OUT GPU PLACEMENT EVALUATION")
print(summary.to_string(index=False))

baseline = summary.loc[
    summary.strategy == "static", "peak_test_load"
].iloc[0]

for _, row in summary.iterrows():
    reduction = 100 * (
        baseline - row.peak_test_load
    ) / baseline
    print(
        f"{row.strategy}: "
        f"{reduction:.2f}% peak-load reduction vs static"
    )

