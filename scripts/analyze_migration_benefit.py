
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare(['outputs/migration_benefit.csv'])
# ---------------------------------------------------------------------------------------

import math
import pandas as pd

OUT = "outputs/"

latency = pd.read_csv(
    OUT + "simulated_inference_comparison.csv"
)

migration = pd.read_csv(
    OUT + "migration_events.csv"
)

static = latency.set_index("strategy").loc["static"]
optimized = latency.set_index("strategy").loc["demand_aware"]

# Use end-to-end makespan for the same three-request batch.
static_ms = float(static["makespan_ms"])
optimized_ms = float(optimized["makespan_ms"])

# Migration copies have already been scheduled in the
# separate migration simulator.
migration_ms = float(migration["finish_ms"].max())

batch_savings_ms = static_ms - optimized_ms

# Idealized policy: pause inference during migration,
# then run subsequent batches using the new placement.
break_even_batches = (
    math.ceil(migration_ms / batch_savings_ms)
    if batch_savings_ms > 0 else None
)

print("\nPHASE 4C — MIGRATION COST-BENEFIT")
print(f"Static batch makespan: {static_ms:.2f} ms")
print(f"Optimized batch makespan: {optimized_ms:.2f} ms")
print(f"Batch savings: {batch_savings_ms:.2f} ms")
print(f"Migration duration: {migration_ms:.2f} ms")
print(f"Break-even batches: {break_even_batches}")

rows = []

for batches in [1, 2, 5, 10, 20, 50, 100]:
    static_total = batches * static_ms
    optimized_total = (
        migration_ms + batches * optimized_ms
    )

    rows.append({
        "batches": batches,
        "static_total_ms": static_total,
        "optimized_total_ms": optimized_total,
        "net_savings_ms": static_total - optimized_total,
    })

result = pd.DataFrame(rows)

result.to_csv(
    OUT + "migration_benefit.csv",
    index=False
)

print("\nAMORTIZATION COMPARISON")
print(result.to_string(index=False))

