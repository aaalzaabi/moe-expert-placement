
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare(['outputs/expert_demand_by_category.csv'])
# ---------------------------------------------------------------------------------------

import pandas as pd

df = pd.read_csv("outputs/workload_routing.csv")
timings = pd.read_csv("outputs/workload_timings.csv")

print("\nRequests:", df.request_id.nunique())
print("Categories:", sorted(df.category.unique()))

summary = (
    df.groupby(["category", "layer_id", "expert_id"], as_index=False)
      .agg(
          assignments=("token_assignments", "sum"),
          input_tokens=("input_tokens", "sum"),
      )
)

# Normalize for differences in prompt length and
# number of requests in each category.
summary["assignments_per_token"] = (
    summary["assignments"] / summary["input_tokens"]
)

summary.to_csv(
    "outputs/expert_demand_by_category.csv",
    index=False,
)

print("\nTop 10 layer-specific experts by assignment rate:")
print(
    summary.sort_values(
        "assignments_per_token",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\nForward-pass timing summary:")
print(
    timings.groupby("category")["forward_seconds"]
           .agg(["count", "mean", "max"])
           .to_string()
)

