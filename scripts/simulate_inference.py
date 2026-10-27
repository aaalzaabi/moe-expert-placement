
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare(['outputs/simulated_request_latency.csv', 'outputs/simulated_inference_comparison.csv'])
# ---------------------------------------------------------------------------------------

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

routing = pd.read_csv(OUT / "workload_routing.csv")
placements = pd.read_csv(
    OUT / "heterogeneous_placements.csv"
)

# Same held-out requests as Phase 3B
TEST_IDS = [1, 3, 5]

# SIMULATED hardware, not measured
GPU_SPEEDS = [1.0, 1.3, 0.8, 1.6]
COMPUTE_MS_PER_ASSIGNMENT = 0.5
ACTIVATION_BYTES_PER_ASSIGNMENT = 4096
NETWORK_BANDWIDTH_GBPS = 12.0
NETWORK_LATENCY_MS = 0.1
GATEWAY_GPU = 0

def simulate(strategy):
    layout = placements[
        placements.strategy == strategy
    ][["layer_id", "expert_id", "gpu_id"]]

    data = routing[
        routing.request_id.isin(TEST_IDS)
    ].merge(
        layout,
        on=["layer_id", "expert_id"],
        how="left",
        validate="many_to_one",
    )

    if data.gpu_id.isna().any():
        raise ValueError("Some experts have no GPU assignment")

    # Persistent simulated resources
    gpu_free = np.zeros(4, dtype=float)
    link_free = defaultdict(float)

    communication_bytes = 0
    remote_calls = 0
    completed = []

    def transfer(src, dst, ready, size_bytes):
        """Reserve a directed network link."""
        nonlocal communication_bytes

        if src == dst:
            return ready

        key = (src, dst)

        start = max(ready, link_free[key])

        bandwidth_bytes_per_ms = (
            NETWORK_BANDWIDTH_GBPS * 1e9 / 1000
        )

        duration = (
            NETWORK_LATENCY_MS
            + size_bytes / bandwidth_bytes_per_ms
        )

        # Link occupied during transfer.
        end = start + duration
        link_free[key] = end
        communication_bytes += size_bytes
        return end

    # Requests arrive at simulated time zero.
    # Deterministic request-order scheduling.
    for request_id in sorted(TEST_IDS):
        request = data[data.request_id == request_id]

        ready = 0.0

        for layer_id in sorted(request.layer_id.unique()):
            layer = request[
                (request.layer_id == layer_id)
                & (request.token_assignments > 0)
            ]

            finishes = []

            for row in layer.itertuples():
                gpu = int(row.gpu_id)
                count = int(row.token_assignments)

                size_bytes = (
                    count * ACTIVATION_BYTES_PER_ASSIGNMENT
                )

                if gpu != GATEWAY_GPU:
                    remote_calls += 1

                arrival = transfer(
                    GATEWAY_GPU,
                    gpu,
                    ready,
                    size_bytes,
                )

                compute_start = max(
                    arrival,
                    gpu_free[gpu]
                )

                compute_duration = (
                    count * COMPUTE_MS_PER_ASSIGNMENT
                    / GPU_SPEEDS[gpu]
                )

                compute_end = (
                    compute_start + compute_duration
                )

                gpu_free[gpu] = compute_end

                return_time = transfer(
                    gpu,
                    GATEWAY_GPU,
                    compute_end,
                    size_bytes,
                )

                finishes.append(return_time)

            # Next MoE layer waits for this one.
            if finishes:
                ready = max(finishes)

        completed.append({
            "strategy": strategy,
            "request_id": request_id,
            "latency_ms": ready,
        })

    return completed, {
        "strategy": strategy,
        "communication_mb": communication_bytes / 1e6,
        "remote_expert_calls": remote_calls,
        "makespan_ms": float(
    max(item["latency_ms"] for item in completed)
),
    }

all_requests = []
summary = []

for strategy in [
    "static",
    "balanced",
    "demand_aware",
]:
    request_results, stats = simulate(strategy)

    all_requests.extend(request_results)

    latencies = [
        x["latency_ms"] for x in request_results
    ]

    summary.append({
        **stats,
        "p50_latency_ms": float(
            np.percentile(latencies, 50)
        ),
        "p95_latency_ms": float(
            np.percentile(latencies, 95)
        ),
    })

request_df = pd.DataFrame(all_requests)
summary_df = pd.DataFrame(summary)

request_df.to_csv(
    OUT / "simulated_request_latency.csv",
    index=False,
)
summary_df.to_csv(
    OUT / "simulated_inference_comparison.csv",
    index=False,
)

print("\nPHASE 4 — SIMULATED INFERENCE")
print(summary_df.to_string(index=False))

print("\nFiles saved:")
print("outputs/simulated_request_latency.csv")
print("outputs/simulated_inference_comparison.csv")

