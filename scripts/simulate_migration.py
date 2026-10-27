
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare(['outputs/migration_events.csv'])
# ---------------------------------------------------------------------------------------

from collections import defaultdict
from pathlib import Path

import pandas as pd

OUT = Path("outputs")
df = pd.read_csv(OUT / "heterogeneous_placements.csv")

# Explicit simulation assumptions
EXPERT_BYTES = 8_000_000
BANDWIDTH_GBPS = 12.0
LINK_LATENCY_MS = 0.1
CAPACITY_SLOTS = 512  # Includes temporary staging space

def get_layout(strategy):
    rows = df[df.strategy == strategy]
    return {
        (int(r.layer_id), int(r.expert_id)): int(r.gpu_id)
        for r in rows.itertuples()
    }

old = get_layout("static")
new = get_layout("demand_aware")

assert old.keys() == new.keys()

def simulate_migration(approved=True):
    state = "proposed"
    events = []
    link_free = defaultdict(float)

    if not approved:
        return {
            "state": "rejected",
            "moved_experts": 0,
            "transfer_mb": 0,
            "activation_ms": 0,
        }

    state = "approved"

    old_counts = {
        g: sum(host == g for host in old.values())
        for g in range(4)
    }
    new_counts = {
        g: sum(host == g for host in new.values())
        for g in range(4)
    }

    # All new copies are staged before global activation.
    # Old copies remain resident throughout preparation.
    for g in range(4):
        incoming = sum(
            new[k] == g and old[k] != g for k in old
        )
        if old_counts[g] + incoming > CAPACITY_SLOTS:
            return {
                "state": "failed",
                "reason": f"GPU {g} lacks staging memory",
                "moved_experts": 0,
                "transfer_mb": 0,
                "activation_ms": None,
            }

    state = "transferring"

    for key in sorted(old):
        src, dst = old[key], new[key]
        if src == dst:
            continue

        start = link_free[(src, dst)]
        transfer_ms = (
            LINK_LATENCY_MS
            + EXPERT_BYTES / (BANDWIDTH_GBPS * 1e6)
        )
        finish = start + transfer_ms
        link_free[(src, dst)] = finish

        events.append({
            "layer_id": key[0],
            "expert_id": key[1],
            "src_gpu": src,
            "dst_gpu": dst,
            "start_ms": start,
            "finish_ms": finish,
            "weight_bytes": EXPERT_BYTES,
        })

    activation_ms = max(
        (event["finish_ms"] for event in events),
        default=0.0,
    )

    # Atomic simulated activation after all copies finish.
    state = "active"

    return {
        "state": state,
        "moved_experts": len(events),
        "transfer_mb": len(events) * EXPERT_BYTES / 1e6,
        "activation_ms": activation_ms,
        "events": events,
    }

result = simulate_migration(approved=True)

events = pd.DataFrame(result.pop("events", []))
events.to_csv(OUT / "migration_events.csv", index=False)

print("\nPHASE 4B — SIMULATED MIGRATION")
for key, value in result.items():
    print(f"{key}: {value}")

print("\nSaved: outputs/migration_events.csv")

