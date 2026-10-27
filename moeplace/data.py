"""Loaders with schema validation. Never imports torch; never loads model weights."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config as C
from .paths import MEASUREMENTS, MODEL_DIR, OUTPUTS, PHASE4F, require

# Required columns for each saved CSV (relative to outputs/).
SCHEMAS: dict[str, set[str]] = {
    "expert_routing.csv": {"layer_id", "expert_id", "token_assignments"},
    "workload_routing.csv": {"request_id", "category", "layer_id", "expert_id", "token_assignments", "input_tokens"},
    "workload_timings.csv": {"request_id", "category", "input_tokens", "forward_seconds"},
    "expert_demand_by_category.csv": {"category", "layer_id", "expert_id", "assignments", "input_tokens", "assignments_per_token"},
    "gpu_placements.csv": {"layer_id", "expert_id", "demand", "gpu_id", "strategy"},
    "placement_comparison.csv": {"strategy", "max_normalized_load", "mean_normalized_load", "load_imbalance"},
    "heterogeneous_placements.csv": {"layer_id", "expert_id", "train_demand", "test_demand", "gpu_id", "strategy"},
    "heldout_comparison.csv": {"strategy", "peak_test_load", "mean_test_load", "test_imbalance"},
    "simulated_inference_comparison.csv": {"strategy", "communication_mb", "remote_expert_calls", "makespan_ms", "p50_latency_ms", "p95_latency_ms"},
    "simulated_request_latency.csv": {"strategy", "request_id", "latency_ms"},
    "migration_events.csv": {"layer_id", "expert_id", "src_gpu", "dst_gpu", "start_ms", "finish_ms", "weight_bytes"},
    "migration_benefit.csv": {"batches", "static_total_ms", "optimized_total_ms", "net_savings_ms"},
    "integrated_comparison.csv": {"policy", "migration_state", "migration_activation_ms", "moved_experts", "migration_mb", "total_link_mb", "remote_expert_calls", "p50_latency_ms", "p95_latency_ms", "makespan_ms"},
    "integrated_request_latency.csv": {"request_id", "arrival_ms", "finish_ms", "latency_ms", "layout", "policy"},
    "integrated_migration_events.csv": {"layer_id", "expert_id", "src_gpu", "dst_gpu", "start_requested_ms", "finish_ms", "weight_bytes"},
    "policy_comparison.csv": {"policy", "migration_state", "moved_experts", "migration_mb", "p50_latency_ms", "p95_latency_ms", "makespan_ms"},
    "policy_decision.csv": {"decision", "forecast_request_ids", "evaluation_request_ids", "forecast_makespan_gain_pct", "forecast_p95_regression_pct", "min_makespan_gain_pct", "max_p95_regression_pct", "migration_feasible"},
    "policy_request_latency.csv": {"request_id", "arrival_ms", "finish_ms", "latency_ms", "layout", "policy"},
    "phase4f/memory/expert_memory_summary.csv": {"layer_id", "expert_tensors", "expert_bytes", "expert_mb", "per_expert_mb_estimate"},
    "phase4f/memory/expert_tensor_metadata.csv": {"layer_id", "tensor_name", "shape", "dtype", "bytes", "size_mb"},
    "phase4f/network/network_results.csv": {"bandwidth_gb_per_s", "policy", "p95_latency_ms", "makespan_ms", "migration_mb"},
    "phase4f/network/network_improvement.csv": {"bandwidth_gb_per_s", "p95_improvement_pct", "makespan_improvement_pct"},
}
for _bw in C.PHASE4F_BANDWIDTHS:
    for _kind in ("network", "calibrated"):
        SCHEMAS[f"phase4f/{_kind}/bandwidth_{_bw}.csv"] = SCHEMAS["integrated_comparison.csv"]


class SchemaError(ValueError):
    pass


def read_csv(name: str | Path, *, base: Path = OUTPUTS, schema: set[str] | None = None) -> pd.DataFrame:
    """Read outputs/<name> and check required columns."""
    path = require(base / name, hint="Run the producing script or see docs/PROJECT_AUDIT.md for the data lineage.")
    df = pd.read_csv(path)
    required = schema if schema is not None else SCHEMAS.get(str(name), set())
    missing = required - set(df.columns)
    if missing:
        raise SchemaError(f"{path.name}: missing required columns {sorted(missing)}")
    return df


def validate_routing(df: pd.DataFrame, top_k: int = C.TOP_K) -> None:
    """Check measured routing invariants: complete expert grid and top-k per token per layer."""
    per_req = df.groupby("request_id").size()
    if (per_req != C.TOTAL_EXPERT_SLOTS).any():
        bad = per_req[per_req != C.TOTAL_EXPERT_SLOTS].to_dict()
        raise SchemaError(f"Each request needs {C.TOTAL_EXPERT_SLOTS} (layer, expert) rows; got {bad}")
    if (df.token_assignments < 0).any():
        raise SchemaError("token_assignments must be non-negative")
    sums = df.groupby(["request_id", "layer_id"]).agg(s=("token_assignments", "sum"), t=("input_tokens", "first"))
    bad = sums[sums.s != sums.t * top_k]
    if not bad.empty:
        raise SchemaError(f"Routing violates top-{top_k}: {len(bad)} (request, layer) pairs do not sum to tokens*{top_k}")


def load_routing(source: str | Path | None = None) -> pd.DataFrame:
    """Measured per-request routing. source=None -> the original Phase 2 file."""
    if source is None:
        df = read_csv("workload_routing.csv")
    else:
        path = Path(source)
        if path.is_dir():
            path = path / "workload_routing.csv"
        df = read_csv(path.name, base=path.parent, schema=SCHEMAS["workload_routing.csv"])
    validate_routing(df)
    return df


def load_timings(source: str | Path | None = None) -> pd.DataFrame:
    if source is None:
        return read_csv("workload_timings.csv")
    path = Path(source)
    if path.is_dir():
        path = path / "workload_timings.csv"
    return read_csv(path.name, base=path.parent, schema=SCHEMAS["workload_timings.csv"])


def measurement_runs() -> dict[str, Path]:
    """Routing datasets available to the dashboard: the original + any new profiling runs."""
    runs = {"phase2_original (6 prompts)": OUTPUTS}
    if MEASUREMENTS.exists():
        for d in sorted(MEASUREMENTS.iterdir()):
            if (d / "workload_routing.csv").exists():
                runs[d.name] = d
    return runs


def demand(routing: pd.DataFrame, request_ids=None) -> pd.Series:
    """Total token assignments per (layer_id, expert_id), complete 1024-slot index."""
    sub = routing if request_ids is None else routing[routing.request_id.isin(request_ids)]
    index = pd.MultiIndex.from_product(
        [range(C.NUM_LAYERS), range(C.EXPERTS_PER_LAYER)], names=["layer_id", "expert_id"]
    )
    return sub.groupby(["layer_id", "expert_id"])["token_assignments"].sum().reindex(index, fill_value=0)


def build_traces(routing: pd.DataFrame, request_ids) -> dict[int, dict[int, list[tuple[int, int]]]]:
    """{request: {layer: [(expert, token_assignments), ...]}} for nonzero assignments."""
    traces = {}
    for rid in request_ids:
        rows = routing[(routing.request_id == rid) & (routing.token_assignments > 0)]
        if rows.empty:
            raise ValueError(f"No nonzero routing for request {rid}")
        traces[int(rid)] = {
            int(layer): [(int(r.expert_id), int(r.token_assignments)) for r in part.itertuples()]
            for layer, part in rows.groupby("layer_id", sort=True)
        }
    return traces


# --------------------------------------------------------------------------
# Lightweight model metadata (JSON only; never touches safetensors payloads)
# --------------------------------------------------------------------------
def model_config() -> dict:
    return json.loads(require(MODEL_DIR / "config.json", "The OLMoE checkpoint folder is missing.").read_text())


def checkpoint_index() -> dict:
    return json.loads(require(MODEL_DIR / "model.safetensors.index.json").read_text())


def checkpoint_files() -> pd.DataFrame:
    rows = []
    if MODEL_DIR.exists():
        for p in sorted(MODEL_DIR.iterdir()):
            if p.is_file():
                rows.append({"file": p.name, "size_gb": p.stat().st_size / C.GB})
    return pd.DataFrame(rows)


def expert_memory() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = read_csv("phase4f/memory/expert_memory_summary.csv")
    tensors = read_csv("phase4f/memory/expert_tensor_metadata.csv")
    tensors["expert_id"] = tensors.tensor_name.str.extract(r"experts\.(\d+)\.").astype(int)
    tensors["projection"] = tensors.tensor_name.str.extract(r"\.(\w+_proj)\.")
    return summary, tensors


def phase4f(kind: str) -> pd.DataFrame:
    """Saved Phase 4F sweep: kind in {'network' (8 MB experts), 'calibrated' (12.583 MB)}."""
    frames = []
    for bw in C.PHASE4F_BANDWIDTHS:
        df = read_csv(f"phase4f/{kind}/bandwidth_{bw}.csv")
        frames.append(df.assign(bandwidth_gb_per_s=bw))
    out = pd.concat(frames, ignore_index=True)
    out["expert_mb"] = C.LEGACY_EXPERT_BYTES / C.MB if kind == "network" else C.CHECKPOINT_EXPERT_BYTES / C.MB
    out["scenario"] = "8 MB placeholder" if kind == "network" else "12.583 MB checkpoint-derived"
    return out
