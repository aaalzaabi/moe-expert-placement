"""Deterministic discrete-event simulator for MoE inference + expert migration.

`run()` is a parameterised port of scripts/simulate_integrated.py::run (Phase 4D),
with identical event ordering, so default/preset configs reproduce the saved
CSVs bit-for-bit (tests/test_reproduction.py). Extensions are opt-in:

* record=True        -> per-task trace (GPU compute, link transfers, layer spans)
* link_model         -> "shared_port" makes transfers contend for per-GPU NIC ports
* gpu_memory_gb      -> byte-based staging-memory check in addition to slot counts
* fail_after_copies  -> fault injection: copy N+1 fails, migration rolls back
* arrival_process    -> "poisson" arrivals with an explicit seed

Model (all HYPOTHETICAL hardware): each request is a prefill pass whose routing
was measured on OLMoE; per layer, the gateway GPU dispatches activations to every
selected expert's GPU, the GPU computes n * compute_ms_per_assignment / speed,
and results return to the gateway. Layer L+1 starts only after every expert of
layer L returned. Each GPU is one FIFO compute queue; links are FIFO queues.
Resources are reserved in event-processing order (no back-filling of idle gaps).

`run_batch_at_zero()` and `migration_only()` port the simpler Phase 4A and 4B scripts.
"""
from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

Layout = dict[tuple[int, int], int]
Traces = dict[int, dict[int, list[tuple[int, int]]]]


@dataclass
class SimResult:
    requests: pd.DataFrame
    migrations: pd.DataFrame
    summary: dict
    states: list[dict] = field(default_factory=list)
    compute: pd.DataFrame | None = None
    transfers: pd.DataFrame | None = None
    layers: pd.DataFrame | None = None


def arrival_times(n: int, cfg: C.SimConfig) -> list[float]:
    if cfg.arrival_process == "fixed":
        return [i * cfg.arrival_gap_ms for i in range(n)]
    rng = np.random.default_rng(cfg.seed)
    gaps = rng.exponential(cfg.arrival_gap_ms, size=n) if cfg.arrival_gap_ms > 0 else np.zeros(n)
    gaps[0] = 0.0
    return list(np.cumsum(gaps).astype(float))


class _Network:
    def __init__(self, cfg: C.SimConfig, totals, record):
        self.cfg = cfg
        self.totals = totals
        self.link_free = defaultdict(float)
        self.egress = defaultdict(float)
        self.ingress = defaultdict(float)
        self.log = [] if record else None

    def move(self, src, dst, at, size, kind, ref=None):
        if src == dst:
            return at
        cost = C.transfer_ms(size, self.cfg.bandwidth_gb_per_s, self.cfg.link_latency_ms)
        if self.cfg.link_model == "directed":
            start = max(float(at), self.link_free[(src, dst)])
            end = start + cost
            self.link_free[(src, dst)] = end
        else:  # shared_port
            start = max(float(at), self.egress[src], self.ingress[dst])
            end = start + cost
            self.egress[src] = end
            self.ingress[dst] = end
        self.totals["communication_bytes"] += size
        if self.log is not None:
            self.log.append({"kind": kind, "src_gpu": src, "dst_gpu": dst, "ready_ms": float(at),
                             "start_ms": start, "end_ms": end, "bytes": size, "ref": ref})
        return end


def run(traces: Traces, layouts: dict[str, Layout], cfg: C.SimConfig, request_ids=None,
        allow_migration: bool = True, record: bool = False) -> SimResult:
    """Simulate requests (in request_ids order) arriving per cfg, optionally migrating
    cfg.source_strategy -> cfg.target_strategy at cfg.migration_start_ms."""
    request_ids = tuple(request_ids if request_ids is not None else cfg.test_ids)
    src_name, dst_name = cfg.source_strategy, cfg.target_strategy
    old, target = layouts[src_name], layouts[dst_name]
    n_gpus = cfg.num_gpus
    for name in (src_name, dst_name):
        bad = set(layouts[name].values()) - set(range(n_gpus))
        if bad:
            raise ValueError(f"Layout {name!r} references GPU ids {sorted(bad)} but config has {n_gpus} GPUs")
    if old.keys() != target.keys():
        raise ValueError("Layouts have different expert inventories")

    serial = itertools.count()
    events: list = []
    gpu_free = [0.0] * n_gpus
    totals = defaultdict(float)
    net = _Network(cfg, totals, record)
    inflight: dict = {}
    records, migration_records, states = [], [], []
    compute_log = [] if record else None
    layer_log = [] if record else None
    state = "proposed" if allow_migration else "disabled"
    states.append({"t_ms": 0.0, "state": state, "note": "initial"})
    active = src_name
    pending_copies = 0
    completed_copies = 0
    staged: set = set()
    migration_completed_at = None
    failure_reason = None

    def set_state(t, new, note=""):
        nonlocal state
        state = new
        states.append({"t_ms": float(t), "state": new, "note": note})

    def push(t, kind, payload):
        heapq.heappush(events, (float(t), next(serial), kind, payload))

    def enter_layer(rid, layer_index, at):
        req = inflight[rid]
        layers = sorted(traces[rid])
        if layer_log is not None and layer_index > 0:
            layer_log.append({"request_id": rid, "layer_id": layers[layer_index - 1],
                              "start_ms": req["layer_start"], "end_ms": at, "layout": req["layout"]})
        if layer_index == len(layers):
            records.append({"request_id": rid, "arrival_ms": req["arrival"], "finish_ms": at,
                            "latency_ms": at - req["arrival"], "layout": req["layout"]})
            return
        layer = layers[layer_index]
        choices = traces[rid][layer]
        req["outstanding"] = len(choices)
        req["max_finish"] = at
        req["next_layer"] = layer_index + 1
        req["layer_start"] = at
        for expert, n in choices:
            push(at, "expert_start", (rid, layer, expert, n))

    for rid, t0 in zip(request_ids, arrival_times(len(request_ids), cfg)):
        if rid not in traces:
            raise KeyError(f"No routing trace for request {rid}")
        push(t0, "request_arrival", rid)
    if allow_migration:
        push(cfg.migration_start_ms, "approve_migration", None)

    while events:
        t, _, kind, payload = heapq.heappop(events)
        if kind == "request_arrival":
            rid = payload
            inflight[rid] = {"arrival": t, "layout": active}  # pin the layout active at arrival
            enter_layer(rid, 0, t)

        elif kind == "expert_start":
            rid, layer, expert, n = payload
            gpu = layouts[inflight[rid]["layout"]][(layer, expert)]
            size = n * cfg.activation_bytes_per_assignment
            inbound = net.move(cfg.gateway_gpu, gpu, t, size, "dispatch", rid)
            begin = max(inbound, gpu_free[gpu])
            end = begin + n * cfg.compute_ms_per_assignment / cfg.gpu_speeds[gpu]
            gpu_free[gpu] = end
            push(end, "expert_computed", (rid, gpu, size))
            if gpu != cfg.gateway_gpu:
                totals["remote_expert_calls"] += 1
            if compute_log is not None:
                compute_log.append({"request_id": rid, "layer_id": layer, "expert_id": expert, "gpu_id": gpu,
                                    "assignments": n, "ready_ms": t, "inbound_ms": inbound,
                                    "start_ms": begin, "end_ms": end, "remote": gpu != cfg.gateway_gpu})

        elif kind == "expert_computed":
            rid, gpu, size = payload
            push(net.move(gpu, cfg.gateway_gpu, t, size, "return", rid), "expert_returned", rid)

        elif kind == "expert_returned":
            req = inflight[payload]
            req["outstanding"] -= 1
            req["max_finish"] = max(req["max_finish"], t)
            if req["outstanding"] == 0:
                enter_layer(payload, req["next_layer"], req["max_finish"])

        elif kind == "approve_migration":
            set_state(t, "approved", "controller approved migration")
            counts = [sum(v == g for v in old.values()) for g in range(n_gpus)]
            arriving = [sum(old[k] != g and new_gpu == g for k, new_gpu in target.items()) for g in range(n_gpus)]
            over_slots = [g for g in range(n_gpus) if counts[g] + arriving[g] > cfg.staging_slots]
            over_bytes = []
            if cfg.gpu_memory_gb is not None:
                budget = cfg.gpu_memory_gb * C.GB
                over_bytes = [g for g in range(n_gpus) if (counts[g] + arriving[g]) * cfg.expert_bytes > budget]
            if over_slots or over_bytes:
                failure_reason = (f"staging capacity exceeded on GPU(s) {sorted(set(over_slots + over_bytes))}: "
                                  "resident + incoming expert copies do not fit")
                set_state(t, "failed", failure_reason)
                continue
            copies = [(k, s, target[k]) for k, s in old.items() if s != target[k]]
            pending_copies = len(copies)
            totals["moved_experts"] = len(copies)
            set_state(t, "transferring", f"{len(copies)} expert copies queued")
            if not copies:
                active, migration_completed_at = dst_name, t
                set_state(t, "active", "no copies required")
            for key, s, d in sorted(copies):
                push(t, "copy_start", (key, s, d))

        elif kind == "copy_start":
            key, s, d = payload
            finish = net.move(s, d, t, cfg.expert_bytes, "migration", key)
            migration_records.append({"layer_id": key[0], "expert_id": key[1], "src_gpu": s, "dst_gpu": d,
                                      "start_requested_ms": t, "finish_ms": finish, "weight_bytes": cfg.expert_bytes})
            totals["migration_bytes"] += cfg.expert_bytes
            push(finish, "copy_done", key)

        elif kind == "copy_done":
            if state != "transferring":
                continue  # migration already failed; late copies are discarded
            if cfg.fail_after_copies is not None and completed_copies >= cfg.fail_after_copies:
                failure_reason = f"injected failure on copy #{completed_copies + 1} of {int(totals['moved_experts'])}"
                staged.clear()  # roll back: discard staged copies, keep serving the old layout
                set_state(t, "failed", failure_reason + "; rolled back to " + src_name)
                continue
            staged.add(payload)
            completed_copies += 1
            pending_copies -= 1
            if pending_copies == 0:
                active = dst_name
                migration_completed_at = t
                set_state(t, "active", f"all {completed_copies} copies staged; atomic switch to {dst_name}")
        else:
            raise RuntimeError(f"Unexpected event {kind}")

    if len(records) != len(request_ids):
        raise AssertionError("Not all requests completed")
    if state == "active" and len(staged) != totals["moved_experts"]:
        raise AssertionError("Activation before all copies arrived")

    req_df = pd.DataFrame(records).sort_values("request_id")
    summary = {
        "policy": "migrate" if allow_migration else "static",
        "migration_state": state,
        "migration_activation_ms": migration_completed_at,
        "moved_experts": int(totals["moved_experts"]),
        "migration_mb": totals["migration_bytes"] / 1e6,
        "total_link_mb": totals["communication_bytes"] / 1e6,
        "remote_expert_calls": int(totals["remote_expert_calls"]),
        "p50_latency_ms": float(np.percentile(req_df.latency_ms, 50)),
        "p95_latency_ms": float(np.percentile(req_df.latency_ms, 95)),
        "makespan_ms": float(req_df.finish_ms.max()),
        "failure_reason": failure_reason,
    }
    mig = pd.DataFrame(migration_records, columns=["layer_id", "expert_id", "src_gpu", "dst_gpu",
                                                   "start_requested_ms", "finish_ms", "weight_bytes"])
    result = SimResult(req_df, mig, summary, states)
    if record:
        result.compute = pd.DataFrame(compute_log)
        result.transfers = pd.DataFrame(net.log)
        result.layers = pd.DataFrame(layer_log)
    return result


# --------------------------------------------------------------------------
# Phase 4A: all requests at t=0, scheduled request-by-request (simulate_inference.py)
# --------------------------------------------------------------------------
def run_batch_at_zero(routing: pd.DataFrame, layout: Layout, cfg: C.SimConfig, request_ids=None) -> tuple[pd.DataFrame, dict]:
    request_ids = sorted(request_ids if request_ids is not None else cfg.test_ids)
    gpu_free = np.zeros(cfg.num_gpus)
    link_free = defaultdict(float)
    comm = 0
    remote = 0
    out = []

    def transfer(src, dst, ready, size):
        nonlocal comm
        if src == dst:
            return ready
        start = max(ready, link_free[(src, dst)])
        end = start + C.transfer_ms(size, cfg.bandwidth_gb_per_s, cfg.link_latency_ms)
        link_free[(src, dst)] = end
        comm += size
        return end

    for rid in request_ids:
        req = routing[(routing.request_id == rid) & (routing.token_assignments > 0)]
        ready = 0.0
        for layer_id in sorted(req.layer_id.unique()):
            finishes = []
            for r in req[req.layer_id == layer_id].itertuples():
                gpu = layout[(int(r.layer_id), int(r.expert_id))]
                n = int(r.token_assignments)
                size = n * cfg.activation_bytes_per_assignment
                if gpu != cfg.gateway_gpu:
                    remote += 1
                arrival = transfer(cfg.gateway_gpu, gpu, ready, size)
                end = max(arrival, gpu_free[gpu]) + n * cfg.compute_ms_per_assignment / cfg.gpu_speeds[gpu]
                gpu_free[gpu] = end
                finishes.append(transfer(gpu, cfg.gateway_gpu, end, size))
            if finishes:
                ready = max(finishes)
        out.append({"request_id": rid, "latency_ms": ready})
    df = pd.DataFrame(out)
    lat = df.latency_ms.to_numpy()
    return df, {"communication_mb": comm / 1e6, "remote_expert_calls": remote, "makespan_ms": float(lat.max()),
                "p50_latency_ms": float(np.percentile(lat, 50)), "p95_latency_ms": float(np.percentile(lat, 95))}


# --------------------------------------------------------------------------
# Phase 4B: migration copies only, starting at t=0 (simulate_migration.py)
# --------------------------------------------------------------------------
def migration_only(old: Layout, new: Layout, cfg: C.SimConfig) -> tuple[pd.DataFrame, dict]:
    link_free = defaultdict(float)
    events = []
    for key in sorted(old):
        s, d = old[key], new[key]
        if s == d:
            continue
        start = link_free[(s, d)]
        finish = start + C.transfer_ms(cfg.expert_bytes, cfg.bandwidth_gb_per_s, cfg.link_latency_ms)
        link_free[(s, d)] = finish
        events.append({"layer_id": key[0], "expert_id": key[1], "src_gpu": s, "dst_gpu": d,
                       "start_ms": start, "finish_ms": finish, "weight_bytes": cfg.expert_bytes})
    df = pd.DataFrame(events)
    return df, {"moved_experts": len(df), "transfer_mb": len(df) * cfg.expert_bytes / 1e6,
                "activation_ms": float(df.finish_ms.max()) if len(df) else 0.0}
