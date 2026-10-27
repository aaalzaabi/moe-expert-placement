"""Phase 4D: integrated, deterministic MoE inference + migration simulator.

Input: outputs/workload_routing.csv and outputs/heterogeneous_placements.csv
Assumptions: requests arrive every ARRIVAL_GAP_MS, layer-level aggregate expert
assignments, gateway GPU 0, per-directed-link serial transfers, one compute
queue per GPU, migration approval at MIGRATION_START_MS. No real GPU measurements.
"""
from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / 'outputs'  # robust to cwd
ROUTING = OUT / 'workload_routing.csv'
PLACEMENTS = OUT / 'heterogeneous_placements.csv'

GPU_SPEEDS = (1.0, 1.3, 0.8, 1.6)
GATEWAY = 0
COMPUTE_MS_PER_ASSIGNMENT = 0.5
ACTIVATION_BYTES_PER_ASSIGNMENT = 4096
EXPERT_BYTES = 12_582_912  # checkpoint-derived: 3 x 2048 x 1024 BF16 values (see phase4f/memory)
BANDWIDTH_GBPS = 100.0     # hypothetical GB/s per directed link
LINK_LATENCY_MS = 0.1
ARRIVAL_GAP_MS = 200.0
MIGRATION_START_MS = 100.0
CAPACITY_SLOTS = 512       # hypothetical temporary staging capacity
TEST_IDS = (1, 3, 5)


def load_data():
    routing = pd.read_csv(ROUTING)
    placement = pd.read_csv(PLACEMENTS)
    require_r = {'request_id', 'layer_id', 'expert_id', 'token_assignments'}
    require_p = {'strategy', 'layer_id', 'expert_id', 'gpu_id'}
    if not require_r.issubset(routing) or not require_p.issubset(placement):
        raise ValueError('Missing required input columns')
    layouts = {}
    for name in ('static', 'demand_aware'):
        frame = placement[placement.strategy == name]
        if frame.duplicated(['layer_id', 'expert_id']).any():
            raise ValueError(f'Duplicate expert in {name} placement')
        layouts[name] = {(int(r.layer_id), int(r.expert_id)): int(r.gpu_id)
                         for r in frame.itertuples()}
    if layouts['static'].keys() != layouts['demand_aware'].keys():
        raise ValueError('Layouts have different expert inventories')
    if not set(layouts['static'].values()) <= set(range(len(GPU_SPEEDS))):
        raise ValueError('Unexpected GPU id')
    if not set(layouts['demand_aware'].values()) <= set(range(len(GPU_SPEEDS))):
        raise ValueError('Unexpected GPU id')
    traces = {}
    for rid in TEST_IDS:
        req = routing[(routing.request_id == rid) & (routing.token_assignments > 0)]
        if req.empty:
            raise ValueError(f'No nonzero routing for request {rid}')
        traces[rid] = {
            int(layer): [(int(r.expert_id), int(r.token_assignments))
                         for r in part.itertuples()]
            for layer, part in req.groupby('layer_id', sort=True)
        }
    return traces, layouts


def run(traces, layouts, allow_migration=True):
    """Deterministic event simulation; time expressed in simulated milliseconds.

    At migration approval, copies are queued as events, not pre-reserved.
    Each new arrival pins the then-active complete layout for its lifetime.
    """
    old, target = layouts['static'], layouts['demand_aware']
    serial = itertools.count()
    events = []
    gpu_free = [0.0] * len(GPU_SPEEDS)
    link_free = defaultdict(float)
    inflight = {}
    records = []
    migration_records = []
    totals = defaultdict(float)
    state = 'proposed' if allow_migration else 'disabled'
    active = 'static'
    pending_copies = 0
    staged = set()
    migration_completed_at = None

    def push(t, kind, payload):
        heapq.heappush(events, (float(t), next(serial), kind, payload))

    def move(src, dst, at, size):
        if src == dst:
            return at
        start = max(float(at), link_free[(src, dst)])
        end = start + LINK_LATENCY_MS + size / (BANDWIDTH_GBPS * 1e6)
        link_free[(src, dst)] = end
        totals['communication_bytes'] += size
        return end

    def enter_layer(rid, layer_index, at):
        req = inflight[rid]
        layers = sorted(traces[rid])
        if layer_index == len(layers):
            records.append({'request_id': rid, 'arrival_ms': req['arrival'],
                            'finish_ms': at, 'latency_ms': at - req['arrival'],
                            'layout': req['layout']})
            return
        layer = layers[layer_index]
        choices = traces[rid][layer]
        req['outstanding'] = len(choices)
        req['max_finish'] = at
        req['next_layer'] = layer_index + 1
        for expert, n in choices:
            push(at, 'expert_start', (rid, layer, expert, n))

    for i, rid in enumerate(TEST_IDS):
        push(i * ARRIVAL_GAP_MS, 'request_arrival', rid)
    if allow_migration:
        push(MIGRATION_START_MS, 'approve_migration', None)

    while events:
        t, _, kind, payload = heapq.heappop(events)
        if kind == 'request_arrival':
            rid = payload
            inflight[rid] = {'arrival': t, 'layout': active}
            enter_layer(rid, 0, t)

        elif kind == 'expert_start':
            rid, layer, expert, n = payload
            layout = layouts[inflight[rid]['layout']]
            gpu = layout[(layer, expert)]
            size = n * ACTIVATION_BYTES_PER_ASSIGNMENT
            inbound = move(GATEWAY, gpu, t, size)
            begin = max(inbound, gpu_free[gpu])
            end = begin + n * COMPUTE_MS_PER_ASSIGNMENT / GPU_SPEEDS[gpu]
            gpu_free[gpu] = end
            push(end, 'expert_computed', (rid, gpu, size))
            if gpu != GATEWAY:
                totals['remote_expert_calls'] += 1

        elif kind == 'expert_computed':
            rid, gpu, size = payload
            returned = move(gpu, GATEWAY, t, size)
            push(returned, 'expert_returned', rid)

        elif kind == 'expert_returned':
            rid = payload
            req = inflight[rid]
            req['outstanding'] -= 1
            req['max_finish'] = max(req['max_finish'], t)
            if req['outstanding'] == 0:
                enter_layer(rid, req['next_layer'], req['max_finish'])

        elif kind == 'approve_migration':
            counts = [sum(v == gpu for v in old.values()) for gpu in range(len(GPU_SPEEDS))]
            arriving = [sum(old[key] != gpu and new_gpu == gpu for key, new_gpu in target.items())
                        for gpu in range(len(GPU_SPEEDS))]
            if any(counts[g] + arriving[g] > CAPACITY_SLOTS for g in range(len(GPU_SPEEDS))):
                state = 'failed_capacity'
                continue
            copies = [(key, src, target[key]) for key, src in old.items() if src != target[key]]
            pending_copies = len(copies)
            state = 'transferring'
            totals['moved_experts'] = len(copies)
            if not copies:
                active, state, migration_completed_at = 'demand_aware', 'active', t
            for key, src, dst in sorted(copies):
                push(t, 'copy_start', (key, src, dst))

        elif kind == 'copy_start':
            key, src, dst = payload
            finish = move(src, dst, t, EXPERT_BYTES)
            migration_records.append({
                'layer_id': key[0], 'expert_id': key[1], 'src_gpu': src,
                'dst_gpu': dst, 'start_requested_ms': t, 'finish_ms': finish,
                'weight_bytes': EXPERT_BYTES,
            })
            totals['migration_bytes'] += EXPERT_BYTES
            push(finish, 'copy_done', key)

        elif kind == 'copy_done':
            staged.add(payload)
            pending_copies -= 1
            if pending_copies == 0:
                active = 'demand_aware'
                state = 'active'
                migration_completed_at = t

        else:
            raise RuntimeError(f'Unexpected event {kind}')

    if len(records) != len(TEST_IDS):
        raise AssertionError('Not all requests completed')
    if state == 'active' and len(staged) != totals['moved_experts']:
        raise AssertionError('Activation before all copies arrived')

    req_df = pd.DataFrame(records).sort_values('request_id')
    summary = {
        'policy': 'migrate' if allow_migration else 'static',
        'migration_state': state,
        'migration_activation_ms': migration_completed_at,
        'moved_experts': int(totals['moved_experts']),
        'migration_mb': totals['migration_bytes'] / 1e6,
        'total_link_mb': totals['communication_bytes'] / 1e6,
        'remote_expert_calls': int(totals['remote_expert_calls']),
        'p50_latency_ms': float(np.percentile(req_df.latency_ms, 50)),
        'p95_latency_ms': float(np.percentile(req_df.latency_ms, 95)),
        'makespan_ms': float(req_df.finish_ms.max()),
    }
    return req_df, pd.DataFrame(migration_records), summary


def main():
    import sys
    sys.path.insert(0, str(OUT.parent))
    from moeplace.legacy import prepare
    prepare(['outputs/integrated_request_latency.csv', 'outputs/integrated_comparison.csv',
             'outputs/integrated_migration_events.csv'])
    traces, layouts = load_data()
    outputs = []
    request_frames = []
    mig_events = None
    for migrate in (False, True):
        requests, events, summary = run(traces, layouts, allow_migration=migrate)
        request_frames.append(requests.assign(policy=summary['policy']))
        outputs.append(summary)
        if migrate:
            mig_events = events
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(request_frames).to_csv(OUT / 'integrated_request_latency.csv', index=False)
    pd.DataFrame(outputs).to_csv(OUT / 'integrated_comparison.csv', index=False)
    mig_events.to_csv(OUT / 'integrated_migration_events.csv', index=False)
    print('\nPHASE 4D — INTEGRATED SIMULATION (hypothetical hardware)')
    print(pd.DataFrame(outputs).to_string(index=False))
    print('\nSaved integrated_request_latency.csv, integrated_comparison.csv, integrated_migration_events.csv')
    print('Note: requests pin their placement at arrival; migration transfers use shared directed-link queues.')


if __name__ == '__main__':
    main()
