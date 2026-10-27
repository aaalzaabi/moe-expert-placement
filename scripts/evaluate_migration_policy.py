"""Phase 4E: train-only, cost-aware policy for the Phase 4D simulator.

Put this script beside simulate_integrated.py in scripts/. No new packages.
The controller chooses using training requests (0, 2, 4) ONLY, then evaluates
on the separate test requests (1, 3, 5). Results are hypothetical simulation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import simulate_integrated as sim

TRAIN_IDS = (0, 2, 4)
TEST_IDS = (1, 3, 5)


def get_traces(ids):
    data = pd.read_csv(sim.ROUTING)
    traces = {}
    for rid in ids:
        rows = data[(data.request_id == rid) & (data.token_assignments > 0)]
        if rows.empty:
            raise ValueError(f"Missing routing data for request {rid}")
        traces[rid] = {
            int(layer): [(int(r.expert_id), int(r.token_assignments))
                         for r in frame.itertuples()]
            for layer, frame in rows.groupby('layer_id', sort=True)
        }
    return traces


def evaluate(ids, layouts, migration):
    # Phase 4D uses TEST_IDS for scheduling and assertion; change only here.
    previous = sim.TEST_IDS
    try:
        sim.TEST_IDS = tuple(ids)
        return sim.run(get_traces(ids), layouts, allow_migration=migration)
    finally:
        sim.TEST_IDS = previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-makespan-gain-pct', type=float, default=2.0,
                        help='Minimum improvement forecast to approve migration')
    parser.add_argument('--max-p95-regression-pct', type=float, default=0.0,
                        help='Maximum allowed forecast p95 increase')
    parser.add_argument('--force', action='store_true', help='overwrite existing policy_*.csv files')
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(sim.OUT.parent))
    from moeplace.legacy import prepare
    prepare(['outputs/policy_comparison.csv', 'outputs/policy_request_latency.csv',
             'outputs/policy_decision.csv'])

    # Validate inputs and load layouts, but never use TEST requests for policy selection.
    _, layouts = sim.load_data()
    t_static, _, forecast_static = evaluate(TRAIN_IDS, layouts, False)
    t_migrate, _, forecast_migrate = evaluate(TRAIN_IDS, layouts, True)

    makespan_gain_pct = 100 * (forecast_static['makespan_ms'] - forecast_migrate['makespan_ms']) / forecast_static['makespan_ms']
    p95_regression_pct = 100 * (forecast_migrate['p95_latency_ms'] - forecast_static['p95_latency_ms']) / forecast_static['p95_latency_ms']
    migration_feasible = forecast_migrate['migration_state'] == 'active'
    approve = (migration_feasible
               and makespan_gain_pct >= args.min_makespan_gain_pct
               and p95_regression_pct <= args.max_p95_regression_pct)
    decision = 'approve' if approve else 'reject'

    # Holdout evaluation: do not use outcomes here to change the decision.
    baseline_requests, _, static = evaluate(TEST_IDS, layouts, False)
    always_requests, migration_events, always = evaluate(TEST_IDS, layouts, True)
    policies = [
        {'policy': 'never_migrate', **{k: v for k, v in static.items() if k != 'policy'}},
        {'policy': 'always_migrate', **{k: v for k, v in always.items() if k != 'policy'}},
        {'policy': 'cost_aware', **{k: v for k, v in (always if approve else static).items() if k != 'policy'}},
    ]
    requests = []
    for name, frame in [('never_migrate', baseline_requests),
                        ('always_migrate', always_requests),
                        ('cost_aware', always_requests if approve else baseline_requests)]:
        requests.append(frame.assign(policy=name))

    out = sim.OUT
    out.mkdir(exist_ok=True)
    pd.DataFrame(policies).to_csv(out / 'policy_comparison.csv', index=False)
    pd.concat(requests, ignore_index=True).to_csv(out / 'policy_request_latency.csv', index=False)
    pd.DataFrame([{
        'decision': decision,
        'forecast_request_ids': ','.join(map(str, TRAIN_IDS)),
        'evaluation_request_ids': ','.join(map(str, TEST_IDS)),
        'forecast_makespan_gain_pct': makespan_gain_pct,
        'forecast_p95_regression_pct': p95_regression_pct,
        'min_makespan_gain_pct': args.min_makespan_gain_pct,
        'max_p95_regression_pct': args.max_p95_regression_pct,
        'migration_feasible': migration_feasible,
    }]).to_csv(out / 'policy_decision.csv', index=False)
    print('\nPHASE 4E — COST-AWARE MIGRATION POLICY')
    print(f'Train-only forecast: makespan gain {makespan_gain_pct:.2f}%, p95 change {p95_regression_pct:+.2f}%')
    print(f'Cost-aware decision: {decision.upper()} (computed without held-out requests)')
    print('\nHELD-OUT COMPARISON (simulated):')
    print(pd.DataFrame(policies)[['policy', 'migration_state', 'p50_latency_ms', 'p95_latency_ms', 'makespan_ms', 'migration_mb']].to_string(index=False))
    print('\nSaved outputs/policy_comparison.csv, outputs/policy_request_latency.csv, outputs/policy_decision.csv')
    print('Caveat: 3 train and 3 test prompts; hypothetical speed, network, and expert size.')


if __name__ == '__main__':
    main()
