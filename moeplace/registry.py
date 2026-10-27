"""Provenance of every saved output file.

kind:
  measured    - recorded from real OLMoE forward passes on Apple MPS
  checkpoint  - read from safetensors metadata (no weights materialised)
  derived     - deterministic transformation of measured data (no hardware assumptions)
  simulated   - output of the simulator under HYPOTHETICAL hardware parameters

The parameter strings for simulated files were verified by regenerating each file
bit-for-bit with moeplace (tests/test_reproduction.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from .paths import OUTPUTS


@dataclass(frozen=True)
class Artifact:
    path: str
    kind: str
    phase: str
    producer: str
    parameters: str
    notes: str = ""

    @property
    def exists(self) -> bool:
        return (OUTPUTS / self.path).exists()


_SIM4 = "4 GPUs, speeds (1.0, 1.3, 0.8, 1.6), 0.5 ms/assignment, 4096 B/assignment, 0.1 ms link latency, gateway GPU 0"

ARTIFACTS: list[Artifact] = [
    Artifact("expert_routing.csv", "measured", "1", "scripts/profile_experts.py",
             "1 prompt, 8 input tokens, FP32 on MPS", "Single-prompt smoke test; not used downstream."),
    Artifact("workload_routing.csv", "measured", "2", "scripts/profile_workloads.py",
             "6 prompts (2 ai, 2 coding, 2 general), 6-10 input tokens each, FP32 eager on MPS, no chat template",
             "Top-8 routing per token per layer, aggregated per (request, layer, expert). Prefill only."),
    Artifact("workload_timings.csv", "measured", "2", "scripts/profile_workloads.py",
             "single forward pass per prompt, MPS synchronised",
             "Request 0 (51.9 s) includes first-call warm-up; no warm-up pass was run."),
    Artifact("expert_demand_by_category.csv", "derived", "2", "scripts/analyze_workloads.py",
             "sum over requests per category, divided by input tokens", ""),
    Artifact("gpu_placements.csv", "derived", "3A", "scripts/optimize_placement.py",
             "identical GPUs (1,1,1,1), 256 slots, demand = ALL six requests",
             "In-sample (no held-out split). balanced == demand_aware when GPUs are identical."),
    Artifact("placement_comparison.csv", "derived", "3A", "scripts/optimize_placement.py",
             "load proxy = token assignments / speed", "In-sample; load proxy, not latency."),
    Artifact("heterogeneous_placements.csv", "simulated", "3B", "scripts/evaluate_placement.py",
             "speeds (1.0, 1.3, 0.8, 1.6), 256 slots, placement from TRAIN ids 0,2,4",
             "Hypothetical heterogeneous speeds. test_demand column is for evaluation only."),
    Artifact("heldout_comparison.csv", "simulated", "3B", "scripts/evaluate_placement.py",
             "peak/mean normalized load on TEST ids 1,3,5", "Load proxy, not latency."),
    Artifact("simulated_inference_comparison.csv", "simulated", "4A", "scripts/simulate_inference.py",
             _SIM4 + ", 12 GB/s, all 3 test requests arrive at t=0", "Request-by-request reservation; no migration."),
    Artifact("simulated_request_latency.csv", "simulated", "4A", "scripts/simulate_inference.py", "as above", ""),
    Artifact("migration_events.csv", "simulated", "4B", "scripts/simulate_migration.py",
             "8 MB per expert (placeholder), 12 GB/s, static -> demand_aware, 776 copies", "Migration in isolation."),
    Artifact("migration_benefit.csv", "simulated", "4C", "scripts/analyze_migration_benefit.py",
             "4A makespans + 4B migration time, inference paused during migration", "Idealised amortisation."),
    Artifact("policy_comparison.csv", "simulated", "4E", "scripts/evaluate_migration_policy.py",
             _SIM4 + ", 8 MB per expert, 12 GB/s, arrivals every 200 ms, migration at 100 ms",
             "Generated before calibration. Re-running the script today would use 12.583 MB / 100 GB/s."),
    Artifact("policy_decision.csv", "simulated", "4E", "scripts/evaluate_migration_policy.py",
             "forecast on TRAIN ids 0,2,4; thresholds gain >= 2%, p95 regression <= 0%", "Decision: reject."),
    Artifact("policy_request_latency.csv", "simulated", "4E", "scripts/evaluate_migration_policy.py", "as policy_comparison", ""),
    Artifact("integrated_comparison.csv", "simulated", "4D/4F", "scripts/simulate_integrated.py (last run by run_calibrated_sweep.py)",
             _SIM4 + ", 12.583 MB per expert, 100 GB/s",
             "Overwritten by every sweep iteration; identical to phase4f/calibrated/bandwidth_100.csv."),
    Artifact("integrated_request_latency.csv", "simulated", "4D/4F", "scripts/simulate_integrated.py", "as integrated_comparison", ""),
    Artifact("integrated_migration_events.csv", "simulated", "4D/4F", "scripts/simulate_integrated.py", "as integrated_comparison", ""),
    Artifact("phase4f/memory/expert_tensor_metadata.csv", "checkpoint", "4F", "scripts/inspect_expert_memory.py",
             "safetensors headers only", "3072 tensors = 16 layers x 64 experts x {gate,up,down}_proj, BF16."),
    Artifact("phase4f/memory/expert_memory_summary.csv", "checkpoint", "4F", "scripts/inspect_expert_memory.py",
             "per-layer sums", "12.885 GB total; 12.582912 MB per expert."),
    Artifact("phase4f/network/network_results.csv", "simulated", "4F", "scripts/analyze_network_sensitivity.py",
             _SIM4 + ", 8 MB per expert, {3, 12, 25, 100} GB/s", "Aggregates phase4f/network/bandwidth_*.csv."),
    Artifact("phase4f/network/network_improvement.csv", "simulated", "4F", "scripts/analyze_network_sensitivity.py",
             "as network_results", ""),
    Artifact("phase4f/network/requests_3.csv", "simulated", "4F", "manual copy of integrated_request_latency.csv",
             "8 MB, 3 GB/s", ""),
    Artifact("phase4f/network/migration_3.csv", "simulated", "4F", "manual copy of integrated_migration_events.csv",
             "8 MB, 3 GB/s", ""),
] + [
    Artifact(f"phase4f/{kind}/bandwidth_{bw}.csv", "simulated", "4F",
             "scripts/simulate_integrated.py" + (" via run_calibrated_sweep.py" if kind == "calibrated" else " (constants edited by hand)"),
             _SIM4 + f", {'8 MB' if kind == 'network' else '12.583 MB'} per expert, {bw} GB/s", "")
    for kind in ("network", "calibrated") for bw in (3, 12, 25, 100)
]

BY_PATH = {a.path: a for a in ARTIFACTS}

KIND_LABEL = {
    "measured": "Measured on OLMoE",
    "checkpoint": "Checkpoint metadata",
    "derived": "Derived from measurements",
    "simulated": "Simulated (hypothetical hardware)",
}
