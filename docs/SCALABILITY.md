# Planner scalability

Code: `moeplace/scalability.py`. Run: `scripts/run_scalability.py` → `outputs/runs/scalability-<id>/`
(`scalability.csv`, `growth_fits.csv`, `optimality_gap.csv`, `metadata.json`). Tests: `tests/test_agent_usecases.py::TestScalability`.

## 1. Two metric families, kept separate

**Planner computational performance (measured, wall clock):**

| Metric | Formula |
|---|---|
| T_plan | median over r = 5 runs of the wall time of `plan_sites()` (`time.perf_counter`), per seed |
| T_plan,p95 | 95th percentile of the same runs |
| C | number of (expert, device) candidate evaluations the planner performed |
| T per 1k candidates | T_plan / (C / 1000) |
| Decision budget | T_plan must be ≤ 2 s at the proposal size and ≤ 10% of the planning window (90 s) at the largest size |

**Predicted inference performance of the resulting plan (analytic model, hypothetical hardware):**

| Metric | Formula |
|---|---|
| J | max( max_g load_g / (speed_g · a · W), max_{s,s'} bytes(s,s') / (bw · W) ) |
| L | Σ_s V_s · 16 · RTT · (1 − (1 − r_s)^8) / Σ_s V_s, added cross-site latency per token |
| remote share | assignments served in another site / all assignments |
| residency violations | assignments from a residency site served elsewhere (must be 0) |

## 2. Baseline assumptions and workload conditions

* CostModel defaults: window W = 900 s; a = 5×10⁴ assignments/s on a speed-1.0 device; 4,096 B per assignment
  per direction; inter-site 1.25 GB/s per direction and 60 ms RTT; device speeds cycle 1.0, 1.3, 0.8, 1.6.
* Demand per window = 50% of total cluster compute capacity. Expert popularity is resampled from the measured
  OLMoE per-expert distribution with lognormal noise; each site perturbs it (σ = 0.3); site shares ~ Dirichlet(5).
* Locality mode with a 0.7 utilisation cap. Slots per device = ⌈1.3 × copies / devices⌉, raised so a residency
  site can hold every expert. Memory budget = slots × 12.583 MB when the memory constraint is on.
* Base point: 1,024 experts, 3 sites × 4 devices, replica budget 0.25 × E, constraints capacity + memory + residency.
  One factor varies at a time. 3 seeds × 5 repeats.

## 3. Results (`scalability-75c1324c80`; Linux aarch64 VM on the author's Mac)

| Factor | Range | Median T_plan | Fitted growth T ∝ x^k | R² |
|---|---|---|---|---|
| Experts | 16 → 16,384 | 0.43 → 272 ms | k = 0.94 | 1.00 |
| Sites | 1 → 8 (4 devices each) | 6.2 → 42.5 ms | k = 0.92 | 0.99 |
| Devices | 3 → 96 (3 sites) | 16.4 → 18.3 ms | k = 0.03 | 0.65 |
| Replica budget | 0.1 → 2.0 × E | 15.3 → 27.2 ms | k = 0.22 | 0.95 |
| Constraints | none → capacity+memory+residency | 16.3 → 16.3 ms | — | — |
| **Proposal point** | 16 experts, 3 sites, 1 device each | **0.34 ms** (target 2 s) | — | — |

Plan quality at the base point (simulated): J ≈ 0.65, L ≈ 730 ms. With a replica budget of 1.0 × E, L falls to
128 ms and with 2.0 × E to 66 ms, while J stays at the 0.7 cap. Residency adds 1,024 copies at the residency site
and lowers L from 932 ms to 731 ms in this configuration. At 96 devices J exceeds 1 because slots per device
become small relative to the residency site's need to hold every expert (a capacity, not a planner-speed, limit).

**Interpretation.** The planner is O(E log E + E·(G + S²) + R·(G + S²)). Wall time follows E and S almost
linearly. G is nearly free because each expert's device choice is a vectorised NumPy operation; Python-loop
overhead dominates until G is in the hundreds.

**Optimality.** On 20 random tiny instances (7 experts, 3 heterogeneous devices, capacity-constrained) the
greedy planner matched the exhaustive optimum of J every time (`optimality_gap.csv`). Exhaustive search is
infeasible at realistic sizes, so this only shows the heuristic is not obviously wrong. An ILP bound at 1,024
experts is the open item.

**Not included.** Event-simulator time (used inside the single-cluster migration controller: about 45 ms per full
policy evaluation, PROJECT_AUDIT.md §5) is reported separately and is not part of T_plan.
