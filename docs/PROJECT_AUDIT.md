# Project audit — MoE Expert Placement Planner

Audit date: 2026-10-22. Scope: every file in the repository except the model weights and `.venv`.
Every claim below was checked by reading the code, inspecting the CSVs, or re-running the computation.
A pre-change snapshot of `scripts/`, `dashboard/` and `outputs/` is at
`../moe-expert-placement_backup_20261022/`.

## 1. Architecture as found

```
models/OLMoE-1B-7B-0125-Instruct/        13 GB checkpoint (3 BF16 safetensors shards)
scripts/                                  17 standalone scripts, relative paths ("outputs/…"), no shared code
outputs/                                  23 CSVs at top level + outputs/phase4f/{memory,network,calibrated}
dashboard/app.py                          one 220-line page (Phase 1 – 4A only)
```

There was no git repository, no tests, no shared configuration, no README and no requirements file.
(As found on 2026-10-22. A local git repository and a CI workflow were added on 2026-10-27; see section 7.)

### Data flow (verified)

```
profile_experts.py      → expert_routing.csv                  (Phase 1, 1 prompt; not used downstream)
profile_workloads.py    → workload_routing.csv, workload_timings.csv            MEASURED
analyze_workloads.py    → expert_demand_by_category.csv                          derived
optimize_placement.py   → gpu_placements.csv, placement_comparison.csv          3A: all 6 requests, identical GPUs
evaluate_placement.py   → heterogeneous_placements.csv, heldout_comparison.csv  3B: train 0,2,4 / test 1,3,5
simulate_inference.py   ← heterogeneous_placements → simulated_*.csv            4A
simulate_migration.py   ← heterogeneous_placements → migration_events.csv       4B
analyze_migration_benefit.py ← 4A + 4B → migration_benefit.csv                  4C
simulate_integrated.py  ← routing + heterogeneous_placements → integrated_*.csv 4D
evaluate_migration_policy.py (imports simulate_integrated) → policy_*.csv       4E
inspect_expert_memory.py ← safetensors headers → phase4f/memory/*.csv           4F
run_calibrated_sweep.py (rewrites simulate_integrated.py source!) → phase4f/calibrated/*.csv
analyze_network_sensitivity.py ← phase4f/network/bandwidth_*.csv → network_*.csv
```

## 2. Implemented components (verified working)

| Component | Status | Evidence |
|---|---|---|
| OLMoE loading + MPS inference | Present | `test_inference.py`, `profile_*.py` (not re-run: expensive) |
| Real routing profiling, 6 prompts | Present | `workload_routing.csv`: 6 × 1024 rows; every (request, layer) sums to tokens × 8 |
| Static / balanced / demand-aware placement | Present | Re-implemented and reproduced exactly |
| Heterogeneous speeds + held-out split | Present | No test demand used for placement (verified) |
| 4A batch simulator, 4B migration, 4C amortisation | Present | Reproduced exactly |
| 4D event simulator: layer barriers, FIFO GPU/link queues, pinning, atomic activation | Present | Reproduced exactly; invariants tested |
| 4E cost-aware policy, train-only forecast | Present | Reproduced exactly |
| 4F checkpoint-calibrated expert size, bandwidth sweeps | Present | 12,582,912 B/expert = 3 × 2048 × 1024 × 2 B (BF16); sweeps reproduced exactly |

## 3. Functional issues

| # | Issue | Priority | Status |
|---|---|---|---|
| F1 | **Re-running scripts silently overwrites original results.** E.g. `evaluate_migration_policy.py` today would replace the saved 8 MB / 12 GB/s policy results with a 12.583 MB / 100 GB/s scenario (verified on a scratch copy: different p95, forecast p95 regression 6.50% vs 32.93%). `profile_workloads.py` would overwrite real measurements. | **High** | Fixed: all writing scripts refuse to overwrite without `--force` (`moeplace/legacy.py`) |
| F2 | `run_calibrated_sweep.py` edits the source of `simulate_integrated.py` with regexes, and each iteration overwrites `outputs/integrated_*.csv`. Root `integrated_*.csv` therefore hold the 100 GB/s calibrated run, not a standalone 4D run. | High | Documented in the provenance registry; superseded by `scripts/run_experiment.py` (no source rewriting). Script kept and still works (`--force` passed to its subprocess). |
| F3 | All scripts use cwd-relative paths; they fail unless run from the project root. | Medium | Fixed: scripts `chdir` to the project root; new code uses `moeplace/paths.py` |
| F4 | Configuration duplicated in 7 scripts (speeds, bandwidth, sizes, ids) with drift (8 MB vs 12.58 MB; 3/12/100 GB/s). | High | Fixed: `moeplace/config.py` with named presets for each saved experiment |
| F5 | Stale comment: `EXPERT_BYTES = 12_582_912  # hypothetical; not checkpoint-calibrated` — the value *is* checkpoint-derived. | Low | Fixed |
| F6 | Three duplicate snapshot scripts (`simulate_integrated_backup.py` = 3 GB/s/8 MB, `simulate_integrated_pre_calibration.py` = 12 GB/s/8 MB, `evaluate_migration_policy_backup.py` = identical copy). | Low | Kept (not deleted); guarded; their parameters are now presets |
| F7 | `phase4f/network/*.csv` were produced by hand-editing constants; `requests_3.csv`/`migration_3.csv` are manual copies. No metadata recorded parameters. | Medium | Parameters recovered by exact regeneration and recorded in `moeplace/registry.py` |
| F8 | Dashboard used deprecated `use_container_width`, cwd-relative paths and `assert` for validation; covered only Phases 1–4A; labelled single-prompt data as the main view. | Medium | Replaced by the 11-page dashboard; original kept as `dashboard/legacy_app.py` with the deprecations fixed |
| F9 | No tests. | High | 90 tests added (`tests/`) |
| F10 | `pytest` is not installed in `.venv`. | Low | Tests use stdlib `unittest` (pytest-compatible); environment unchanged |
| F11 | Validation by bare `assert` (disabled under `python -O`), no informative errors. | Medium | New code raises `ConfigError` / `SchemaError` / `FileNotFoundError` with hints |

## 4. Scientific and evaluation review

| # | Question | Finding | Priority |
|---|---|---|---|
| S1 | Sample size | 6 prompts of 6–10 tokens (44 tokens total). 15% of the 1,024 experts get zero assignments over all six prompts, 24% in the train split and 27% in the test split, so many demand estimates rest on 0–2 observations. p95 over 3 requests is ≈ the max (linear interpolation). No statistical claim is possible. | **High** (documented; `profile_routing.py` added for more prompts) |
| S2 | Held-out set | 3 requests, one per category. Splits are disjoint (verified, enforced by `SimConfig.validate`). | High |
| S3 | Synthetic replay vs new measurements | All latency results replay the 6 measured traces; none come from new inference. | Medium (documented) |
| S4 | Train/test leakage | **None** in 3B placement or the 4E controller (tests corrupt held-out routing and show placements/decision unchanged). **3A** (`gpu_placements.csv`) is in-sample by design and is labelled so. The controller's forecast reuses the training requests the placement was fitted on (in-sample forecast → optimistic), which is legitimate but should be stated. | Medium |
| S5 | Load proxy vs latency | Phase 3 metrics are token counts ÷ hypothetical speed. They were never labelled as utilisation in code, but the old dashboard plotted them as "GPU Load". Now labelled "load proxy" everywhere. | Medium |
| S6 | Layer dependencies | Enforced: a request's layer L+1 is queued only when every expert of layer L has returned (tested). | OK |
| S7 | Network contention | One independent FIFO per directed GPU pair; no shared NIC, switch or PCIe/NVLink topology. Added an opt-in `shared_port` model as a sensitivity check (not calibrated). | Medium |
| S8 | Requests keep placement active at arrival | Yes (tested, including compute-GPU checks against the pinned layout). | OK |
| S9 | Activation after all copies | Yes; activation time = last copy finish (tested). | OK |
| S10 | Memory includes resident + staged | Yes, as slot counts (resident + incoming ≤ 512). Added an optional byte budget. Old copies are never freed after activation, and non-expert weights, activations and KV cache are not modelled. | Medium |
| S11 | Inference and migration share network | Yes, same link queues (tested: request 1 latency 2.4× higher during 3 GB/s migration). All 776 copies are reserved at approval, so inference queues behind the whole migration on shared links. | OK (documented) |
| S12 | Failure / rollback | Original: only a capacity pre-check. Added fault injection (`fail_after_copies`) with rollback to the old layout; in-flight requests unaffected (tested). Link time already reserved by aborted copies is not reclaimed (conservative). | Medium |
| S13 | Scheduler semantics | Resources are reserved in event-processing order, so a GPU cannot back-fill an idle gap with a later-processed task (non-work-conserving). Deterministic and reproducible, but pessimistic under load. | Low (documented) |
| S14 | Controller conservatism | With a 0% p95 limit and 3 training requests, the controller rejects migration at **every** tested bandwidth (3–100 GB/s), even where held-out p95 improves 12% (100 GB/s). The forecast is dominated by the one request in flight during copying. | **High** (finding; see methodology) |
| S15 | Units | `BANDWIDTH_GBPS` was used as giga**bytes**/s (bytes per ms = value × 1e6). Consistent across scripts, but the name invites Gbps confusion. New code uses `bandwidth_gb_per_s`; a unit test pins 1 GB at 1 GB/s = 1000 ms. | Medium |
| S16 | Timing measurements | Request 0's 51.9 s forward pass includes first-call MPS warm-up (others 12–22 s). FP32, eager attention. Not comparable to GPU serving. | Low (documented; new script does warm-up) |
| S17 | Precision | The checkpoint is BF16 (12.583 MB/expert), but the profilers load FP32 (25.2 MB/expert in memory). Migration sizes assume BF16 serving. | Low (documented) |

## 5. Performance bottlenecks

| Item | Observation | Action |
|---|---|---|
| OLMoE forward pass on MPS | 12–52 s per 6–10-token prompt, FP32 eager | Not changed (out of scope); `profile_routing.py --dtype bfloat16` is available |
| Old dashboard | No caching; re-read CSVs on every interaction | `st.cache_data` on every loader and simulation |
| Simulation cost | Full policy evaluation (4 simulations) ≈ 45 ms; all 20 saved outputs regenerate in 0.17 s | Interactive simulation in the dashboard is practical |
| Model loading in UI | Never happened, and still never happens (test asserts `torch` is not imported) | — |

## 6. UI/UX problems (old dashboard)

Single long page; no navigation; single-prompt Phase 1 data shown first as the main view; no measured/simulated labelling except
two captions; Viridis heatmap on a light default theme; no interaction beyond two selectboxes; no Phase 4B–4F content;
deprecated parameters. All are addressed by the new dashboard (see `docs/ARCHITECTURE.md`).

## 7. Recommended modifications (status)

| Priority | Recommendation | Status |
|---|---|---|
| High | Protect original results from overwrite | Done |
| High | Centralised config + presets reproducing saved runs | Done |
| High | Reproducible runner with run ids, metadata and sweeps | Done (`scripts/run_experiment.py`) |
| High | Tests for reproduction, invariants, leakage, schemas, dashboard | Done (175) |
| High | Version control | Done: local git repository (branch `main`). Weights, tokenizer and `.venv` excluded by `.gitignore`. **No remote configured, nothing pushed.** |
| High | Continuous integration | Workflow written (`.github/workflows/ci.yml`: Python 3.11, `requirements-ci.txt`, full unittest suite). Verified by running the same steps locally in a clean clone with a fresh venv and no weights (175 tests OK). **Not yet run on GitHub Actions**, which needs a GitHub remote. |
| High | More real prompts (≥ 100), decode routing | Tooling done (`profile_routing.py`); **not run** (expensive, explicit action) |
| High | Replicates with seeded Poisson arrivals + confidence intervals | Tooling done; results not yet produced |
| Medium | Robust controller forecast (more requests, rolling window, expected-cost) | Open |
| Medium | Topology-aware network model, calibrated on real GPUs | Open (`shared_port` sensitivity added) |
| Medium | Incremental migration (per layer) to cap contention | Open |
| Low | Free old copies after activation; model non-expert memory | Open |
