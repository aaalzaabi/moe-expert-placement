# MoE Expert Placement Planner

Demand-aware GPU expert placement and cost-aware migration planning for distributed Mixture-of-Experts
inference, built on **real routing measurements** from `allenai/OLMoE-1B-7B-0125-Instruct`
(16 MoE layers × 64 experts, top-8 routing → 1,024 layer-specific experts).

> **What is real and what is simulated.** Expert routing and forward-pass timings are measured on the real
> model (Apple M4 Max, MPS). Expert weight sizes come from the checkpoint (12.583 MB per expert, BF16).
> Everything about GPUs — speeds, compute cost, network bandwidth, queues, migration — is a **simulation with
> hypothetical parameters**. No result in this project is a measured multi-GPU speed-up.

## Quick start

```bash
cd ~/Desktop/MOE/moe-expert-placement

# Dashboard (never loads the model)
.venv/bin/streamlit run dashboard/app.py

# Test suite (stdlib unittest; ~5 s)
.venv/bin/python -m unittest discover -s tests -t .

# Multi-site planner, forecasting, regret, agent and use cases (feedback response; ~2 min)
.venv/bin/python scripts/run_usecases.py                 # six use cases -> outputs/runs/usecases-<id>/
.venv/bin/python scripts/run_scalability.py              # planner scaling -> outputs/runs/scalability-<id>/

# Simulation experiments (write to outputs/runs/, never to the original files)
.venv/bin/python scripts/run_experiment.py                                  # default scenario
.venv/bin/python scripts/run_experiment.py --preset phase4e_legacy          # regenerate saved Phase 4E
.venv/bin/python scripts/run_experiment.py --bandwidth 3 6 12 25 50 100     # sweep
.venv/bin/python scripts/run_experiment.py --list

# New real measurements (explicit, expensive: loads the 13 GB checkpoint)
.venv/bin/python scripts/profile_routing.py --dry-run
.venv/bin/python scripts/profile_routing.py --prompt-set extended --warmup 1
```

## Repository layout

```
moeplace/              shared core (no torch): config, data, placement, simulator, policy, experiment, registry,
                       multisite (sites/replicas/constraints planner), workload (site-aware demand tensor),
                       forecast (forecasters, regret, drift), usecases, agent, scalability
scripts/               original phase scripts (guarded) + run_experiment.py + profile_routing.py
dashboard/app.py       12-page Streamlit control center (dashboard/app_pages/*.py, dashboard/ui.py,
                       dashboard/ms_views.py = view models of the Multi-site page)
dashboard/legacy_app.py  original single-page dashboard (deprecations fixed)
outputs/               ORIGINAL results (read-only by convention; scripts refuse to overwrite without --force)
outputs/runs/          new simulation runs, one folder per run id with metadata.json
outputs/measurements/  new real profiling runs (created by profile_routing.py)
tests/                 165 tests: reproduction, invariants, leakage, schemas, runner, dashboard,
                       multi-site planner, forecasting/regret, agent, scalability, Multi-site page
docs/                  audit, architecture, methodology, assumptions, demo guide, GPU validation plan
models/                OLMoE checkpoint (not modified)
```

## Project phases

| Phase | What | Kind | Script |
|---|---|---|---|
| 1 | Model loading, single-prompt routing | measured | `test_inference.py`, `profile_experts.py` |
| 2 | Six prompts (AI / coding / general), routing + timing | measured | `profile_workloads.py`, `analyze_workloads.py` |
| 3A | Static / balanced / demand-aware placement (in-sample) | derived | `optimize_placement.py` |
| 3B | Heterogeneous GPUs, train 0,2,4 / held-out 1,3,5 | simulated | `evaluate_placement.py` |
| 4A–4C | Batch inference, migration copies, amortisation | simulated | `simulate_inference.py`, `simulate_migration.py`, `analyze_migration_benefit.py` |
| 4D | Event simulator: queues, links, pinning, atomic activation | simulated | `simulate_integrated.py` |
| 4E | Cost-aware migration controller | simulated | `evaluate_migration_policy.py` |
| 4F | Checkpoint-calibrated memory, network sensitivity | checkpoint + simulated | `inspect_expert_memory.py`, `run_calibrated_sweep.py`, `analyze_network_sensitivity.py` |
| 5 | Feedback response: multi-site planner with replicas and residency, site-aware forecasting, placement regret, decision-aware drift, operations agent, six use cases, scalability study | measured routing profiles + synthetic traffic + analytic model | `run_usecases.py`, `run_scalability.py` |

The refactored `moeplace` package regenerates **every** saved simulated CSV bit-for-bit
(`tests/test_reproduction.py`), so the original findings are preserved and their parameters are documented
in `moeplace/registry.py` and on the dashboard's *Data provenance* page.

## Key results (simulated unless noted)

* Demand-aware placement lowers the held-out peak per-GPU load proxy by **29.4%** vs static (load-balanced: 3.1%).
* Phase 4A batch simulation: **−11.9%** simulated p95 latency vs static.
* Migrating 776 experts costs 9.76 GB (checkpoint size). Held-out p95 gets worse below ≈15–23 GB/s, better above.
* The cost-aware controller rejected migration in the saved 8 MB / 12 GB/s scenario (correct on held-out data);
  it is conservative and also rejects at high bandwidth (see `docs/EXPERIMENT_METHODOLOGY.md`).

## Key results of the feedback response (simulated unless noted; see docs/FEEDBACK_RESPONSE.md)

* Planner time (measured): 0.34 ms at the proposal size (16 experts, 3 sites) vs a 2 s target; 272 ms at 16,384
  experts; grows ~linearly with experts (k = 0.94) and sites (k = 0.92), ~flat with devices.
* Site-aware forecast: expert-popularity error 0.004 vs 0.048 for a single global popularity vector.
* Forecast error is a poor alarm: in UC1, 15 of 47 decisions had real placement regret with low WAPE.
* Operations agent vs a never-re-planned plan: e.g. device failure 442 vs 1,371 (objective), max utilisation 0.77 vs 1.19;
  zero residency violations in the sovereign use case.

## Dashboard

`.venv/bin/streamlit run dashboard/app.py` opens on **Start here**, a plain-language walkthrough of the project. The
sidebar then follows the story as 8 numbered steps (real data → placement → moving experts → results), with
**Presentation mode** for talks and an *Advanced & reference* section (multi-site, technical summary, data sources,
glossary). Every page begins with an "About this page" box. **Ask the assistant** (and the *Ask a quick question* box in the sidebar) answers questions in plain words from the project's own results; it is an offline lookup assistant, not an AI model. The page **Advanced & reference →
Multi-site, forecasting & agent** has a use-case selector and six tabs:

| Tab | Shows | Interactive |
|---|---|---|
| Use case explorer | users, assumptions, synthetic demand per site, constraints, objective | live planner for one window; **Run scenario** (one evaluation day, in memory, ~5–20 s) |
| Site-aware forecasting | expert-demand heatmap, predicted vs actual, WAPE and popularity error by site, model comparison | site, window, models |
| Placement regret & drift | error vs regret, harmful/harmless errors, capacity violations, drift alerts, retraining advice | thresholds; saved run or this session's run |
| Operations agent | workflow, observation, forecast, proposal, authoriser checks, tool calls, audit, final action | **Run next cycle**, **Approve/Deny (simulated)**, reset |
| Scalability benchmarks | measured planner time vs the 2 s target; simulated latency and cost, separately labelled | factor |
| Governance & safety | compliance per use case, residency, approval rules, risk register, audit trail, blocked reasons | — |

Default views read saved runs (`outputs/runs/usecases-*`, `scalability-*`); computations run only when a button is
pressed, in memory. The dashboard never loads OLMoE and never writes files (both tested). Demo script:
`docs/DEMO_GUIDE.md`.

## Version control and CI

* Git repository on branch `main`. `.gitignore` excludes the 13 GB weights, tokenizer, `.venv`, caches and
  resume files; only `config.json` and `model.safetensors.index.json` from the checkpoint are tracked, because tests
  and the dashboard read them.
* `.github/workflows/ci.yml` runs on every push and pull request: Python 3.11, installs `requirements-ci.txt`
  (streamlit, plotly, pandas, numpy; no torch, no model download), then runs the full unittest suite and the
  reproduction tests.
* Status: these exact steps pass locally in a clean clone with a fresh virtual environment (165 tests). The workflow
  has **not** run on GitHub yet: no remote is configured. To enable it:

```bash
git remote add origin https://github.com/<you>/moe-expert-placement.git
git push -u origin main      # CI then runs under the repository's Actions tab
```

## Documentation

* `docs/PROJECT_AUDIT.md` — what was found, fixed and still open
* `docs/ARCHITECTURE.md` — modules, data flow, dashboard pages
* `docs/EXPERIMENT_METHODOLOGY.md` — splits, metrics, runner, how to reproduce
* `docs/ASSUMPTIONS.md` — every hypothetical parameter, with units
* `docs/DEMO_GUIDE.md` — 7–10 minute professor demonstration
* `docs/FUTURE_GPU_VALIDATION.md` — what must be measured on NVIDIA GPUs
* `docs/FEEDBACK_RESPONSE.md` — point-by-point response to the proposal feedback, scope change
* `docs/USE_CASES.md`, `docs/AGENT.md`, `docs/FORECASTING_AND_REGRET.md`, `docs/SCALABILITY.md`, `docs/GOVERNANCE.md`

## Environment

Python 3.11 `.venv` (unchanged): torch 2.14.1, transformers 4.57.6, streamlit 1.66.1, plotly 7.1.0,
pandas 3.0.6, numpy 2.4.6, safetensors 0.8.0. See `requirements.txt`. No packages were added.
