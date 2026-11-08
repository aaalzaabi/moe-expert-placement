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

# Dashboard (loads the model only if you press 'Run on the real model' on 'Try your own prompt')
.venv/bin/streamlit run dashboard/app.py

# Test suite (stdlib unittest; ~25 s)
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
dashboard/app.py       Streamlit control center (dashboard/app_pages/*.py, dashboard/ui.py, gpu_viz.py,
                       dashboard/ms_views.py = view models of the Multi-site page)
dashboard/legacy_app.py  original single-page dashboard (deprecations fixed)
outputs/               ORIGINAL results (read-only by convention; scripts refuse to overwrite without --force)
outputs/runs/          new simulation runs, one folder per run id with metadata.json
outputs/measurements/  new real profiling runs (created by profile_routing.py)
tests/                 183 tests: reproduction, invariants, leakage, schemas, runner, dashboard,
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

`.venv/bin/streamlit run dashboard/app.py` opens on **Welcome**. The default menu has five choices:
**Welcome**, **Guided story**, **Try your own prompt**, **See the agent decide**, and **Ask the assistant**.

* **Guided story** is a six-scene presentation (7–10 minutes): experts → measured demand → animated placement →
  migration decision → operations agent → findings. Each scene has one main visual, a takeaway and optional presenter notes.
  It uses the fixed default hardware scenario, and the migration scene explicitly uses the saved Phase 4E preset.
* **Try your own prompt** generates a written OLMoE answer first (select up to 64, 128 or 256 output tokens).
  Answers receive a visible AI/MoE system context to disambiguate terms such as LLM; explicit other-topic questions remain supported.
  A **See how the experts handled it** toggle reveals measured input routing and simulated GPU animations.
  Routing is a separate raw-input prefill pass; it does not trace the generated answer's decode tokens.
* **See the agent decide** runs the existing operations agent one simulated hour per click for GPU failure,
  a demand shift or residency. Five stages show observation, forecast, proposal, rule checks and action/learning.
  Approval is held for a person when required, with approve/deny actions and a tool/audit trail.
* **Research tools** reveals every original page, including the detailed multi-site experiments and the original
  ten-step demo. **Show all details** inside the sidebar settings expands secondary content on research pages.
* **Ask the assistant** is an offline lookup helper, separate from both the real OLMoE model and the operations agent.

The research page **Multi-site experiments** has a use-case selector and six tabs:

| Tab | Shows | Interactive |
|---|---|---|
| Use case explorer | users, assumptions, synthetic demand per site, constraints, objective | live planner for one window; **Run scenario** (one evaluation day, in memory, ~5–20 s) |
| Site-aware forecasting | expert-demand heatmap, predicted vs actual, WAPE and popularity error by site, model comparison | site, window, models |
| Placement regret & drift | error vs regret, harmful/harmless errors, capacity violations, drift alerts, retraining advice | thresholds; saved run or this session's run |
| Operations agent | workflow, observation, forecast, proposal, authoriser checks, tool calls, audit, final action | **Run next cycle**, **Approve/Deny (simulated)**, reset |
| Scalability benchmarks | measured planner time vs the 2 s target; simulated latency and cost, separately labelled | factor |
| Governance & safety | compliance per use case, residency, approval rules, risk register, audit trail, blocked reasons | — |

Default views read saved runs (`outputs/runs/usecases-*`, `scalability-*`); computations run only when a button is
pressed, in memory. The dashboard never writes files, and it loads OLMoE only when a user presses **Run on the real model** on **Try your own prompt** (BF16 on the Mac's GPU, about 15 GB peak; generation time depends on answer length; *Unload the model* frees it). Every other page, and the whole test suite, runs without torch (tested). Demo script:
`docs/DEMO_GUIDE.md`.

## Version control and CI

* Git repository on branch `main`. `.gitignore` excludes the 13 GB weights, tokenizer, `.venv`, caches and
  resume files; only `config.json` and `model.safetensors.index.json` from the checkpoint are tracked, because tests
  and the dashboard read them.
* `.github/workflows/ci.yml` runs on every push and pull request: Python 3.11, installs `requirements-ci.txt`
  (streamlit, plotly, pandas, numpy; no torch, no model download), then runs the full unittest suite and the
  reproduction tests.
* Status: the current full suite passes locally (183 tests). The earlier 175-test suite also passed in a clean
  clone with a fresh environment containing only CI dependencies. The workflow has **not** run on GitHub yet:
  no remote is configured. To enable it:

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
