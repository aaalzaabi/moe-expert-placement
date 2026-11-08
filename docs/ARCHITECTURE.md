# Architecture

```mermaid
flowchart LR
  M["OLMoE checkpoint<br/>(models/)"] -->|"profile_routing.py / profile_workloads.py<br/>explicit, expensive"| R["Measured routing<br/>workload_routing.csv"]
  M -->|"safetensors headers"| W["Expert sizes<br/>phase4f/memory"]
  R --> P["moeplace.placement<br/>static / balanced / demand-aware"]
  P --> S["moeplace.simulator<br/>event simulation"]
  W --> CFG["moeplace.config<br/>SimConfig + presets"]
  CFG --> P & S & POL
  S --> POL["moeplace.policy<br/>controller + state machine"]
  POL --> E["moeplace.experiment<br/>runs, sweeps, metadata"]
  E --> O["outputs/runs/&lt;run_id&gt;/"]
  R & O & S --> UI["dashboard/<br/>Streamlit control center"]
```

## `moeplace/` (no torch import anywhere)

| Module | Responsibility |
|---|---|
| `paths.py` | Absolute project-root paths (`MOE_PROJECT_ROOT` overrides) |
| `config.py` | `SimConfig` frozen dataclass: cluster, cost model, network, workload, migration, controller thresholds. `validate()` raises `ConfigError` with all problems. Model constants from `config.json`. `PRESETS` reproduce saved experiments. Units: GB/s, ms, bytes. |
| `data.py` | CSV loaders with required-column schemas; routing invariants (1,024 rows per request, top-8 sums); trace building; JSON-only model metadata |
| `placement.py` | `assign()` (three strategies, capacity-aware), `plan()`, per-GPU metrics (load proxy, slots, memory feasibility), moved-expert diff |
| `simulator.py` | `run()` discrete-event simulator (port of Phase 4D with optional tracing, shared-port network, byte memory budget, fault injection, Poisson arrivals); `run_batch_at_zero()` (4A); `migration_only()` (4B) |
| `policy.py` | `MigrationStateMachine` (proposed → approved/rejected → transferring → active/failed); `decide()` (train-only forecast + reasons); `evaluate()` (never / always / cost-aware on held-out) |
| `experiment.py` | `run_scenario()` and `sweep()` with deterministic run ids `name-<hash(config, routing sha256)>`, atomic writes, `metadata.json`, error capture |
| `registry.py` | Provenance (measured / checkpoint / derived / simulated) and parameters for every original output |
| `legacy.py` | Project-root `chdir` + overwrite guard used by the original scripts |

### Simulator event model

Events: `request_arrival`, `expert_start`, `expert_computed`, `expert_returned`, `approve_migration`,
`copy_start`, `copy_done`, ordered by (time, insertion serial). A request pins the active layout at arrival.
For each layer, every selected expert is dispatched gateway → GPU (link queue), computed (GPU FIFO queue,
`n × compute_ms / speed`), and returned (link queue). The next layer starts when all returned. At approval, the
staging check (resident + incoming ≤ staging slots, optional byte budget) runs, then all copies are queued on
the same link queues as inference. Activation is atomic when the last copy lands; a failed copy rolls back.

## Multi-site extension (feedback response)

```mermaid
flowchart LR
  R["Measured routing<br/>workload_routing.csv"] --> WL["moeplace.workload<br/>category profiles + synthetic site traffic<br/>Y[t, s, e]"]
  UC["moeplace.usecases<br/>6 scenarios"] --> WL
  WL --> FC["moeplace.forecast<br/>5 forecasters, WAPE, regret, drift"]
  FC --> AG["moeplace.agent<br/>tools · authoriser · approval · audit"]
  AG --> MS["moeplace.multisite<br/>sites, replicas, capacity/memory/residency,<br/>utilisation + latency model, migration cost"]
  MS --> AG
  MS --> SC["moeplace.scalability<br/>planner timing sweep"]
  AG & FC & SC --> RUNS["outputs/runs/usecases-*, scalability-*"]
  RUNS --> UI2["dashboard: Multi-site, forecast & agent"]
```

| Module | Responsibility |
|---|---|
| `multisite.py` | `Site`, `Device`, `CostModel`, `Constraints`, `Topology`; `plan_sites()` greedy min-bottleneck placement with residency copies, budgeted replicas, hysteresis; `evaluate()` utilisation/latency/residency; `migration_cost()`; `exact_small()` brute force for tiny instances |
| `workload.py` | Measured per-category routing profiles; synthetic site traffic (daily cycle by time zone, mixes, shifts) → Y[t, s, e] |
| `forecast.py` | naive, seasonal-naive, EWMA, site-aware EWMA, global EWMA; WAPE, popularity and volume error; placement regret; drift table |
| `usecases.py` | Six use cases with users, KPIs, governance focus, topology, traffic, events |
| `agent.py` | `PlacementAgent` (tools with permissions, state, learning on regret), `Authoriser`, approval hook, explainer hook |
| `scalability.py` | One-factor-at-a-time timing sweep, growth fits, optimality gap |

## Dashboard

`dashboard/app.py` (entry) → `st.navigation` → `dashboard/app_pages/*.py`. Shared code in `dashboard/ui.py`:
Plotly dark template, provenance badges, `st.cache_data` loaders and cached simulations keyed by the full
config JSON. Theme: `dashboard/.streamlit/config.toml` (script-level config, applies from any launch directory).

The dashboard now opens on `welcome.py` with five visible destinations: Welcome, Guided story, Try your own prompt,
See the agent decide and Ask the assistant. `present.py` is a six-scene, one-visual-at-a-time walkthrough using saved
measurements and the existing GPU visualiser. `agent_visual.py` calls the same `ms_views.run_agent_cycles()` as the
multi-site research page, with separate per-scenario state (`av_states`), no background execution and the existing
approval gate. It commits new cycles only after success and immediately disables further cycles while approval is pending.
All old pages remain registered and linkable; the **Research tools** toggle reveals them in the menu. **Show all details**
in sidebar settings expands secondary charts on research pages. `ui.explain` and `ui.step_footer` still support those pages.

`my_prompt.py` caches a versioned model resource and calls `live_model.generate()` only on form submission. Generation
uses the local instruction chat template with a visible AI/MoE system instruction (LLM defaults to large language model),
greedy decoding, KV caching and a bounded output length. Explicitly specified other domains remain allowed. `live_model.route()`
then measures a separate raw-input prefill. `Generated` and `Routed` are stored in session state, so opening charts never
reruns inference. The cached model has an inference lock for concurrent callers. The answer is visible first; charts are
explicitly opt-in and do not claim to cover decode routing. User-facing errors are shown outside the collapsed status box.

The following registry includes the original research pages (now under Research tools):

| Sidebar section | Page (file) | Data |
|---|---|---|
| Presentation | Welcome (`welcome.py`, default) | clear entry point and one primary presentation button |
| Presentation | Guided story (`present.py`) | six scenes; saved evidence, fixed and labelled presets |
| Presentation | See the agent decide (`agent_visual.py`) | live simulated cycles using the existing agent and approval gate |
| — | Project overview (`start.py`, research) | five-step plain-language story from saved files |
| — | Presentation mode (`demo.py`) | 10-step guided demo, saved files only |
| — | Ask the assistant (`assistant_chat.py`) | offline Q&A engine `dashboard/assistant.py` (topic + glossary matching, live facts from `ui.assistant_facts`); also a sidebar quick-ask popover |
| Part 1 · real data | 1. What is inside the model? (`model.py`) | config.json, index.json, phase4f/memory |
| Part 1 · real data | 2. Which experts are busy? (`routing.py`) | workload_routing / timings (+ new measurement runs) |
| Part 2 · simulation | 3. Where should experts go? (`placement.py`) | live placement under the session scenario |
| Top | Try your own prompt (`my_prompt.py`, `dashboard/live_model.py`) | the only page that runs OLMoE: on button press, loads the local checkpoint in BF16 (never downloads), records the user's prompt's top-8 routing per layer (real), generates a written answer and replays input routing on the simulated GPUs for each placement (simulated). Nothing written to disk |
| Part 2 · simulation | 3b. Watch the GPUs at work (`gpu_sim.py`, view model `dashboard/gpu_viz.py`) | animated placement (same order as `placement.assign`) and one real question flowing through the layers; simplified compute-only layer barrier, labelled as such |
| Part 2 · simulation | 4. How fast would requests run? (`inference.py`) | live event simulation with traces |
| Part 3 · simulation | 5. Should we move experts? (`migration.py`) | live controller + session-only sandbox |
| Part 3 · simulation | 6. Does network speed matter? (`network.py`) | saved Phase 4F + optional live fine sweep |
| Part 4 · results | 7. All results side by side (`evaluation.py`) | saved, or live re-simulation |
| Part 4 · results | 8. What we learned and the limits (`findings.py`) | numbers read from saved CSVs |
| Advanced & reference | Multi-site, forecasting & agent (`multisite.py`) | six tabs (use cases, forecasting, regret & drift, agent, scalability, governance); view models in `dashboard/ms_views.py`; saved `usecases-*` / `scalability-*` runs plus in-memory runs on button press; governance tables parsed from `docs/GOVERNANCE.md` |
| Advanced & reference | Technical summary (`overview.py`) | saved + one cached live decision |
| Advanced & reference | Where the data comes from (`provenance.py`) | registry, outputs/runs, outputs/measurements |
| Advanced & reference | Glossary (`glossary.py`) | static term list |

A session-wide **scenario** (`st.session_state["scenario"]`, a validated `SimConfig`) is edited on the planner,
inference and migration pages and shown in the sidebar. The migration sandbox keeps its own state machine and
event log in session state; it never writes files.
