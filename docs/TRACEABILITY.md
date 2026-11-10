# Proposal-to-implementation traceability matrix

Sources: *Project Proposal* (2 Nov 2026, objectives O1–O5, sections 4–10) and *Technical Architecture* document
(sections A–L, Figures 1–11). Status was checked on 10 Nov 2026 against the code, the test suite (218 tests, all
passing locally, in Docker and on GitHub Actions) and the saved runs.

Status values: **Completed** (implemented and tested) · **Partially completed** · **Modified** (implemented
differently from the design, with a reason) · **Not implemented** (with impact) · **Simulated** (works, but the
infrastructure behaviour is modelled, not measured).

Evidence paths are relative to the repository root. `EVIDENCE.md` = `outputs/evidence/EVIDENCE.md`.

## 1. Objectives (proposal section 3)

| # | Proposed objective | Status | Evidence / justification |
|---|---|---|---|
| O1a | Preserve the six measured prompts; check top-8 routing and expert-identity schemas | Completed | `outputs/workload_routing.csv` (6 × 1,024 rows), schema + invariant checks in `moeplace/data.py`; `tests/test_data_and_schemas.py` (17 tests) |
| O1b | Collect ≥ 100 additional prompts, grouped 60/20/20 split | **Partially completed** | The 24-prompt `extended` set (8 per category, 231 tokens) was profiled on OLMoE on 10 Nov 2026: `outputs/measurements/routing-20261110T123026Z/` (top-8 invariant verified; 1.8 % of experts unobserved vs 15 % before). Placement was re-evaluated on it (`outputs/runs/extended_prompts-5e1b709690/`): −29.8 % vs static on the 18 new prompts, better in all 1,000 random splits; `tests/test_extended_prompts.py`. The ≥ 100-prompt target was not reached. **Impact:** migration, controller and agent results still rest on 6 prompts / 44 tokens; held-out p95 over 3 requests is descriptive only. Stated in report §8. |
| O2 | Compare static, random, load-balanced and demand-aware placement on held-out demand; 10 % peak-load target | Completed (simulated hardware) | Demand-aware −29.4 % held-out peak load vs static (−29.8 % on 18 unseen prompts, `EVIDENCE.md` §3.1); beats 100 % of 10,000 random placements; 5.6 % above the speed-proportional lower bound (`EVIDENCE.md` §3, `scripts/collect_evidence.py`) |
| O3a | Compare never / always / cost-aware migration | Completed (simulated) | `moeplace/policy.py`; 30 seeded Poisson scenarios at 12 and 100 GB/s with 95 % CIs (`EVIDENCE.md` §4, `outputs/runs/replicates30*`) |
| O3b | Six operational use cases with reasons for every action; infeasible moves blocked; approvals | Completed (simulated) | `moeplace/usecases.py`, `moeplace/agent.py`; `outputs/runs/usecases-f497898bea/`; zero residency violations; `tests/test_agent_usecases.py`, `tests/test_dashboard_multisite.py` |
| O4a | Scaling over 16–16,384 experts, 3–96 devices, 1–8 sites, replica budgets | Completed (measured planner time) | `outputs/runs/scalability-75c1324c80/`; 0.34 ms at proposal size, 272 ms at 16,384 experts (`EVIDENCE.md` §5) |
| O4b | Five forecasters, numerical error and decision regret | Completed (synthetic site traffic) | `moeplace/forecast.py`; site-aware popularity error 0.004 vs 0.048 global (`EVIDENCE.md` §6) |
| O5a | Dashboard, CLI runners, metadata, tests | Completed | `dashboard/`, `scripts/run_experiment.py`, `metadata.json` per run, 218 tests |
| O5b | GitHub CI, passing before claiming it | Completed | `.github/workflows/ci.yml`; runs on github.com/aaalzaabi/moe-expert-placement are green |
| O5c | Container for CPU-only planning and dashboard replay (proposal §6: "not yet implemented") | **Completed in this deliverable** | `Dockerfile`, `docker-compose.yml`; 218 tests pass in the container; 14/14 end-to-end checks on SQLite and on PostgreSQL (`EVIDENCE.md` §2) |

## 2. Architectural components (architecture document, Figures 1, 2, 2b, 7, 8)

| Proposed component | Implementation status | Evidence / justification |
|---|---|---|
| Router profiler (real OLMoE routing) | Completed | `scripts/profile_routing.py`, `profile_workloads.py`; measured CSVs |
| Expert-size calibration from the checkpoint | Completed | `scripts/inspect_expert_memory.py`; 12,582,912 B per expert, test-pinned |
| Configuration layer (devices, network, seeds, sizes) | Completed | `moeplace/config.py` (`SimConfig.validate()` → `ConfigError`), `service/settings.py` (`MOE_*` env vars) |
| Placement planner (static / balanced / demand-aware) | Completed | `moeplace/placement.py`; `tests/test_placement.py` |
| Discrete-event simulator (queues, links, pinning, atomic activation, rollback) | Completed (simulated) | `moeplace/simulator.py`; 21 invariant tests |
| Cost-aware migration controller + state machine | Completed (simulated) | `moeplace/policy.py`; illegal transitions raise `InvalidTransition` (tested) |
| Multi-site planner (replicas, memory, residency) | Completed (analytic model) | `moeplace/multisite.py`; greedy = exhaustive optimum on 20/20 tiny instances |
| Site-aware forecaster, regret, drift | Completed (synthetic traffic) | `moeplace/forecast.py`, `moeplace/workload.py` |
| Operations agent with typed tools, authoriser, approval, audit | Completed (simulated actions) | `moeplace/agent.py`; human approval is a dashboard button / API call |
| Experiment runner with hashed run ids and metadata | Completed | `moeplace/experiment.py`; reproduction tests regenerate every saved CSV bit-for-bit |
| Streamlit dashboard | Completed | `dashboard/` (28 files); 52 UI tests via Streamlit `AppTest` |
| Planner API: plan, propose, approve, apply, audit | **Modified** | `service/`. Paths are `/plan`, `/proposals/{id}/approve` instead of `/v1/placements:plan`, `/v1/migrations/{id}:approve`. Reason: one version exists, so a `/v1/` prefix added no value yet; adding it is a router prefix change. Idempotency-Key and 409 semantics are implemented as designed. |
| Decision / approval / audit persistence (PostgreSQL) | **Modified → Completed** | SQLAlchemy models with DB-level invariants; SQLite by default for zero-setup review, PostgreSQL 16 verified end-to-end in Docker (`outputs/evidence/e2e_postgres_container.json`) |
| Authentication (OIDC roles planner.read/write/approve) | **Not implemented** | **Impact:** the API must not be exposed beyond localhost; the approver name is self-declared. Mitigation path documented in `docs/SERVICE.md` (identity-aware proxy or OAuth2). |
| SSE migration status stream | Not implemented | Polling `GET /proposals/{id}` and `/audit` covers the need for a single operator. |
| Migration executor (gRPC + mTLS) and vLLM expert-parallel serving | **Not implemented (deferred)** | Requires multi-GPU hardware (ADR-6). `apply` changes the active plan in the database only — stated in API docs and the report. A real two-device expert-parallel smoke run (MPS + CPU) exists: `outputs/real_runs/smoke_mps_cpu-*` (logits identical to the single-device model). |
| Prometheus + Grafana monitoring | Not implemented | Structured request logs with `X-Request-ID` and `/health` are implemented; metrics export is future work. |
| Scheduled production worker for hourly cycles | **Modified** | Cycles are user-initiated (dashboard button, API call). Proposal §4 already states no unattended service runs. |
| Docker Compose control plane (planner-api, dashboard, postgres, prometheus, grafana) | **Partially completed** | `docker-compose.yml` runs api, dashboard, postgres (profile) and tests; Prometheus/Grafana omitted (above). |
| Data retention / pruning policy | Not implemented | Documented as a policy (architecture §H); no personal data is stored. |

## 3. Proposal deployment stages (proposal section 6)

| Stage | Status | Evidence |
|---|---|---|
| Current prototype (local profiling, saved results, simulated planning, agent cycles) | Completed | README quick start |
| Reproducible package (pinned deps, GitHub remote + CI, container) | Completed | `requirements*.txt` (exact pins), CI green, Docker image |
| Shadow mode on real telemetry | Not started | Needs a serving system to observe |
| Controlled GPU trial | Not started | `docs/FUTURE_GPU_VALIDATION.md` lists what must be measured |

## 4. Changes that improved on the original design

| Change | Why it is better |
|---|---|
| Single-cluster event simulator built before the multi-site model | Migration cost under real 12.58 MB experts had to be understood first; the multi-site planner reduces exactly to demand-aware placement for one site (tested). |
| Deterministic Authoriser, no LLM in the decision path | Rules are auditable and unit-tested; the API and the agent share the same `Authoriser`, so the two cannot disagree. |
| Database invariants (partial unique index: one active plan; unique approval per proposal) | Hold even if two requests race or application code has a bug. |
| Optimistic stale check on `apply` (`base_plan_id` must still be active) | A proposal's gain is only valid against the plan it was assessed on; tested end-to-end (409). |
| Overwrite guards and hashed run ids for experiments | Original results can never be silently replaced. |
