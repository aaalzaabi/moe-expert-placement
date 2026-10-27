# Response to the proposal feedback (October 2026)

This file maps every feedback point to what changed in the project, where it lives, and what the evidence
shows. Numbers come from `outputs/runs/usecases-f497898bea/` and `outputs/runs/scalability-75c1324c80/`
(re-run with `scripts/run_usecases.py` and `scripts/run_scalability.py`). Unless marked otherwise, every
latency, utilisation and regret figure is **simulated** with hypothetical hardware; routing profiles are
**measured** on OLMoE; planner run times are **measured** wall clock.

A snapshot of the project before these changes is in `../moe-expert-placement_backup_20261022_before_feedback/`.
All earlier results (Phases 1–4F) are untouched and still regenerate bit-for-bit (`tests/test_reproduction.py`).

## Scope change (stated explicitly, as the architecture brief requires)

| | Proposal | Phases 1–4F (built first) | Now |
|---|---|---|---|
| Infrastructure | heterogeneous, geographically distributed sites | one cluster, 4 heterogeneous GPUs | both: single-cluster event simulator **and** multi-site planner (sites, WAN links, RTT, residency) |
| Experts | 16 experts, 3 sites | 1,024 layer-specific OLMoE experts | 1,024 in use cases; 16 → 16,384 in the scalability study |
| Replication | proposed | not implemented | implemented (budgeted replicas, residency copies) |
| Forecasting | EWMA, possibly learned | none (raw training counts) | 5 forecasters on a site-aware demand tensor |

Reason for the order: the single-cluster event simulator was needed first to understand migration cost
under real expert sizes (12.583 MB). The multi-site layer reuses the same planner logic (it reduces exactly to
the demand-aware placement when there is one site — tested).

## Point-by-point

| # | Feedback | Response | Where |
|---|---|---|---|
| 1–4, 7, 9 | Positive / no action | — | — |
| 5 | Why are inference platform engineers the primary users? Can it be agentic? | Users are now per use case (platform engineers, SRE, compliance officer, FinOps/sustainability lead). An **operations agent** runs the observe → forecast → propose → check → approve → act → learn loop with typed tools, permissions, state, audit log and a human-approval gate. A deterministic authoriser, not the agent, decides whether an action may run. | `moeplace/agent.py`, `docs/AGENT.md`, dashboard *Multi-site → Agent* |
| 6 | Only one use case | Six runnable use cases: multi-region assistant, sovereign/residency tenant, heterogeneous single cluster, workload shift, device failure, off-peak consolidation. Each has users, a decision question, KPIs and governance obligations. | `moeplace/usecases.py`, `docs/USE_CASES.md` |
| 8 | Governance superficial | Use-case-specific compliance (UAE PDPL, UAE health-data law, EU GDPR Chapter V, EU AI Act), frameworks (NIST AI RMF, ISO/IEC 42001, ISO/IEC 23894, UAE AI Charter), a risk register and controls mapped to code. | `docs/GOVERNANCE.md` |
| 10 | Decision-time target too narrow; study scaling; define metrics; separate planner compute from predicted inference | Metric formulas and baseline assumptions defined. One-factor-at-a-time sweep over experts (16–16,384), devices (3–96), sites (1–8), replica budget (0–2×E) and constraint sets, 3 seeds × 5 repeats. Planner time and plan quality reported in separate columns. | `moeplace/scalability.py`, `scripts/run_scalability.py`, `docs/SCALABILITY.md` |
| 11 | Forecast target undefined; global popularity inadequate across sites | Target defined as the time-indexed, site-aware tensor **Y[t, s, e]** (assignments per 15-min window from site s to expert e); horizon 1 h; training on days 1–2; WAPE loss; plus popularity, volume and decision metrics. A site-aware model is compared with a global-popularity model. | `moeplace/workload.py`, `moeplace/forecast.py`, `docs/FORECASTING_AND_REGRET.md` |
| 12 | Drift and placement regret need definitions; distinguish harmless from harmful errors | Placement regret = objective of the plan built from the forecast minus objective of the plan built with perfect foresight (same planner), on realised demand. Every decision point is classified as harmless-high-error or harmful-low-error. The agent refits its forecaster on **realised regret**, not on error. Fragile experts (best and second-best device within 1%) are flagged. | `moeplace/forecast.py`, `moeplace/agent.py` |

## Key results

**Scalability (feedback 10).** Planner time grows about linearly with experts (fitted exponent k = 0.94,
R² = 1.00) and sites (k = 0.92), is almost flat in devices (k = 0.03, vectorised), and grows mildly with
the replica budget (k = 0.22). 16,384 experts × 12 devices × 3 sites: **272 ms** median. The proposal
point (16 experts, 3 sites): **0.34 ms** against the 2 s target. Measured in a Linux VM on the author's
Mac (aarch64); re-run with `.venv/bin/python scripts/run_scalability.py` to time macOS directly. On 20
tiny instances (7 experts, 3 devices) the greedy planner matched the exhaustive optimum every time; that
does not prove optimality at scale.

**Site-aware forecasting (feedback 11).** In the multi-region use case the site-aware model's
expert-popularity error is **0.004** against **0.048** for one global popularity vector (≈11× lower).
Placement regret (aggregate) drops from 30% to 23%. Under the coding surge (UC4) seasonal models fail
(seasonal-naive regret 102%) and the plain EWMA adapts best (7.6%), which is why the agent refits on regret.

**Regret vs error (feedback 12).** Forecast error is a weak alarm. For the site-aware model in UC1, 15 of 47
decision points had real regret (> 2%) with *low* error (WAPE ≤ 10%), and only 2 had high error. The
correlation between WAPE and regret ranged from about 0.0 to 0.6 across models. A small error near the
utilisation cap flips which replicas fit, so regret is the trigger, not error.

**Agent (feedback 5).** Over two evaluation days (47 hourly decisions per use case), mean realised
objective, agent vs never-re-planned static plan (lower is better; oracle = perfect foresight):

| Use case | Agent | Static | Oracle | Note |
|---|---|---|---|---|
| UC1 multi-region | 174 | 319 | 136 | 24 re-plans, 30 GB moved in total |
| UC2 residency | 254 | 299 | 242 | 0 residency violations; 13 changes needed human approval |
| UC3 single cluster | 0.50 | 0.50 | 0.50 | balance mode (J); nothing to gain from re-planning |
| UC4 workload shift | 532 | 1,086 | 432 | static plan overloads (max utilisation 1.11) |
| UC5 device failure | 442 | 1,371 | 384 | one forced restore; static max utilisation 1.19 |
| UC6 consolidation | 402 | 319 | 342 | releases 63 GPU-hours in 2 days at a latency cost |

Locality-mode objective = added cross-site latency per token (ms) + a large penalty for utilisation above
0.7 (`moeplace/multisite.py`). All of this is simulated.

## Dashboard integration and tests

The page *Multi-site, forecast & agent* exposes every feature above in six tabs (README, "Dashboard"). View models
live in `dashboard/ms_views.py` (no Streamlit, unit-tested). `tests/test_dashboard_multisite.py` adds 27 tests:
page registration, six use cases, scenario parameters, validated output of every forecaster, empty and missing data,
hindsight labelling and agent causality, tool permissions, unsafe-proposal and residency blocking (including a
deliberately violating planner), human approval (pending / approve / deny), audit events, scalability loading,
timing labels (a Linux VM is never called Mac hardware), no OLMoE import, unique widget keys, per-use-case session
isolation, and a before/after hash of `outputs/` proving that no dashboard action writes files. Full suite: 150 tests.

## What is still open

* Everything multi-site is an analytic model: no WAN, GPU or serving measurements exist.
* Site traffic is synthetic; only the per-category routing profiles are measured (6 prompts).
* The latency model assumes the 8 experts of a layer are remote independently.
* The agent's human approver is simulated (auto-approve, or deny in a test).
* An LLM can be plugged in as the agent's explainer; none is used or evaluated.
