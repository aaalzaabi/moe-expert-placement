# Use cases

Defined in `moeplace/usecases.py`; run with `scripts/run_usecases.py`; shown on the dashboard page
*Multi-site → Use cases / Agent*. Sites, rates, mixes, speeds and events are **hypothetical**. Per-category
expert-routing profiles (ai / coding / general) are **measured** on OLMoE.

Common setup for the geographic use cases: three sites, Abu Dhabi (UTC+4), Frankfurt (UTC+1) and Singapore
(UTC+8); 4 devices each with relative speeds 1.0, 1.3, 0.8, 1.6 and 256 expert slots; 1.25 GB/s and 60 ms RTT
between sites. Traffic peaks at 14:00 local time (3,000 / 2,500 / 2,800 requests per 15 minutes, 400 tokens per
request); category mix differs by site (Abu Dhabi general-heavy, Frankfurt coding-heavy, Singapore AI-heavy).
Locality mode: minimise cross-site latency while utilisation stays ≤ 0.7, with up to 2 extra copies per expert.
4 simulated days: days 1–2 train the forecasters, days 3–4 are evaluated with a decision every hour.

| # | Use case | Primary users | Decision question | KPIs | Governance focus |
|---|---|---|---|---|---|
| UC1 | Multi-region enterprise assistant | Platform engineers; operations agent | Where should each expert live when regions peak at different hours and ask different things? | J, cross-site latency, WAPE, regret | Cross-border transfer of prompt-derived data |
| UC2 | Sovereign / regulated tenant | Compliance officer; platform engineers; agent | What does keeping every UAE token in the UAE cost? | Residency violations (must be 0), J, extra copies, approvals | Data residency; auditability |
| UC3 | Single cluster, mixed GPU generations | Platform engineers | How to spread experts over fast and slow GPUs in one data centre? | J, imbalance; request-level p95 from the event simulator | Change management |
| UC4 | Workload shift (coding surge in Frankfurt, day 3, ×1.6 traffic, 85% coding) | Operations agent; SRE | Does the system notice its forecast stopped supporting good placements? | Regret-triggered refits; regret before/after | Model-risk monitoring and drift |
| UC5 | Device failure (abudhabi-g3 lost on day 3) | SRE; operations agent | How fast is a feasible plan restored, at what transfer cost? | Time to feasible plan, reload bytes, J after failure | Incident management, continuity |
| UC6 | Off-peak consolidation | FinOps / sustainability lead; agent | Which devices can be released at night without overload? | GPU-hours released, J ≤ 0.6, migrations | Cost/energy reporting; human approval for capacity changes |

## Results (simulated, two evaluation days, `usecases-f497898bea`)

Objective: locality mode = added latency per token (ms) + penalty above 0.7 utilisation; UC3 = utilisation J.

| Use case | Best forecaster | Agent objective | Static plan | Oracle | Max utilisation agent / static | Other |
|---|---|---|---|---|---|---|
| UC1 | site_ewma | 174 | 319 | 136 | 0.76 / 0.84 | 24 re-plans, 30 GB moved |
| UC2 | site_ewma | 254 | 299 | 242 | 0.83 / 0.83 | 0 residency violations; 13 human approvals; 110 GB moved |
| UC3 | (all equal) | 0.50 | 0.50 | 0.50 | 0.82 / 0.82 | no re-plans needed |
| UC4 | ewma | 532 | 1,086 | 432 | 0.90 / **1.11** | 25 forecaster refits |
| UC5 | site_ewma | 442 | 1,371 | 384 | 0.77 / **1.19** | 1 forced restore after the failure |
| UC6 | site_ewma | 402 | 319 | 342 | 0.76 / 0.84 | **63 GPU-hours released**; 12 consolidations, each approved |

Reading the table:
* **UC1:** time-zone offsets make one placement wrong for most of the day; re-planning hourly roughly halves the
  objective versus a static plan.
* **UC2:** residency is satisfied by construction (Abu Dhabi holds every expert). It costs more copies and more
  approvals, and in this setup *lowers* average latency because Abu Dhabi traffic never leaves the site.
* **UC3:** with one site and a stable mix there is nothing to gain from re-planning; the Phase 3–4 event-simulator
  results remain the evidence for this setting.
* **UC4 / UC5:** the static plan overloads (utilisation > 1) after the shift or the failure; the agent does not.
* **UC6:** the trade-off is explicit: GPU-hours saved for higher latency at night.
