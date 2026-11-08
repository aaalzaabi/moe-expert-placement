# Demo guide (7–10 minutes)

## Before the meeting

```bash
cd ~/Desktop/MOE/moe-expert-placement
.venv/bin/python -m unittest discover -s tests -t .     # optional: runs the automated test suite
.venv/bin/streamlit run dashboard/app.py                # opens http://localhost:8501
```

## Recommended presentation: six scenes

Open **Welcome → Start the presentation**. The **Guided story** needs no model loading or internet. Use the
**Next / Back** buttons. Each scene has one main visual and a takeaway; **Presenter notes — what to say** contains
speaking guidance and a link to the underlying experiment. The sidebar shows only five presentation choices until
**Research tools** is enabled.

1. **Meet the experts (~1 min).** Explain a router selecting 8 of 64 experts, repeated across 16 layers.
2. **See uneven demand (~1 min).** Show measured expert-selection counts from six real prompts. Only three prompts
   train the placement; three are held out.
3. **Place them on GPUs (~2 min).** Press Play; compare simple and demand-aware placement. Every square is a
   real expert identity, but the GPUs and speeds are simulated. Bars are load proxies, not milliseconds.
4. **Check the cost of moving (~1 min).** Explain the saved controller's rejection: the forecast p95 regression
   exceeds the 0% limit despite a makespan gain. This scene is explicitly the 8 MB / 12 GB/s legacy experiment.
5. **Meet the agent (~2 min).** Show the loop. Optionally open **See the agent decide**, choose GPU failure and
   press **Run the agent**. The agent restores a simulated plan. Select Observe / Forecast / Propose / Check /
   Act & learn to explain its tools. For human approval, select Data must stay local and run two hours: the second
   cycle waits for approval; approve or deny before continuing. Start again resets only that scenario's session state.
6. **What we learned (~1 min).** Summarise the held-out comparison, the six-prompt limitation and future hardware validation.

Optional live model moment: **Try your own prompt** → type a short question → **Run on the real model**. Read
**Model’s answer** first, then turn on **See how the experts handled it**. The answer uses the checkpoint's chat
format; routing is a separate raw-input pass and the pictures do not trace generation. Select a longer answer if the
length limit cuts it off. This is the only part of the demo that loads the model. **Unload the model** frees it.

## Original ten-step presentation (retained)

Enable **Research tools → Original ten-step demo**. The earlier script below remains available as a detailed alternative.

## Script

| # | Step | Time | Say | If asked, open |
|---|---|---|---|---|
| 1 | Real OLMoE architecture | 0:45 | 16 layers × 64 experts, top-8; experts are (layer, expert) pairs; 12.583 MB each, 93% of the checkpoint | Model explorer → expert inspector |
| 2 | Measured routing | 1:00 | Real top-8 choices for 6 prompts; demand is skewed and workload-dependent | Routing analytics → category pills, divergence chart |
| 3 | Static placement | 0:45 | Round-robin on a hypothetical heterogeneous cluster; slow GPU 2 is the bottleneck | Placement planner |
| 4 | Optimized placement | 1:00 | Greedy on (load + demand) ÷ speed, trained on prompts 0/2/4, shown on held-out 1/3/5 | Placement planner → assignment matrix |
| 5 | Predicted advantage | 1:00 | −29.4% peak load proxy; −11.9% simulated p95 in batch; load-balanced alone gets −3.1% | Experimental evaluation |
| 6 | Propose migration | 0:45 | 776 of 1,024 experts must move; copies staged, then atomic switch | Migration controller (top) |
| 7 | Transfer cost | 1:00 | 9.76 GB; activation 318 ms at 3 GB/s vs 17 ms at 100 GB/s; copies compete with inference | Network sensitivity |
| 8 | Controller decision | 1:00 | Train-only forecast: +4.1% makespan but +32.9% p95 → reject; no held-out leakage | Migration controller → sandbox |
| 9 | Simulated comparison | 1:00 | Held-out: always-migrate p95 +2.8% worse; cost-aware = never-migrate; crossover ≈15–23 GB/s | Evaluation → policies |
| 10 | What remains | 1:00 | Simulation, small sample, needs real NVIDIA calibration | Findings & limitations |

Total ≈ 9:15.

## Optional live moment (1 minute)

On **Migration controller**: click **Approve (override)** to show the sandbox moving
Proposed → Approved → Transferring → Active, then drag the **transfer playback** slider. Then **New proposal**,
switch on **Inject copy failure**, approve again, and show the rollback (requests stay on the static layout).
State clearly: "This is a simulation in my browser session — no GPU or file is touched."

## Likely questions

* *Is the 29% a speed-up?* No — a load proxy under hypothetical speeds on 3 held-out prompts.
* *Why only six prompts?* Each forward pass took 12–52 s on MPS in FP32; `scripts/profile_routing.py` is ready for a larger set.
* *Could the controller cheat?* Tests corrupt held-out routing and show the decision is unchanged.
* *Why does the controller reject at 100 GB/s?* Its p95 forecast over 3 training requests is dominated by the in-flight request; this is a documented limitation and a next experiment.


## Feedback walkthrough (10 minutes): points 5, 6, 8, 10, 11, 12

Open **Advanced & reference → Multi-site, forecasting & agent**. Everything runs from saved runs or in memory;
no model loading, no files written. Say once at the start: *"Traffic here is synthetic over measured OLMoE routing;
latency and cost are simulated; planner times are measured."*

| # | Time | Where | Do | Say | Feedback point |
|---|---|---|---|---|---|
| 1 | 0:45 | Use case selector | Open the list | Six use cases, each with users, a decision question, KPIs and governance focus | 6 |
| 2 | 1:00 | Use case explorer (UC1) | Show the demand chart, move the window slider | Regions peak at different hours with different mixes; the live planner re-places all experts in well under a second | 6 |
| 3 | 1:30 | Site-aware forecasting | Heatmap, then predicted vs actual, then the two error charts | The target is Y[t, site, expert] per 15 min over a 1-hour horizon; site-aware popularity error ≈0.004 vs 0.048 for one global vector | 11 |
| 4 | 1:30 | Placement regret & drift | Scatter, quadrant table, the "Why a small error can hurt" note | Regret uses hindsight for evaluation only; small errors near the cap can cause real regret, so drift is triggered on regret | 12 |
| 5 | 2:00 | Operations agent (switch to UC2) | Run next cycle ×4 → approval pending → Approve | The agent proposes, a deterministic authoriser decides, residency-site changes wait for a human; show tool calls and audit | 5, 8 |
| 6 | 1:15 | Scalability benchmarks | Proposal-point metric, then factor = experts | 0.34 ms vs the 2 s target; ~linear in experts and sites; timings recorded on a Linux VM, not Mac hardware | 10 |
| 7 | 1:15 | Governance & safety (UC2) | Compliance row, residency, approval rules, risk register | Documented considerations (PDPL, UAE health-data law, GDPR Ch. V, EU AI Act; NIST AI RMF, ISO/IEC 42001), not a legal sign-off | 8 |
| 8 | 0:45 | Use case explorer (UC5) | Run scenario (1 day) | The real pipeline runs live; the agent restores feasibility after the GPU failure where a static plan overloads | 5, 6 |

If asked *"is this real serving?"*: no. The multi-site layer is an analytic model; the single-cluster pages hold the
event-simulator evidence; real GPU validation is in `docs/FUTURE_GPU_VALIDATION.md`.
