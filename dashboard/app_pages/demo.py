import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import C

st.title("Presentation mode")
st.caption("Use this when presenting: a 7–10 minute walkthrough in 10 steps, with talking points for each. Press **Next** to advance. Uses saved measurements and precomputed "
           "simulation results only — the OLMoE model is not loaded and nothing is recomputed.")

STEPS = [
    ("The real OLMoE architecture", 45, "model"),
    ("Measured expert-routing demand", 60, "routing"),
    ("Static placement (baseline)", 45, "placement"),
    ("Optimized placement", 60, "placement"),
    ("Predicted load-balancing advantage", 60, "evaluation"),
    ("Proposing an expert migration", 45, "migration"),
    ("Migration transfer costs", 60, "network"),
    ("The cost-aware controller decides", 60, "migration"),
    ("Simulated inference comparison", 60, "evaluation"),
    ("What remains before real GPU deployment", 60, "findings"),
]
if "demo_step" not in st.session_state:
    st.session_state.demo_step = 0


def go_to(i):
    st.session_state.demo_step = max(0, min(len(STEPS) - 1, i))


step = st.session_state.demo_step
title, secs, page = STEPS[step]
elapsed = sum(s for _, s, _ in STEPS[:step])
total = sum(s for _, s, _ in STEPS)

with st.container(border=True):
    n1, n2, n3, n4 = st.columns([1, 1, 4, 2], vertical_alignment="center")
    n1.button("Back", icon=":material/arrow_back:", on_click=go_to, args=(step - 1,), disabled=step == 0, width="stretch")
    n2.button("Next", icon=":material/arrow_forward:", on_click=go_to, args=(step + 1,), disabled=step == len(STEPS) - 1,
              width="stretch", type="primary")
    n3.progress((step + 1) / len(STEPS), text=f"Step {step + 1} of {len(STEPS)} · **{title}**")
    n4.caption(f":material/schedule: ~{secs} s · {elapsed // 60}:{elapsed % 60:02d} of {total // 60}:{total % 60:02d}")

held = ui.csv("heldout_comparison.csv")
hp = ui.csv("heterogeneous_placements.csv")
SPEEDS = C.DEFAULT.gpu_speeds

visual, notes = st.columns([3, 2])


def talk(points: list[str], kind: str, takeaway: str):
    with notes:
        with st.container(border=True, height="stretch"):
            ui.badge(kind)
            st.markdown("**Talking points**")
            st.markdown("\n".join(f"- {p}" for p in points))
            st.success(takeaway, icon=":material/lightbulb:")


with visual:
    with st.container(border=True):
        if step == 0:
            mc = ui.model_config()
            with st.container(horizontal=True):
                st.metric("Layers", mc["num_hidden_layers"], border=True)
                st.metric("Experts / layer", mc["num_experts"], border=True)
                st.metric("Top-k", mc["num_experts_per_tok"], border=True)
                st.metric("Distinct experts", 1024, border=True)
                st.metric("Per expert", "12.58 MB", border=True)
            st.mermaid_chart("""
flowchart LR
  T[token] --> A[attention] --> R{router: 64 scores}
  R -->|top-8| E["8 expert MLPs of this layer"] --> S((sum)) --> N[next of 16 layers]
""", alt="OLMoE layer: router selects 8 of 64 experts per token")
        elif step == 1:
            r = ui.routing()
            mat = r.groupby(["layer_id", "expert_id"]).token_assignments.sum().unstack()
            fig = go.Figure(go.Heatmap(z=mat.values, colorscale=ui.HEAT, colorbar=dict(title="assignments"),
                                       hovertemplate="expert (L%{y}, E%{x}): %{z}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", title="Layer")
            fig.update_xaxes(title="Expert")
            ui.chart(fig, "Measured routing heatmap over 6 prompts", height=400)
        elif step in (2, 3):
            strat = "static" if step == 2 else "demand_aware"
            rows = hp[hp.strategy == strat]
            load = rows.groupby("gpu_id").test_demand.sum() / pd.Series(SPEEDS)
            fig = go.Figure(go.Bar(x=[f"GPU {g}\n(speed {s:g})" for g, s in enumerate(SPEEDS)], y=load.values,
                                   marker_color=[ui.GPU_COLORS[g] for g in range(4)],
                                   text=[f"{v:.0f}" for v in load.values], textposition="outside"))
            fig.add_hline(y=load.mean(), line_dash="dot", line_color="#94a3b8", annotation_text="mean")
            fig.update_layout(yaxis_title="Held-out load proxy (assignments / speed)", yaxis_range=[0, 980],
                              title=f"{ui.STRATEGY_LABEL[strat]} placement: per-GPU load")
            ui.chart(fig, f"Per-GPU load proxy for {strat} placement", height=400)
        elif step == 4:
            fig = go.Figure(go.Bar(x=held.strategy.map(ui.STRATEGY_LABEL), y=held.peak_test_load,
                                   marker_color=[ui.STRATEGY_COLORS[s] for s in held.strategy],
                                   text=[f"{v:.0f}  (imbalance {i:.2f})" for v, i in zip(held.peak_test_load, held.test_imbalance)],
                                   textposition="outside"))
            fig.update_layout(yaxis_title="Peak held-out load proxy", yaxis_range=[0, 1050])
            ui.chart(fig, "Peak load by strategy on held-out prompts", height=400)
        elif step == 5:
            s = hp[hp.strategy == "static"].set_index(["layer_id", "expert_id"]).gpu_id
            d = hp[hp.strategy == "demand_aware"].set_index(["layer_id", "expert_id"]).gpu_id
            mv = pd.DataFrame({"src": s, "dst": d})[s != d]
            mat = mv.groupby(["src", "dst"]).size().unstack(fill_value=0).reindex(index=range(4), columns=range(4), fill_value=0)
            st.metric("Experts to relocate", f"{len(mv)} of 1024", border=True)
            fig = go.Figure(go.Heatmap(z=mat.values, x=[f"→ GPU {g}" for g in range(4)], y=[f"GPU {g} →" for g in range(4)],
                                       colorscale=ui.HEAT, text=mat.values, texttemplate="%{text}"))
            fig.update_yaxes(autorange="reversed")
            ui.chart(fig, "Expert copies per directed link for the proposed migration", height=330)
        elif step == 6:
            rows = []
            for kind, label in [("network", "8 MB"), ("calibrated", "12.583 MB")]:
                df = ui.phase4f(kind)
                m = df[df.policy == "migrate"]
                for x in m.itertuples():
                    rows.append({"size": label, "bw": f"{x.bandwidth_gb_per_s} GB/s", "activation": x.migration_activation_ms - 100})
            a = pd.DataFrame(rows)
            with st.container(horizontal=True):
                st.metric("Volume (8 MB placeholder)", "6.21 GB", border=True)
                st.metric("Volume (checkpoint 12.583 MB)", "9.76 GB", border=True)
            fig = go.Figure()
            for size, color in [("8 MB", ui.SLATE), ("12.583 MB", ui.TEAL)]:
                d = a[a["size"] == size]
                fig.add_bar(name=size, x=d.bw, y=d.activation, marker_color=color)
            fig.update_layout(barmode="group", yaxis_title="Approval → activation (ms)")
            ui.chart(fig, "Migration activation time by bandwidth and expert size", height=330)
        elif step == 7:
            d = ui.csv("policy_decision.csv").iloc[0]
            st.markdown(f"### Decision: :red[{d.decision.upper()}]")
            checks = [(f"Migration feasible (staging memory)", bool(d.migration_feasible)),
                      (f"Forecast makespan gain {d.forecast_makespan_gain_pct:+.2f}% ≥ {d.min_makespan_gain_pct:.0f}%",
                       d.forecast_makespan_gain_pct >= d.min_makespan_gain_pct),
                      (f"Forecast p95 change {d.forecast_p95_regression_pct:+.2f}% ≤ {d.max_p95_regression_pct:+.0f}%",
                       d.forecast_p95_regression_pct <= d.max_p95_regression_pct)]
            for text, ok in checks:
                st.markdown(f"{':material/check_circle:' if ok else ':material/cancel:'} {':green' if ok else ':red'}[{text}]")
            st.caption(f"Forecast on training requests {d.forecast_request_ids}; evaluated afterwards on {d.evaluation_request_ids}. "
                       "Scenario: 8 MB experts, 12 GB/s.")
            st.mermaid_chart("stateDiagram-v2\n direction LR\n [*] --> Proposed\n Proposed --> Approved\n Proposed --> Rejected\n"
                             " Approved --> Transferring\n Transferring --> Active\n Transferring --> Failed\n"
                             " classDef current fill:#7f1d1d,stroke:#f87171,color:#fff\n class Rejected current",
                             alt="Migration state machine with Rejected highlighted")
        elif step == 8:
            pol = ui.csv("policy_comparison.csv")
            fig = go.Figure()
            for m, color in [("p95_latency_ms", ui.TEAL), ("makespan_ms", ui.SLATE)]:
                fig.add_bar(name=m.replace("_ms", "").replace("_", " "), x=pol.policy.map(ui.POLICY_LABEL), y=pol[m],
                            marker_color=color, text=[f"{v:.0f}" for v in pol[m]], textposition="outside")
            fig.update_layout(barmode="group", yaxis_title="Simulated ms (held-out)")
            ui.chart(fig, "Held-out p95 and makespan per migration policy", height=400)
        else:
            st.markdown("""
| Done (this project) | Still required on real GPUs |
|---|---|
| Real OLMoE routing measured (6 prompts) | Hundreds of prompts incl. decode tokens |
| Checkpoint-calibrated expert size | Measured per-expert kernel time on A100/H100 |
| Placement algorithms + held-out evaluation | Expert-parallel serving (e.g. vLLM/SGLang EP) with custom placement |
| Event simulator with queues, links, pinning, atomic activation | Real NVLink / PCIe / InfiniBand bandwidth & contention |
| Cost-aware controller, train-only forecast | Live weight transfer with rollback, measured p50/p95 |
""")

if step == 0:
    talk(["OLMoE-1B-7B: 16 MoE layers × 64 experts, each token uses 8 experts per layer.",
          "An expert is identified by (layer, expert) — 1,024 distinct expert MLPs.",
          "Checkpoint inspection: 12.885 GB of expert weights, 12.583 MB per expert in BF16."],
         "checkpoint", "Experts dominate model memory, so where they live is an infrastructure decision.")
elif step == 1:
    talk(["We ran the real model on six prompts (AI, coding, general) on Apple MPS and recorded the router's top-8 choices.",
          "Demand is highly skewed: a few experts per layer take most tokens; many get none.",
          "Different workloads prefer different experts."],
         "measured", "Expert demand is uneven and workload-dependent — a static layout ignores this.")
elif step == 2:
    talk(["Baseline: experts assigned round-robin, 256 per simulated GPU.",
          "The cluster is heterogeneous (hypothetical speeds 1.0, 1.3, 0.8, 1.6).",
          "Load proxy = held-out token assignments ÷ speed. The slow GPU 2 becomes the bottleneck."],
         "simulated", f"Static peak load: {held.set_index('strategy').loc['static', 'peak_test_load']:.0f}.")
elif step == 3:
    talk(["Demand-aware greedy: place the hottest experts first on the GPU with the lowest (load + demand) ÷ speed.",
          "Planned with training prompts only (0, 2, 4); shown here on held-out prompts (1, 3, 5).",
          "Fast GPUs take more work; per-GPU load is nearly flat."],
         "simulated", f"Peak load falls to {held.set_index('strategy').loc['demand_aware', 'peak_test_load']:.0f}.")
elif step == 4:
    h = held.set_index("strategy")
    talk([f"Demand-aware: {ui.pct(h.loc['demand_aware', 'peak_test_load'], h.loc['static', 'peak_test_load']):.1f}% lower peak than static on unseen prompts.",
          f"Load-balanced (speed-blind) gains only {ui.pct(h.loc['balanced', 'peak_test_load'], h.loc['static', 'peak_test_load']):.1f}%.",
          "In the batch simulation this translates to ~12% lower simulated p95 latency.",
          "This is a load proxy under hypothetical speeds — not a measured speed-up."],
         "simulated", "Using measured demand and GPU speed together gives the balance.")
elif step == 5:
    talk(["A running cluster starts from the static layout; reaching demand-aware means copying weights.",
          "776 of 1,024 experts change GPU; copies are staged next to old weights, then switched atomically.",
          "Requests keep the layout that was active when they arrived."],
         "simulated", "Better placement is not free: most experts must move.")
elif step == 6:
    talk(["776 × 12.583 MB ≈ 9.8 GB (6.2 GB with the old 8 MB placeholder).",
          "Copies share links with live inference traffic, delaying in-flight requests.",
          "At 3 GB/s activation takes ~318 ms; at 100 GB/s ~17 ms."],
         "simulated", "Migration cost is dominated by link bandwidth.")
elif step == 7:
    talk(["The controller forecasts never-migrate vs migrate on training requests only.",
          "Makespan would improve ~4%, but the slowest request's latency would rise ~33% → reject.",
          "The held-out requests are never used for this decision (no leakage)."],
         "simulated", "Approve only when the gain clears the threshold without hurting tail latency.")
elif step == 8:
    talk(["Held-out check (8 MB, 12 GB/s): always-migrate would make p95 2.8% worse for a 1.3% makespan gain.",
          "Cost-aware matches never-migrate, so rejecting was the right call here.",
          "Network sweep: above ~15–25 GB/s migration starts to pay off on p95."],
         "simulated", "The controller avoided a harmful migration in this scenario.")
else:
    talk(["Everything after routing is simulated with hypothetical GPU and network parameters.",
          "Sample size: 6 prompts, 3 held-out → indicative, not statistically conclusive.",
          "Next: more prompts with decode routing, seeded replicates, calibration on real NVIDIA GPUs."],
         "simulated", "A validated planning methodology, ready for real multi-GPU calibration.")

PAGE_FILES = {"model": "app_pages/model.py", "routing": "app_pages/routing.py", "placement": "app_pages/placement.py",
              "evaluation": "app_pages/evaluation.py", "migration": "app_pages/migration.py",
              "network": "app_pages/network.py", "findings": "app_pages/findings.py"}
st.page_link(PAGE_FILES[page], label="Open the full interactive page", icon=":material/open_in_new:")
