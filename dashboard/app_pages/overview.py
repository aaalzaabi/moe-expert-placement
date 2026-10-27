import plotly.graph_objects as go
import streamlit as st

import ui
from ui import C

ui.page_header("Technical summary",
               "All key numbers on one page, for readers who already know the project. New here? Start with "
               "**Start here** instead.", ("measured", "simulated"))
ui.explain(
    "A compact, technical version of the whole project: model facts, the best placement, the controller's "
    "decisions, the system diagram and the three most important charts.",
    ["**Top row:** real facts about the model and the data collected.",
     "**Second row:** the best placement method and the controller's decisions (saved, and for your current settings).",
     "**Diagram:** the pipeline from the real model (green) to the simulation (amber).",
     "**Charts:** busiest-GPU load, moving policies, and the effect of network speed."],
    "Same story as *Start here*, in technical form.", expanded=False)

cfg = ui.scenario()
mc = ui.model_config()
routing = ui.routing()
tim = ui.timings()
heldout = ui.csv("heldout_comparison.csv")
saved_decision = ui.csv("policy_decision.csv").iloc[0]

# ---------------------------------------------------------------- KPI rows
st.markdown("##### Real model (measured / checkpoint)")
with st.container(horizontal=True):
    st.metric("MoE layers", mc["num_hidden_layers"], border=True, help="config.json: num_hidden_layers")
    st.metric("Experts per layer", mc["num_experts"], border=True)
    st.metric("Experts selected per token", mc["num_experts_per_tok"], border=True, help="Top-k routing")
    st.metric("Layer-specific experts", mc["num_hidden_layers"] * mc["num_experts"], border=True,
              help="Expert identity is (layer_id, expert_id); expert 5 in layer 0 differs from expert 5 in layer 1.")
    st.metric("Profiled requests", routing.request_id.nunique(), border=True,
              help=f"{int(tim.input_tokens.sum())} input tokens across AI, coding and general prompts (prefill only).")

best = heldout.sort_values("peak_test_load").iloc[0]
static_peak = heldout.set_index("strategy").loc["static", "peak_test_load"]
live_pol, _, live_dec, _ = ui.controller(ui.cfg_key(cfg))

st.markdown("##### Simulated planning results")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Best placement (held-out load proxy)", ui.STRATEGY_LABEL[best.strategy],
              f"{ui.pct(best.peak_test_load, static_peak):.1f}% lower peak load vs static", border=True,
              height="stretch",
              help="Peak per-GPU token assignments divided by hypothetical speed, on held-out requests 1, 3, 5. "
                   "A load proxy, not latency.")
with c2:
    st.metric("Saved Phase 4E controller decision", saved_decision.decision.upper(),
              f"p95 forecast {saved_decision.forecast_p95_regression_pct:+.1f}% (limit "
              f"{saved_decision.max_p95_regression_pct:+.0f}%)", delta_color="off", border=True, height="stretch",
              help="From outputs/policy_decision.csv (8 MB experts, 12 GB/s). Forecast on training requests only.")
with c3:
    st.metric("Active-scenario decision (live)", live_dec.decision.upper(),
              f"makespan forecast {live_dec.forecast_makespan_gain_pct:+.1f}%", delta_color="off", border=True,
              height="stretch", help=ui.scenario_summary(cfg))

# ---------------------------------------------------------------- architecture
st.markdown("##### System architecture")
st.mermaid_chart("""
flowchart LR
    A["OLMoE-1B-7B<br/>16 x 64 experts, top-8<br/>(real model, MPS)"]:::measured --> B["Router profiler<br/>token to expert counts<br/>per layer & request"]:::measured
    B --> C["Placement optimizer<br/>static / balanced /<br/>demand-aware"]:::derived
    C --> D["Migration controller<br/>forecast on TRAIN requests<br/>approve / reject"]:::sim
    D --> E["Inference simulator<br/>GPU queues, links,<br/>weight copies"]:::sim
    E -->|"held-out evaluation<br/>(TEST requests)"| F["Results &<br/>dashboard"]:::sim
    classDef measured fill:#0f3d36,stroke:#2dd4bf,color:#e6fffa
    classDef derived fill:#10294a,stroke:#60a5fa,color:#e0ecff
    classDef sim fill:#3a2a0c,stroke:#f59e0b,color:#fff4dc
""", alt="Pipeline from the real OLMoE model through router profiling, placement, migration control and simulation")
st.caption("Green = measured on the real model · Blue = derived from measurements · Amber = simulated with "
           "hypothetical hardware. The simulator does not place experts on physical GPUs.")

# ---------------------------------------------------------------- key charts
left, right = st.columns(2)
with left:
    with st.container(border=True):
        ui.section("Held-out peak GPU load proxy", "simulated", source="heldout_comparison.csv")
        fig = go.Figure(go.Bar(
            x=[ui.STRATEGY_LABEL[s] for s in heldout.strategy], y=heldout.peak_test_load,
            marker_color=[ui.STRATEGY_COLORS[s] for s in heldout.strategy],
            text=[f"{v:.0f}" for v in heldout.peak_test_load], textposition="outside"))
        fig.update_layout(yaxis_title="Peak normalized load (assignments / speed)", showlegend=False)
        ui.chart(fig, "Peak held-out GPU load by placement strategy; demand-aware lowest", height=330)
        st.caption("Lower is better. Demand-aware reduces the most loaded GPU's work by "
                   f"{ui.pct(heldout.peak_test_load.min(), static_peak):.1f}% on unseen prompts.")
with right:
    with st.container(border=True):
        pol = ui.csv("policy_comparison.csv")
        ui.section("Migration policies (saved Phase 4E)", "simulated", source="policy_comparison.csv")
        fig = go.Figure()
        for metric, name in [("p95_latency_ms", "p95 latency"), ("makespan_ms", "Makespan")]:
            fig.add_bar(name=name, x=[ui.POLICY_LABEL[p] for p in pol.policy], y=pol[metric],
                        text=[f"{v:.0f}" for v in pol[metric]], textposition="outside")
        fig.update_layout(barmode="group", yaxis_title="Simulated ms")
        ui.chart(fig, "Simulated p95 latency and makespan for never, always and cost-aware migration", height=330)
        st.caption("Cost-aware rejected the move (p95 regression forecast), matching never-migrate.")

with st.container(border=True):
    net = ui.phase4f("network")
    cal = ui.phase4f("calibrated")
    ui.section("Network sensitivity: when does migrating pay off?", "simulated",
               caption="p95 latency change from migrating, held-out requests. Positive = migration helps.")
    fig = go.Figure()
    for df, name, color in [(net, "8 MB placeholder", ui.SLATE), (cal, "12.583 MB checkpoint-derived", ui.TEAL)]:
        piv = df.pivot(index="bandwidth_gb_per_s", columns="policy", values="p95_latency_ms")
        imp = 100 * (piv["static"] - piv["migrate"]) / piv["static"]
        fig.add_scatter(x=imp.index, y=imp.values, mode="lines+markers", name=name, line=dict(color=color, width=3))
    fig.add_hline(y=0, line_dash="dot", line_color="#64748b")
    fig.update_xaxes(type="log", title="Link bandwidth (GB/s, log scale)", tickvals=list(C.PHASE4F_BANDWIDTHS))
    fig.update_yaxes(title="p95 improvement (%)")
    ui.chart(fig, "p95 improvement from migration rises with bandwidth and crosses zero between 12 and 25 GB/s",
             height=320)

with st.expander("What the metrics mean", icon=":material/info:"):
    st.markdown("""
| Metric | Meaning | Kind |
|---|---|---|
| **Token assignments** | How many (token, top-8 slot) selections an expert received in a forward pass | Measured |
| **Normalized load (load proxy)** | Sum of assignments on a GPU ÷ its hypothetical relative speed. Not utilization, not latency | Derived/simulated |
| **p50 / p95 latency** | Median / 95th percentile of simulated request completion time (arrival → last layer). With 3 requests, p95 ≈ the slowest request | Simulated |
| **Makespan** | Simulated time when the last request finishes | Simulated |
| **Migration volume** | Experts moved × bytes per expert (8 MB placeholder or 12.583 MB checkpoint BF16) | Simulated |
| **Activation time** | Time when every copy of the new placement has finished and the layout switches atomically | Simulated |
""")
st.info(ui.SIM_DISCLAIMER, icon=":material/model_training:")
