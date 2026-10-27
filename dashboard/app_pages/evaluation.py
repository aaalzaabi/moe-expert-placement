import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import C, P

ui.page_header("All results side by side",
               "Every saved experiment in one place, with real and simulated evidence kept apart.",
               ("measured", "simulated"), step="Part 4 · Step 7 of 8")
ui.explain(
    "This page collects the results from all earlier steps in four numbered sections: **1** the real data, "
    "**2** how balanced each placement is, **3** simulated request times, **4** the three moving policies "
    "(never move, always move, let the controller decide).",
    ["Each section has a coloured label: green = real, orange = simulation.",
     "In every chart **lower bars are better** (less load, shorter time), except the balance chart where **1.0 is "
     "perfect**.",
     "The switch at the top shows the **saved** results by default; *Re-simulate* recomputes them with your current "
     "settings.",
     "Only 3 test questions were used, so small differences (a few %) are not reliable."],
    "Demand-aware placement wins on balance and speed. The cost-aware controller avoided a move that would have made "
    "the slowest request slower, at the price of a ~1% longer total time.")

view = st.segmented_control("Results", ["saved", "live"], default="saved", key="ev_view",
                            format_func=lambda v: "Saved experiments (as recorded)" if v == "saved"
                            else "Re-simulate under active scenario")

# ---------------------------------------------------------------- measured
st.markdown("### 1 · Measured evidence")
tim = ui.timings()
routing = ui.routing()
with st.container(border=True):
    ui.section("Routing workload", "measured", source="workload_routing.csv")
    with st.container(horizontal=True):
        st.metric("Real prompts profiled", len(tim), border=True)
        st.metric("Train / held-out", "3 / 3", border=True, help="One prompt per category in each split")
        st.metric("Input tokens", int(tim.input_tokens.sum()), border=True)
        st.metric("Median forward pass", f"{tim.forward_seconds.median():.1f} s", border=True,
                  help="Single Apple M4 Max device, FP32, eager attention — not a multi-GPU measurement")
        st.metric("Experts used by held-out prompts but not train",
                  int(((routing[routing.request_id.isin([1, 3, 5])].groupby(["layer_id", "expert_id"]).token_assignments.sum() > 0)
                       & (routing[routing.request_id.isin([0, 2, 4])].groupby(["layer_id", "expert_id"]).token_assignments.sum() == 0)).sum()),
                  border=True, help="Demand the planner could not have seen — a source of generalisation error")

# ---------------------------------------------------------------- placement
st.markdown("### 2 · Placement quality (load proxy)")
if view == "live":
    cfg = ui.scenario()
    pl, _ = ui.layouts_for(cfg)
    held = P.summary(pl, "test_demand", cfg).rename(columns={"peak_load": "peak_test_load", "mean_load": "mean_test_load",
                                                             "imbalance": "test_imbalance"})
    kind, src = "live", None
else:
    held = ui.csv("heldout_comparison.csv")
    kind, src = "simulated", "heldout_comparison.csv"
insample = ui.csv("placement_comparison.csv")
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        ui.section("Held-out peak and mean load (heterogeneous GPUs)", kind, source=src)
        fig = go.Figure()
        fig.add_bar(name="peak", x=held.strategy.map(ui.STRATEGY_LABEL), y=held.peak_test_load,
                    marker_color=[ui.STRATEGY_COLORS[s] for s in held.strategy])
        fig.add_scatter(name="mean", x=held.strategy.map(ui.STRATEGY_LABEL), y=held.mean_test_load, mode="markers",
                        marker=dict(symbol="line-ew-open", size=40, color="#e2e8f0", line_width=2))
        fig.update_layout(yaxis_title="Load proxy (assignments / speed)")
        ui.chart(fig, "Held-out peak load by strategy with mean marker", height=300)
with c2:
    with st.container(border=True):
        ui.section("GPU-load balance (max ÷ mean)", kind)
        both = pd.DataFrame({
            "strategy": insample.strategy.map(ui.STRATEGY_LABEL),
            "In-sample, identical GPUs (3A)": insample.load_imbalance,
            "Held-out, heterogeneous GPUs (3B)": held.test_imbalance.to_numpy()})
        fig = go.Figure()
        for col, color in [("In-sample, identical GPUs (3A)", ui.SLATE), ("Held-out, heterogeneous GPUs (3B)", ui.TEAL)]:
            fig.add_bar(name=col, x=both.strategy, y=both[col], marker_color=color)
        fig.add_hline(y=1.0, line_dash="dot", line_color="#64748b", annotation_text="perfect balance")
        fig.update_layout(barmode="group", yaxis_title="Imbalance", yaxis_range=[0.9, max(1.5, both.iloc[:, 1:].max().max() + 0.05)])
        ui.chart(fig, "Imbalance ratio for in-sample and held-out evaluations", height=300)
st.caption("With identical GPUs, load-balanced and demand-aware produce the same placement (identical greedy rule). "
           "Demand-aware's held-out advantage comes from speed-awareness plus train demand.")

# ---------------------------------------------------------------- inference
st.markdown("### 3 · Simulated inference latency")
if view == "live":
    cfg = ui.scenario()
    rows = []
    for s in C.STRATEGIES:
        r = ui.simulate(ui.cfg_key(cfg.with_(source_strategy=s)), tuple(cfg.test_ids), False).summary
        rows.append({"strategy": s, "p50_latency_ms": r["p50_latency_ms"], "p95_latency_ms": r["p95_latency_ms"],
                     "makespan_ms": r["makespan_ms"], "communication_mb": r["total_link_mb"],
                     "remote_expert_calls": r["remote_expert_calls"]})
    inf = pd.DataFrame(rows)
    note = f"Staggered arrivals ({cfg.arrival_gap_ms:g} ms), {ui.scenario_summary(cfg)}."
else:
    inf = ui.csv("simulated_inference_comparison.csv")
    note = "Phase 4A: all three held-out requests arrive at t = 0 (batch), 12 GB/s, no migration."
with st.container(border=True):
    ui.section("Placement strategies: latency, makespan and communication", kind,
               source=None if view == "live" else "simulated_inference_comparison.csv", caption=note)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        for m, color in [("p50_latency_ms", ui.BLUE), ("p95_latency_ms", ui.TEAL), ("makespan_ms", ui.SLATE)]:
            fig.add_bar(name=m.replace("_ms", "").replace("_", " "), x=inf.strategy.map(ui.STRATEGY_LABEL), y=inf[m],
                        marker_color=color)
        fig.update_layout(barmode="group", yaxis_title="Simulated ms")
        ui.chart(fig, "Simulated p50, p95 and makespan per placement", height=300)
    with c2:
        base = inf.set_index("strategy").loc["static"]
        rel = inf.assign(**{
            "p95 vs static": [f"{-ui.pct(v, base.p95_latency_ms):+.1f}%" for v in inf.p95_latency_ms],
            "makespan vs static": [f"{-ui.pct(v, base.makespan_ms):+.1f}%" for v in inf.makespan_ms],
            "comm vs static": [f"{-ui.pct(v, base.communication_mb):+.1f}%" for v in inf.communication_mb]})
        rel["strategy"] = rel.strategy.map(ui.STRATEGY_LABEL)
        ui.table(rel[["strategy", "p95 vs static", "makespan vs static", "comm vs static", "remote_expert_calls"]],
                 "Relative change versus static placement")
        st.caption("Negative = lower (better) latency. Demand-aware slightly increases communication: it moves hot "
                   "experts off the gateway GPU toward faster GPUs.")

# ---------------------------------------------------------------- migration policies
st.markdown("### 4 · Migration policies")
if view == "live":
    pol, req, dec, _ = ui.controller(ui.cfg_key(ui.scenario()))
    dec_txt = f"{dec.decision.upper()} (forecast makespan {dec.forecast_makespan_gain_pct:+.2f}%, p95 {dec.forecast_p95_regression_pct:+.2f}%)"
else:
    pol, req = ui.csv("policy_comparison.csv"), ui.csv("policy_request_latency.csv")
    d = ui.csv("policy_decision.csv").iloc[0]
    dec_txt = (f"{d.decision.upper()} (forecast makespan {d.forecast_makespan_gain_pct:+.2f}%, "
               f"p95 {d.forecast_p95_regression_pct:+.2f}%) — 8 MB experts, 12 GB/s")
with st.container(border=True):
    ui.section("Never vs always vs cost-aware migration (held-out requests)", kind,
               source=None if view == "live" else "policy_comparison.csv", caption=f"Controller decision: {dec_txt}")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for m, color in [("p50_latency_ms", ui.BLUE), ("p95_latency_ms", ui.TEAL), ("makespan_ms", ui.SLATE)]:
            fig.add_bar(name=m.replace("_ms", "").replace("_", " "), x=pol.policy.map(ui.POLICY_LABEL), y=pol[m],
                        marker_color=color)
        fig.update_layout(barmode="group", yaxis_title="Simulated ms")
        ui.chart(fig, "Latency and makespan per migration policy", height=300)
    with c2:
        fig = go.Figure()
        for p_name in pol.policy:
            d = req[req.policy == p_name]
            fig.add_bar(name=ui.POLICY_LABEL[p_name], x=[f"req {r}" for r in d.request_id], y=d.latency_ms,
                        marker_color=ui.POLICY_COLORS[p_name], text=d.layout, textposition="inside")
        fig.update_layout(barmode="group", yaxis_title="Latency (ms)")
        ui.chart(fig, "Per-request latency under each policy", height=300)
    show = pol.copy()
    show["policy"] = show.policy.map(ui.POLICY_LABEL)
    ui.table(show.drop(columns=["failure_reason"], errors="ignore").round(2), "Migration policy metrics",
             column_config={"migration_mb": st.column_config.NumberColumn("migration overhead (MB)")})

# ---------------------------------------------------------------- amortisation
if view == "saved":
    st.markdown("### 5 · Migration amortisation (idealised)")
    with st.container(border=True):
        mb = ui.csv("migration_benefit.csv")
        ui.section("Cumulative time: stay static vs migrate then run optimised", "simulated", source="migration_benefit.csv",
                   caption="Phase 4C assumption: inference pauses during migration, then repeats the Phase 4A batch.")
        fig = go.Figure()
        fig.add_scatter(x=mb.batches, y=mb.static_total_ms, name="stay static", line=dict(color=ui.SLATE, width=3))
        fig.add_scatter(x=mb.batches, y=mb.optimized_total_ms, name="migrate (8 MB @ 12 GB/s) then demand-aware",
                        line=dict(color=ui.TEAL, width=3))
        fig.update_xaxes(type="log", title="Batches served")
        fig.update_yaxes(type="log", title="Total simulated ms")
        ui.chart(fig, "Static and migrate cumulative time; migration breaks even after one batch", height=300)

st.info(ui.SIM_DISCLAIMER + " With 3 held-out requests, no confidence intervals can be computed; differences of a few "
        "percent should not be read as significant.", icon=":material/model_training:")

ui.step_footer("app_pages/evaluation.py")
