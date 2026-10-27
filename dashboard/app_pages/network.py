import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import C

ui.page_header("Does network speed matter?",
               "How the speed of the connection between GPUs decides whether moving experts pays off.",
               ("simulated", "checkpoint"), step="Part 3 · Step 6 of 8")
ui.explain(
    "We repeated the move-or-not experiment at four network speeds: **3, 12, 25 and 100 GB/s** (gigabytes per "
    "second). We did it twice: with an early guess of **8 MB per expert**, and with the **real size, 12.6 MB**, "
    "measured from the model files.",
    ["**Top numbers (break-even):** the network speed at which moving starts to help. Below it, moving hurts.",
     "**Line charts:** request times with and without moving. The dotted line = never moving. Where a coloured "
     "line drops below the dotted one, moving helps.",
     "**Bar charts:** how long the copy takes, and how much data is copied.",
     "**Compare bandwidths:** pick speeds to see the numbers side by side.",
     "The x-axis uses a log scale (3 → 12 → 25 → 100)."],
    "On slow networks moving experts hurts; on fast networks (above ≈ 15–23 GB/s) it helps. These speeds are "
    "assumptions, so the exact break-even point applies to this scenario only.")

if not ui.detailed():
    st.caption(":material/warning: Simulation with assumed hardware (4 GPUs, assumed speeds). GB/s = gigabytes per "
               "second; 100 Gbps ≈ 12.5 GB/s.")
else:
    st.warning("Hardware assumptions: 4 simulated GPUs (speeds 1.0/1.3/0.8/1.6), one FIFO queue per directed link, "
               "0.1 ms link latency, migration approved at 100 ms, arrivals every 200 ms. Bandwidth is in GB/s "
               "(gigabytes); 100 Gbps ≈ 12.5 GB/s. The crossover below is specific to this scenario.",
               icon=":material/warning:")

net, cal = ui.phase4f("network"), ui.phase4f("calibrated")
allf = pd.concat([net, cal], ignore_index=True)


def improvement(df):
    piv = df.pivot_table(index="bandwidth_gb_per_s", columns="policy",
                         values=["p95_latency_ms", "makespan_ms", "migration_activation_ms", "migration_mb"])
    out = pd.DataFrame({
        "p95_static": piv["p95_latency_ms"]["static"], "p95_migrate": piv["p95_latency_ms"]["migrate"],
        "makespan_static": piv["makespan_ms"]["static"], "makespan_migrate": piv["makespan_ms"]["migrate"],
        "activation_ms": piv["migration_activation_ms"]["migrate"] - 100.0,
        "migration_mb": piv["migration_mb"]["migrate"]})
    out["p95_improvement_pct"] = 100 * (out.p95_static - out.p95_migrate) / out.p95_static
    out["makespan_improvement_pct"] = 100 * (out.makespan_static - out.makespan_migrate) / out.makespan_static
    return out.reset_index()


imp = {"8 MB placeholder": improvement(net), "12.583 MB checkpoint-derived": improvement(cal)}
colors = {"8 MB placeholder": ui.SLATE, "12.583 MB checkpoint-derived": ui.TEAL}

# ---------------------------------------------------------------- crossover KPIs
with st.container(horizontal=True):
    for name, d in imp.items():
        x95 = ui.crossover(d.bandwidth_gb_per_s, d.p95_improvement_pct)
        xmk = ui.crossover(d.bandwidth_gb_per_s, d.makespan_improvement_pct)
        st.metric(f"p95 break-even · {name}", f"≈ {x95:.0f} GB/s" if x95 else "not crossed", border=True,
                  help="Interpolated (linear in log-bandwidth) between tested points. Not measured.")
        st.metric(f"Makespan break-even · {name}", f"≈ {xmk:.0f} GB/s" if xmk else "not crossed", border=True)

with st.container():
    with st.container(border=True):
        ui.section("Bandwidth vs p95 latency", "simulated", source="phase4f/network/network_results.csv")
        fig = go.Figure()
        fig.add_scatter(x=imp["8 MB placeholder"].bandwidth_gb_per_s, y=imp["8 MB placeholder"].p95_static,
                        name="never migrate", line=dict(color="#e2e8f0", dash="dot", width=2))
        for name, d in imp.items():
            fig.add_scatter(x=d.bandwidth_gb_per_s, y=d.p95_migrate, name=f"migrate · {name}", mode="lines+markers",
                            line=dict(color=colors[name], width=3))
        fig.update_xaxes(type="log", title="GB/s (log)", tickvals=list(C.PHASE4F_BANDWIDTHS))
        fig.update_yaxes(title="Simulated p95 (ms)")
        ui.chart(fig, "p95 latency falls with bandwidth; migration is worse below about 15-25 GB/s", height=320)
with ui.more("makespan, copy time, transfer volume, compare speeds, fine sweep"):
    with st.container():
        with st.container(border=True):
            ui.section("Bandwidth vs makespan", "simulated")
            fig = go.Figure()
            fig.add_scatter(x=imp["8 MB placeholder"].bandwidth_gb_per_s, y=imp["8 MB placeholder"].makespan_static,
                            name="never migrate", line=dict(color="#e2e8f0", dash="dot", width=2))
            for name, d in imp.items():
                fig.add_scatter(x=d.bandwidth_gb_per_s, y=d.makespan_migrate, name=f"migrate · {name}",
                                mode="lines+markers", line=dict(color=colors[name], width=3))
            fig.update_xaxes(type="log", title="GB/s (log)", tickvals=list(C.PHASE4F_BANDWIDTHS))
            fig.update_yaxes(title="Simulated makespan (ms)")
            ui.chart(fig, "Makespan versus bandwidth for both expert sizes", height=320)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            ui.section("Migration activation time", "simulated",
                       caption="From approval (t = 100 ms) to atomic switch, including contention with inference traffic.")
            fig = go.Figure()
            for name, d in imp.items():
                fig.add_bar(name=name, x=[f"{b:g} GB/s" for b in d.bandwidth_gb_per_s], y=d.activation_ms,
                            marker_color=colors[name])
            fig.update_layout(barmode="group", yaxis_title="ms")
            ui.chart(fig, "Activation time per bandwidth and expert size", height=300)
    with c2:
        with st.container(border=True):
            ui.section("Migration transfer volume", "checkpoint",
                       caption="776 experts × size. Bandwidth does not change the volume, only how long it occupies links.")
            vol = pd.DataFrame({"scenario": list(imp), "GB": [d.migration_mb.iloc[0] / 1000 for d in imp.values()]})
            fig = go.Figure(go.Bar(x=vol.scenario, y=vol.GB, marker_color=[colors[s] for s in vol.scenario],
                                   text=[f"{v:.2f} GB" for v in vol.GB], textposition="outside"))
            fig.update_layout(yaxis_title="GB moved")
            ui.chart(fig, "Total migration volume for 8 MB and 12.583 MB experts", height=300)

    # ---------------------------------------------------------------- compare two bandwidths
    with st.container(border=True):
        st.markdown("**Compare bandwidths**")
        sel = st.pills("Bandwidths (saved runs)", list(C.PHASE4F_BANDWIDTHS), selection_mode="multi", default=[12, 25],
                       format_func=lambda b: f"{b} GB/s", key="nw_sel")
        scen = st.segmented_control("Expert size", list(imp), default="12.583 MB checkpoint-derived", key="nw_scen")
        if sel and scen:
            d = imp[scen].set_index("bandwidth_gb_per_s").loc[sorted(sel)]
            show = d[["p95_static", "p95_migrate", "p95_improvement_pct", "makespan_static", "makespan_migrate",
                      "makespan_improvement_pct", "activation_ms"]].round(2).reset_index()
            ui.table(show, "Side-by-side comparison of selected bandwidths", column_config={
                "p95_improvement_pct": st.column_config.NumberColumn("p95 Δ (%)", help="Positive = migration helps"),
                "makespan_improvement_pct": st.column_config.NumberColumn("makespan Δ (%)")})

    # ---------------------------------------------------------------- live fine sweep
    with st.container(border=True):
        ui.section("Fine-grained sweep (live simulation)", "live",
                   caption="Same simulator and assumptions as the saved runs, at 16 log-spaced bandwidths. Reproduces the "
                           "saved points exactly at 3, 12, 25 and 100 GB/s.")
        if st.toggle("Run fine sweep", key="nw_live", help="~2 s the first time; cached afterwards"):
            grid = tuple(sorted(set(np.round(np.geomspace(2, 200, 16), 2).tolist()) | set(map(float, C.PHASE4F_BANDWIDTHS))))
            with st.spinner("Simulating..."):
                sw = ui.bandwidth_sweep(ui.cfg_key(C.DEFAULT), grid, (C.LEGACY_EXPERT_BYTES, C.CHECKPOINT_EXPERT_BYTES))
            fig = go.Figure()
            for eb, name in [(C.LEGACY_EXPERT_BYTES, "8 MB placeholder"), (C.CHECKPOINT_EXPERT_BYTES, "12.583 MB checkpoint-derived")]:
                d = sw[np.isclose(sw.expert_mb, eb / C.MB)]
                piv = d.pivot(index="bandwidth_gb_per_s", columns="policy", values="p95_latency_ms")
                y = 100 * (piv["static"] - piv["migrate"]) / piv["static"]
                fig.add_scatter(x=y.index, y=y.values, name=name, mode="lines+markers", line=dict(color=colors[name], width=3))
                x0 = ui.crossover(y.index.to_numpy(), y.to_numpy())
                if x0:
                    fig.add_vline(x=x0, line_dash="dot", line_color=colors[name],
                                  annotation_text=f"≈{x0:.0f} GB/s", annotation_position="top")
            fig.add_hline(y=0, line_color="#64748b")
            fig.update_xaxes(type="log", title="GB/s (log)")
            fig.update_yaxes(title="p95 improvement from migrating (%)")
            ui.chart(fig, "Fine-grained p95 improvement versus bandwidth with interpolated crossover", height=340)
            st.caption("The default scenario's GPU speeds, arrivals and timing are used regardless of the sidebar scenario, "
                       "so this curve is directly comparable to the saved Phase 4F runs.")

ui.step_footer("app_pages/network.py")
