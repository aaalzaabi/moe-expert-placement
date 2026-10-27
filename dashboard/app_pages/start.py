import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import C, D

st.title("Start here")
st.markdown("#### Where should the parts of a large AI model live on GPUs, and when is it worth moving them?")
st.markdown(
    "This project takes a real AI model (**OLMoE**), measures which of its internal parts are used most, "
    "and then — in a **computer simulation** — works out how to spread those parts across several GPUs so no GPU "
    "is overloaded, and whether rearranging them is worth the cost.")

with st.container(border=True):
    st.markdown("**How to use this dashboard**")
    c1, c2, c3 = st.columns(3)
    c1.markdown(":material/format_list_numbered: **Follow the numbered steps** in the left sidebar, top to bottom. "
                "Each step answers one question.")
    c2.markdown(":material/help: **Every page starts with an \"About this page\" box** that explains what you see "
                "and the main takeaway.")
    c3.markdown(":material/label: **Coloured labels say where numbers come from.** "
                "Green = real measurement. Orange/violet = simulation with assumed hardware.")
    st.caption("Unfamiliar word? See **Glossary** under *Advanced & reference* in the sidebar.")

st.subheader("The story in five steps")

r = ui.routing()
held = ui.csv("heldout_comparison.csv").set_index("strategy")
dec = ui.csv("policy_decision.csv").iloc[0]
pol = ui.csv("policy_comparison.csv").set_index("policy")


def step(n: int, title: str, kind: str, body: str, link: str, link_label: str):
    st.markdown(f"### {n} · {title}")
    ui.badge(kind)
    st.markdown(body)
    st.page_link(link, label=link_label, icon=":material/arrow_forward:")


# 1 ------------------------------------------------------------------
with st.container(border=True):
    a, b = st.columns([3, 2], vertical_alignment="center")
    with a:
        step(1, "The model is made of many small \"experts\"", "checkpoint",
             "OLMoE is a *Mixture-of-Experts* model. Inside it are **16 layers**, and each layer has **64 small "
             "networks called experts** — 1,024 in total. For every word it reads, a small \"router\" picks only "
             "**8 of the 64 experts** in each layer to do the work.\n\n"
             "*Analogy: a hospital with 16 floors and 64 specialists per floor. Each patient sees 8 specialists on "
             "every floor, chosen by a receptionist.*",
             "app_pages/model.py", "See inside the model")
    with b:
        with st.container(horizontal=True):
            st.metric("Layers", 16, border=True)
            st.metric("Experts per layer", 64, border=True)
        with st.container(horizontal=True):
            st.metric("Used per word, per layer", 8, border=True)
            st.metric("Size of one expert", "12.6 MB", border=True)

# 2 ------------------------------------------------------------------
d = D.demand(r).unstack()
vals = np.sort(d.to_numpy(float), axis=1)
top8_share = float((vals[:, -8:].sum(axis=1) / vals.sum(axis=1)).mean())
layer0 = d.loc[0].sort_values(ascending=False).to_numpy()
with st.container(border=True):
    a, b = st.columns([3, 2], vertical_alignment="center")
    with a:
        step(2, "Some experts are much busier than others", "measured",
             f"We ran the real model on **{r.request_id.nunique()} questions** (about AI, coding and general topics) "
             "and recorded which experts the router picked.\n\n"
             f"The work is uneven: in a typical layer, the **8 busiest experts (out of 64) do {top8_share:.0%} of "
             "all the work**, and some experts are hardly used. Different kinds of questions also prefer "
             "different experts.",
             "app_pages/routing.py", "Explore which experts are busy")
    with b:
        fig = go.Figure(go.Bar(x=list(range(1, 65)), y=layer0, marker_color=[ui.TEAL] * 8 + [ui.SLATE] * 56))
        fig.update_layout(title="Layer 0: experts sorted from busiest to least busy",
                          xaxis_title="Expert (ranked)", yaxis_title="Times chosen", showlegend=False)
        ui.chart(fig, "Expert usage in layer 0, sorted; a few experts take most of the work", height=260)

# 3 ------------------------------------------------------------------
cut = ui.pct(held.loc["demand_aware", "peak_test_load"], held.loc["static", "peak_test_load"])
with st.container(border=True):
    a, b = st.columns([3, 2], vertical_alignment="center")
    with a:
        step(3, "Smart placement keeps any one GPU from being overloaded", "simulated",
             "Imagine the 1,024 experts spread over **4 GPUs** of different speeds. The simple way is to deal them "
             "out in order, like cards (**static**). Our method (**demand-aware**) puts busy experts on the GPUs that "
             "can handle them, using the measurements from step 2.\n\n"
             f"On questions it had **not** seen before, the busiest GPU's workload drops by **{cut:.0f}%**. "
             "*(Simulation with assumed GPU speeds — not a real GPU test.)*",
             "app_pages/placement.py", "Try the placement planner")
    with b:
        labels = {"static": "Simple (static)", "balanced": "Balanced only", "demand_aware": "Our method"}
        fig = go.Figure(go.Bar(x=[labels[s] for s in held.index], y=held.peak_test_load,
                               marker_color=[ui.STRATEGY_COLORS[s] for s in held.index],
                               text=[f"{v:.0f}" for v in held.peak_test_load], textposition="outside"))
        fig.update_layout(title="Workload on the busiest GPU (lower is better)", yaxis_title="Workload units",
                          yaxis_range=[0, held.peak_test_load.max() * 1.2])
        ui.chart(fig, "Busiest GPU workload for three placement methods; ours is lowest", height=260)

# 4 ------------------------------------------------------------------
with st.container(border=True):
    a, b = st.columns([3, 2], vertical_alignment="center")
    with a:
        step(4, "Rearranging experts has a cost — so a controller decides", "simulated",
             "A running system starts with the simple layout. Switching to the better one means **copying 776 of the "
             "1,024 experts** (about **9.8 GB**) between GPUs, which slows down requests while it happens.\n\n"
             "A **controller** checks, using past questions only, whether moving is worth it. In the saved scenario "
             f"it said **{dec.decision.upper()}**: overall time would improve a little "
             f"({dec.forecast_makespan_gain_pct:+.1f}%), but the slowest request would get much slower "
             f"({dec.forecast_p95_regression_pct:+.0f}%). Checking on new questions confirmed that moving would "
             "have made the slowest request worse — so saying no was right.",
             "app_pages/migration.py", "Try the migration controller")
    with b:
        st.metric("Experts that must move", "776 of 1,024", border=True)
        st.metric("Data to copy", "≈ 9.8 GB", border=True)
        st.metric("Controller decision", dec.decision.upper(), border=True)

# 5 ------------------------------------------------------------------
cal = ui.phase4f("calibrated").pivot(index="bandwidth_gb_per_s", columns="policy", values="p95_latency_ms")
imp = 100 * (cal["static"] - cal["migrate"]) / cal["static"]
x0 = ui.crossover(imp.index.to_numpy(), imp.to_numpy())
with st.container(border=True):
    a, b = st.columns([3, 2], vertical_alignment="center")
    with a:
        step(5, "Whether moving pays off depends on network speed", "simulated",
             "Copying 9.8 GB is quick on a fast connection and slow on a slow one. We simulated four network speeds.\n\n"
             f"Below about **{x0:.0f} GB/s**, moving the experts makes the slowest request **worse**; above it, "
             "moving makes it **better**. *(GB/s = gigabytes per second.)*",
             "app_pages/network.py", "See the network results")
    with b:
        fig = go.Figure(go.Scatter(x=imp.index, y=imp.values, mode="lines+markers", line=dict(color=ui.TEAL, width=3)))
        fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", annotation_text="no change")
        fig.update_xaxes(type="log", title="Network speed (GB/s)", tickvals=list(imp.index))
        fig.update_layout(title="Effect of moving on the slowest request (above 0 = better)", yaxis_title="Change (%)")
        ui.chart(fig, "Moving experts helps only at higher network speeds", height=260)

# ------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True, height="stretch"):
        st.markdown("#### :material/science: What is real")
        st.markdown("- The OLMoE model and its 1,024 experts\n- Which experts were picked for 6 real questions\n"
                    "- How long the model took on your Mac\n- The size of each expert (from the model files)")
with c2:
    with st.container(border=True, height="stretch"):
        st.markdown("#### :material/model_training: What is simulated")
        st.markdown("- The GPUs, their speeds and the network between them\n- All latency / time numbers\n"
                    "- Moving experts between GPUs\n- Nothing is run on real GPUs; nothing on this dashboard changes "
                    "real hardware")

st.info("**Main result in one sentence:** using measured expert demand to place experts lowers the busiest GPU's load "
        f"by about {cut:.0f}% in simulation, and a cost-aware controller avoids moving experts when the network is too "
        "slow for the move to pay off.", icon=":material/lightbulb:")
ui.step_footer("app_pages/start.py")
