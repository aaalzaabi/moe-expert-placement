import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import C, D

ui.page_header("Which experts are busy?",
               "Real measurements: which experts the model's router picked when it read 6 questions on your Mac. "
               "Nothing on this page is simulated.", ("measured", "derived"), step="Part 1 · Step 2 of 8")
ui.explain(
    "We ran OLMoE on **6 questions** (2 about AI, 2 about coding, 2 general) and counted, for every one of the "
    "1,024 experts, **how many times it was chosen**. This 'expert demand' is the input for everything that follows.",
    ["**Big heatmap:** one small square per expert (rows = 16 layers, columns = 64 experts). "
     "Brighter = chosen more often. Bright spots are the busy experts.",
     "**Layer chart (left):** for one layer, compares how much AI, coding and general questions use each expert.",
     "**Imbalance chart:** how uneven the work is inside each layer (1.0 would be perfectly even).",
     "**Difference between workloads:** 0 = two topics use the same experts, 1 = completely different experts.",
     "**Filters** (optional) let you pick topics, questions or a layer."],
    "A few experts in each layer do most of the work, and different topics prefer different experts — so a smart "
    "placement can use this.")

runs = D.measurement_runs()
source_name = st.selectbox("Routing dataset", list(runs), key="rt_source",
                           help="New datasets appear here after running scripts/profile_routing.py")
source = None if runs[source_name] == D.OUTPUTS else str(runs[source_name])
routing = ui.routing(source)
tim = ui.timings(source)

# ---------------------------------------------------------------- filters
with st.expander("Filter the data (optional)", icon=":material/filter_alt:"):
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1], vertical_alignment="bottom")
    cats = sorted(routing.category.unique())
    sel_cats = f1.pills("Workload category", cats, selection_mode="multi", default=cats, key="rt_cats")
    req_options = sorted(routing[routing.category.isin(sel_cats or cats)].request_id.unique())
    sel_reqs = f2.multiselect("Request ID", req_options, default=req_options, key="rt_reqs")
    layer_sel = f3.selectbox("Layer focus", list(range(C.NUM_LAYERS)), key="rt_layer")
    per_token = f4.toggle("Per token", value=True, key="rt_norm", help="Divide by input tokens (prompts differ in length)")

if not sel_reqs:
    st.warning("Select at least one request.", icon=":material/warning:")
    st.stop()
sub = routing[routing.request_id.isin(sel_reqs)]
tokens = tim[tim.request_id.isin(sel_reqs)].input_tokens.sum()
agg = sub.groupby(["layer_id", "expert_id"]).token_assignments.sum().reset_index()
value = agg.token_assignments / tokens if per_token else agg.token_assignments
unit = "assignments per input token" if per_token else "token assignments"

with st.container(horizontal=True):
    st.metric("Requests", len(sel_reqs), border=True)
    st.metric("Input tokens", int(tokens), border=True)
    st.metric("Routing decisions", int(sub.token_assignments.sum()), border=True, help="tokens × 16 layers × 8")
    st.metric("Experts used ≥ once", f"{int((agg.token_assignments > 0).sum())} / 1024", border=True)
    st.metric("Never selected", int((agg.token_assignments == 0).sum()), border=True,
              help="With so few tokens, many experts are never selected; this is a sample-size effect.")

# ---------------------------------------------------------------- heatmap
with st.container(border=True):
    ui.section("Expert demand across all 16 layers", "measured", source="workload_routing.csv",
               caption="Each cell is one layer-specific expert (layer_id, expert_id).")
    mat = agg.assign(v=value).pivot(index="layer_id", columns="expert_id", values="v")
    fig = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, colorscale=ui.HEAT,
                               colorbar=dict(title=dict(text=unit, side="right")),
                               hovertemplate="expert (L%{y}, E%{x})<br>%{z:.3f}<extra></extra>"))
    fig.update_xaxes(title="Expert ID", dtick=8)
    fig.update_yaxes(title="Layer ID", autorange="reversed", dtick=1)
    ui.chart(fig, "Heatmap of expert demand for 16 layers by 64 experts", height=460)

with ui.more("per-layer charts, imbalance, per-question timing, expert drill-down"):
    c1, c2 = st.columns([3, 2])
    with c1:
        with st.container(border=True):
            ui.section(f"Layer {layer_sel}: demand by workload", "derived",
                       caption="Assignments per input token, per category; reveals workload-specific experts.")
            lay = sub[sub.layer_id == layer_sel].groupby(["category", "expert_id"]).token_assignments.sum().reset_index()
            cat_tok = tim[tim.request_id.isin(sel_reqs)].groupby("category").input_tokens.sum()
            lay["rate"] = lay.token_assignments / lay.category.map(cat_tok)
            fig = go.Figure()
            for cat in sorted(lay.category.unique()):
                d = lay[lay.category == cat]
                fig.add_bar(name=cat, x=d.expert_id, y=d.rate, marker_color=ui.CATEGORY_COLORS.get(cat, ui.SLATE))
            fig.update_layout(barmode="group", xaxis_title="Expert ID", yaxis_title="Assignments per token", bargap=0.15)
            ui.chart(fig, f"Per-category expert demand in layer {layer_sel}", height=330)
    with c2:
        with st.container(border=True):
            ui.section("Most-selected experts", "measured")
            top = agg.assign(**{unit: value}).sort_values("token_assignments", ascending=False).head(15)
            top.insert(0, "expert", [f"({l}, {e})" for l, e in zip(top.layer_id, top.expert_id)])
            ui.table(top[["expert", "token_assignments", unit]].round(3), "Top 15 layer-specific experts", height=330)

    # ---------------------------------------------------------------- imbalance + divergence
    def jsd(p, q):
        p, q = p / p.sum(), q / q.sum()
        m = 0.5 * (p + q)
        kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0))
        return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


    rows = []
    for layer, g in agg.groupby("layer_id"):
        v = g.token_assignments.to_numpy(float)
        rows.append({"layer_id": layer, "max_over_mean": v.max() / v.mean(), "cv": v.std() / v.mean(),
                     "share_top8": np.sort(v)[-8:].sum() / v.sum()})
    imb = pd.DataFrame(rows)

    div_rows = []
    cat_list = sorted(sub.category.unique())
    for layer in range(C.NUM_LAYERS):
        L = sub[sub.layer_id == layer].groupby(["category", "expert_id"]).token_assignments.sum().unstack(fill_value=0)
        for i, a in enumerate(cat_list):
            for b in cat_list[i + 1:]:
                if a in L.index and b in L.index:
                    div_rows.append({"layer_id": layer, "pair": f"{a} vs {b}", "jsd": jsd(L.loc[a].to_numpy(float), L.loc[b].to_numpy(float))})
    div = pd.DataFrame(div_rows)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            ui.section("Demand imbalance within each layer", "derived",
                       caption="Max ÷ mean assignments across the 64 experts of a layer. 1.0 = perfectly even.")
            fig = go.Figure(go.Bar(x=imb.layer_id, y=imb.max_over_mean, marker_color=ui.TEAL,
                                   customdata=imb[["cv", "share_top8"]],
                                   hovertemplate="layer %{x}<br>max/mean %{y:.2f}<br>CV %{customdata[0]:.2f}"
                                                 "<br>top-8 share %{customdata[1]:.0%}<extra></extra>"))
            fig.update_layout(xaxis_title="Layer", yaxis_title="Max / mean", xaxis_dtick=1)
            ui.chart(fig, "Per-layer expert demand imbalance", height=300)
    with c2:
        with st.container(border=True):
            ui.section("Expert-selection difference between workloads", "derived",
                       caption="Jensen–Shannon distance between categories' expert distributions (0 = identical, 1 = disjoint).")
            if div.empty:
                st.info("Select at least two categories.")
            else:
                fig = go.Figure()
                for i, (pair, d) in enumerate(div.groupby("pair")):
                    fig.add_scatter(x=d.layer_id, y=d.jsd, name=pair, mode="lines+markers",
                                    line=dict(color=[ui.TEAL, ui.BLUE, ui.VIOLET][i % 3]))
                fig.update_layout(xaxis_title="Layer", yaxis_title="JS distance", xaxis_dtick=1, yaxis_range=[0, 1])
                ui.chart(fig, "Workload divergence of expert selection per layer", height=300)

    # ---------------------------------------------------------------- per request
    with st.container(border=True):
        ui.section("Per-request routing metrics and forward-pass timing", "measured", source="workload_timings.csv")
        per = sub.groupby("request_id").apply(
            lambda g: pd.Series({"distinct_experts": int((g.token_assignments > 0).sum()),
                                 "max_expert_share": g.groupby("layer_id").token_assignments.max().mean() / g.input_tokens.iloc[0]}),
            include_groups=False).reset_index()
        per = tim[tim.request_id.isin(sel_reqs)].merge(per, on="request_id")
        per["split"] = np.where(per.request_id.isin(C.DEFAULT.train_ids), "train", "test")
        per["seconds_per_token"] = per.forward_seconds / per.input_tokens
        t1, t2 = st.columns([3, 2])
        with t1:
            ui.table(per[["request_id", "category", "split", "input_tokens", "forward_seconds", "seconds_per_token",
                          "distinct_experts", "max_expert_share"]].round(3), "Per-request routing and timing",
                     column_config={"max_expert_share": st.column_config.NumberColumn(
                         "top expert share", help="Mean over layers of the busiest expert's assignments ÷ tokens")})
            st.caption("Request 0 includes first-call warm-up on MPS (no warm-up pass was run). Timing is a single "
                       "Apple-silicon device in FP32, not distributed inference.")
        with t2:
            fig = go.Figure(go.Bar(x=[f"req {r}" for r in per.request_id], y=per.forward_seconds,
                                   marker_color=[ui.CATEGORY_COLORS.get(c, ui.SLATE) for c in per.category],
                                   text=[f"{t} tok" for t in per.input_tokens], textposition="outside"))
            fig.update_layout(yaxis_title="Forward pass (s)", showlegend=False)
            ui.chart(fig, "Measured forward-pass time per request", height=280)

    with st.container(border=True):
        ui.section("Expert drill-down", "measured")
        e = st.number_input("Expert ID in focus layer", 0, C.EXPERTS_PER_LAYER - 1, 0, key="rt_expert")
        d = sub[(sub.layer_id == layer_sel) & (sub.expert_id == e)]
        st.markdown(f"Expert `({layer_sel}, {e})` received **{int(d.token_assignments.sum())}** assignments from the selected requests.")
        ui.table(d[["request_id", "category", "token_assignments", "input_tokens"]], "Assignments per request for the chosen expert")

st.caption("Measured routing ≠ simulated placement: GPU assignments and latencies on other pages are simulation outputs "
           "that use this routing as input.")

ui.step_footer("app_pages/routing.py")
