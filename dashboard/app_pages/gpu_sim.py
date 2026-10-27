import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import gpu_viz as V
import ui
from ui import C

ui.page_header("Watch the GPUs at work",
               "A visual simulation: see the 1,024 experts being placed on the GPUs, then watch a real question run "
               "through them layer by layer.", ("derived", "live"), step="Part 2 · Visual simulator (goes with step 3)")
ui.explain(
    "Each **big square grid is one simulated GPU**, and each **small square is a slot** that can hold one expert "
    "(256 slots per GPU). You can watch two things:\n\n"
    "1. **Placing the experts.** The placement rule fills the GPUs one expert at a time, busiest experts first.\n"
    "2. **Running a question.** A real question from the OLMoE measurements goes through the 16 layers. In each "
    "layer, the experts it uses light up on whichever GPU holds them.",
    ["Press **▶ Play** under a picture to animate it, or drag the slider to go step by step.",
     "**Top part:** the GPUs. Colour in part 1 = how busy the expert is (brighter = busier). "
     "In part 2, lit squares are the experts this layer uses, coloured by GPU.",
     "**Bottom bars:** how much work each GPU has, divided by its speed (≈ how long it would be busy). "
     "Bars of similar height = well balanced.",
     "Switch the **placement method** at the top to compare. GPU speeds are assumed, not measured."],
    "With our demand-aware method the bars end up almost level, so no GPU holds up the others. With the simple "
    "static method one GPU ends up far busier than the rest.")

cfg = ui.scenario()
cfg_json = ui.cfg_key(cfg)

strategy = st.segmented_control(
    "Placement method", list(C.STRATEGIES), default="demand_aware", key="gs_strategy",
    format_func=lambda s: {"static": "Static (simple)", "balanced": "Load-balanced",
                           "demand_aware": "Demand-aware (ours)"}[s]) or "demand_aware"
st.caption(f"Simulated cluster: {ui.scenario_summary(cfg)}. Change it under *Change the simulated setup* on page 3.")


@st.cache_data(show_spinner=False, max_entries=16)
def traces(cfg_json: str) -> dict[str, pd.DataFrame]:
    cfg = C.SimConfig.from_dict(json.loads(cfg_json))
    pl = ui.plan(cfg_json)
    return {s: V.placement_trace(pl, s, cfg) for s in C.STRATEGIES}


@st.cache_data(show_spinner=False, max_entries=16)
def placing_figure(cfg_json: str, strategy: str) -> go.Figure:
    cfg = C.SimConfig.from_dict(json.loads(cfg_json))
    tr = traces(cfg_json)
    ymax = max(V.busy_time(t, cfg).max() for t in tr.values())
    return V.fig_placement_animation(tr[strategy], cfg, ui.STRATEGY_LABEL[strategy], ymax=ymax)


@st.cache_data(show_spinner=False, max_entries=32)
def question(cfg_json: str, request_id: int) -> dict:
    cfg = C.SimConfig.from_dict(json.loads(cfg_json))
    pl = ui.plan(cfg_json)
    return {s: V.question_work(ui.routing(), request_id, pl, s, cfg) for s in C.STRATEGIES}


@st.cache_data(show_spinner=False, max_entries=32)
def question_figure(cfg_json: str, request_id: int, strategy: str) -> go.Figure:
    cfg = C.SimConfig.from_dict(json.loads(cfg_json))
    q = question(cfg_json, request_id)
    ymax = max(per.to_numpy().max() for _, per in q.values())
    active, per = q[strategy]
    return V.fig_question_animation(traces(cfg_json)[strategy], active, per, cfg,
                                    f"Question {request_id} · {ui.STRATEGY_LABEL[strategy]}", ymax=ymax)


tr = traces(cfg_json)
trace = tr[strategy]

# ---------------------------------------------------------------- 1. placing
with st.container(border=True):
    ui.section("1 · Watch the experts being placed", "live",
               caption="Uses the training questions' demand (0, 2, 4), exactly as the planner does. "
                       "Press ▶ Play — the picture starts empty and fills up. Outlined squares = empty slots; "
                       "grey-blue = an expert the training questions never used; brighter = busier.")
    ui.chart(placing_figure(cfg_json, strategy),
             f"Animation of {ui.STRATEGY_LABEL[strategy]} placing 1,024 experts into {cfg.num_gpus} GPUs", height=640,
             key=f"gs_place_{strategy}")

    finals = {s: V.busy_time(t, cfg) for s, t in tr.items()}
    ideal = float(trace.train_demand.sum() / sum(cfg.gpu_speeds))
    with st.container(horizontal=True):
        for s, t in finals.items():
            st.metric(f"{ui.STRATEGY_LABEL[s]} · busiest GPU", f"{t.max():,.0f}",
                      f"{100 * (t.max() / ideal - 1):+.1f}% above perfect balance", delta_color="inverse", border=True,
                      help="Busy time (training work ÷ speed) of the most loaded GPU after all experts are placed. "
                           "Perfect balance = total work ÷ total speed.")
        st.metric("Perfect balance", f"{ideal:,.0f}", border=True,
                  help="Every GPU busy for exactly the same time. No placement can do better.")

    with ui.more("look at one placement decision"):
        st.markdown("##### Look at one decision")
        st.caption("Pick a step to see the numbers the rule compared and why the expert went where it did.")
        c1, c2 = st.columns([1, 2])
        with c1:
            step = st.number_input("Step (1 = first expert placed)", 1, len(trace), 1, key=f"gs_step_{strategy}")
            st.markdown(V.RULE[strategy].format(n=cfg.num_gpus))
        row, dt = V.decision_table(trace, int(step), cfg)
        with c2:
            if strategy == "static":
                i = int(row.layer_id) * C.EXPERTS_PER_LAYER + int(row.expert_id)
                why = (f"It is expert number {i}, and {i} mod {cfg.num_gpus} = {i % cfg.num_gpus}, so it goes to "
                       f"**GPU {row.gpu_id}**" + ("" if row.gpu_id == i % cfg.num_gpus else " (the next GPU with a free slot)")
                       + ". Its demand is not looked at.")
            elif strategy == "balanced":
                why = (f"GPU {row.gpu_id} had the **least work so far** "
                       f"({dt.loc[row.gpu_id, 'work already there']:g}), so it gets this expert.")
            else:
                why = (f"Adding this expert to GPU {row.gpu_id} gives a busy time of "
                       f"**{dt.loc[row.gpu_id, 'busy time if placed here']:,.1f}** — the smallest of all GPUs.")
            st.markdown(f"**Step {int(step)}:** layer {row.layer_id}, expert {row.expert_id} "
                        f"(demand **{row.train_demand:g}**). {why}")
            ui.table(dt, f"Numbers compared at placement step {int(step)}", column_config={
                "speed": st.column_config.NumberColumn(format="%.2f×"),
                "work already there": st.column_config.NumberColumn(format="%.0f"),
                "busy time now": st.column_config.NumberColumn(format="%.1f"),
                "busy time if placed here": st.column_config.NumberColumn(format="%.1f",
                                                                          help="Empty = GPU is full (no free slot)"),
                "chosen": st.column_config.CheckboxColumn("chosen")})

# ---------------------------------------------------------------- 2. a question
r = ui.routing()
info = r.groupby("request_id").agg(category=("category", "first"), tokens=("input_tokens", "first"))
with st.container(border=True):
    ui.section("2 · Send a real question through the GPUs", "live",
               caption="Which experts each layer uses is the real OLMoE measurement. How long each GPU takes is "
                       "simulated: tokens ÷ assumed GPU speed.")
    ids = [i for i in cfg.test_ids] + [i for i in cfg.train_ids]
    rid = st.segmented_control(
        "Question", ids, default=ids[0], key="gs_q",
        format_func=lambda i: f"Q{i} · {info.loc[i, 'category']} · {'unseen' if i in cfg.test_ids else 'training'}")
    rid = ids[0] if rid is None else rid
    st.caption(f"Question {rid}: {info.loc[rid, 'category']} topic, {info.loc[rid, 'tokens']} words/tokens. "
               "Unseen = the placement was planned without it, so this is a fair test.")
    ui.chart(question_figure(cfg_json, rid, strategy),
             f"Animation of question {rid} flowing through 16 layers on the {ui.STRATEGY_LABEL[strategy]} placement",
             height=640, key=f"gs_q_{strategy}_{rid}")

    q = question(cfg_json, rid)
    summ = {s: V.question_summary(per) for s, (_, per) in q.items()}
    st.markdown("**How long the whole question takes** — each layer must wait for its slowest GPU before the next "
                "layer can start, so the total is the sum of the slowest GPU in every layer.")
    with st.container(horizontal=True):
        base = summ["static"]["total"]
        for s, m in summ.items():
            st.metric(f"{ui.STRATEGY_LABEL[s]} · total time", f"{m['total']:,.1f}",
                      None if s == "static" else f"{100 * (m['total'] / base - 1):+.1f}% vs static",
                      delta_color="inverse", border=True,
                      help=f"Sum over 16 layers of the slowest GPU's work ÷ speed. GPUs spend "
                           f"{m['waiting_pct']:.0f}% of the time waiting for the slowest one.")
    with ui.more("time per layer for each placement"):
        st.markdown("**Time per layer for each placement** (lower = faster). The thick line is the method selected above.")
        fig = go.Figure()
        for s, m in summ.items():
            fig.add_scatter(x=np.arange(1, C.NUM_LAYERS + 1), y=m["layer_time"], mode="lines+markers",
                            name=ui.STRATEGY_LABEL[s], line=dict(color=ui.STRATEGY_COLORS[s], width=3 if s == strategy else 1.5))
        fig.update_layout(xaxis_title="Layer", yaxis_title="Slowest GPU's work (÷ speed)")
        fig.update_xaxes(dtick=1)
        ui.chart(fig, f"Per-layer time of question {rid} for three placements", height=320)

    m, best = summ[strategy], min(summ, key=lambda s: summ[s]["total"])
    waits = ", ".join(f"GPU {g}: {n} layers" for g, n in m["bottleneck_counts"].items() if n)
    st.markdown(f"With **{ui.STRATEGY_LABEL[strategy]}**, the GPUs spend **{m['waiting_pct']:.0f}%** of the time "
                f"waiting for the slowest GPU. Which GPU was the slowest: {waits}. "
                f"Fastest placement for this question: **{ui.STRATEGY_LABEL[best]}**.")
    st.caption("Simplified on purpose so it is easy to follow: computation only, layer by layer. It leaves out the "
               "network between GPUs, queues of several questions and copying experts. Page 4 runs the full "
               "simulator with all of those.")

st.divider()
c1, c2 = st.columns(2)
c1.page_link("app_pages/placement.py", label="Back to step 3: Where should experts go?", icon=":material/arrow_back:")
c2.page_link("app_pages/inference.py", label="Next: step 4 · How fast would requests run?",
             icon=":material/arrow_forward:")
