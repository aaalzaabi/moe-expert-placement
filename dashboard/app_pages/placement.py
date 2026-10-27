import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import C, P

ui.page_header("Where should experts go?",
               "Decide which simulated GPU each of the 1,024 experts lives on, and compare three ways of doing it.",
               ("derived", "live"), step="Part 2 · Step 3 of 8")
ui.explain(
    "We pretend to have **4 GPUs with different speeds** (assumed: 1.0, 1.3, 0.8 and 1.6) and place all 1,024 experts "
    "on them in three ways:\n\n"
    "- **Static** — deal experts out in order, like cards (the simple baseline).\n"
    "- **Load-balanced** — spread busy experts evenly, but ignore GPU speed.\n"
    "- **Demand-aware (our method)** — put busy experts on GPUs that can handle them, using demand *and* speed.\n\n"
    "Plans are made from questions 0, 2, 4 and **checked on questions 1, 3, 5** that the planner never saw.",
    ["**Top numbers:** the workload on the busiest GPU for each method. Lower = better (no GPU is a bottleneck).",
     "**Left chart:** each GPU's workload. Even bars = well balanced.",
     "**Right chart:** how much expert memory each GPU holds.",
     "**Colourful grid:** every expert (row = layer, column = expert) coloured by the GPU it goes to. "
     "Pick a GPU below it to list its experts.",
     "'Workload' is a rough estimate (work ÷ GPU speed), not a real GPU usage %."],
    "Our demand-aware method gives the busiest GPU the least work, so the cluster is better balanced.")

st.page_link("app_pages/gpu_sim.py", label="Prefer pictures? Watch the experts being placed and a question running on the GPUs (visual simulator)", icon=":material/play_circle:")

cfg = ui.scenario()

# ---------------------------------------------------------------- cluster editor
with st.expander("Change the simulated setup (optional)", icon=":material/tune:"):
    st.caption("Number of GPUs, their speeds and memory. Changes apply to every simulation page.")
    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        n = st.number_input("Simulated GPUs", 2, 8, cfg.num_gpus, key="pl_n")
        demand_source = st.segmented_control(
            "Demand used to plan", ["train", "all"], default="train", key="pl_src",
            format_func=lambda v: "Train requests" if v == "train" else "All 6 (in-sample)",
            help="Train = requests 0, 2, 4 (held-out evaluation stays clean). All = Phase 3A style, leaks test demand.")
    with c2:
        speeds = list(cfg.gpu_speeds)[:n] + [1.0] * max(0, n - cfg.num_gpus)
        edited = st.data_editor(pd.DataFrame({"gpu_id": range(n), "relative_speed": speeds}), hide_index=True,
                                disabled=["gpu_id"], key=f"pl_speeds_{n}", width="stretch",
                                column_config={"relative_speed": st.column_config.NumberColumn(
                                    "relative speed", min_value=0.1, max_value=4.0, step=0.1, format="%.2f",
                                    help="Hypothetical multiplier; 1.0 = baseline GPU")})
    with c3:
        auto_slots = math.ceil(C.TOTAL_EXPERT_SLOTS / n)
        slots = st.number_input("Expert slots per GPU", auto_slots, C.TOTAL_EXPERT_SLOTS,
                                max(cfg.capacity_slots, auto_slots) if n == cfg.num_gpus else auto_slots, key=f"pl_slots_{n}",
                                help=f"At least {auto_slots} so 1,024 experts fit")
        use_mem = st.toggle("Enforce expert-weight memory budget", value=cfg.gpu_memory_gb is not None, key="pl_mem_on")
        mem = st.number_input("Budget per GPU (GB)", 0.5, 192.0, float(cfg.gpu_memory_gb or 8.0), 0.5,
                              key="pl_mem", disabled=not use_mem,
                              help="Expert weights only (resident + staged during migration). Excludes activations, "
                                   "KV cache and the ~0.95 GB of shared non-expert weights.")
        expert_choice = st.segmented_control("Expert size", ["checkpoint", "legacy"], key="pl_esize",
                                             default="checkpoint" if cfg.expert_bytes == C.CHECKPOINT_EXPERT_BYTES else "legacy",
                                             format_func=lambda v: "12.583 MB (checkpoint BF16)" if v == "checkpoint" else "8 MB (placeholder)")

new = ui.set_scenario(
    gpu_speeds=tuple(float(s) for s in edited.relative_speed),
    placement_capacity_slots=int(slots),
    staging_capacity_slots=cfg.staging_slots if int(slots) == cfg.capacity_slots else 2 * int(slots),
    gpu_memory_gb=float(mem) if use_mem else None,
    expert_bytes=C.CHECKPOINT_EXPERT_BYTES if expert_choice != "legacy" else C.LEGACY_EXPERT_BYTES,
    gateway_gpu=min(cfg.gateway_gpu, n - 1))
if new is None:
    st.stop()
cfg = new
if demand_source == "all":
    st.warning("Planning with all six requests uses the held-out requests' demand, so held-out metrics below are "
               "in-sample (optimistic).", icon=":material/warning:")

placements, layouts = ui.layouts_for(cfg, demand_source or "train")
summ = P.summary(placements, "test_demand", cfg)
gpu = pd.concat([P.gpu_metrics(placements, s, "test_demand", cfg).assign(strategy=s) for s in C.STRATEGIES])

# ---------------------------------------------------------------- comparison
static_peak = summ.set_index("strategy").loc["static", "peak_load"]
with st.container(horizontal=True):
    for row in summ.itertuples():
        st.metric(f"{ui.STRATEGY_LABEL[row.strategy]} · peak load", f"{row.peak_load:,.0f}",
                  None if row.strategy == "static" else f"{-ui.pct(row.peak_load, static_peak):+.1f}% vs static",
                  delta_color="inverse", border=True,
                  help=f"Held-out (test) assignments on the busiest GPU ÷ speed. Imbalance (max/mean) {row.imbalance:.3f}.")
    st.metric("Experts moved static → demand-aware", int(len(P.moved_experts(placements, "static", "demand_aware"))),
              border=True)

with st.container():
    with st.container(border=True):
        ui.section("Normalized load per GPU (held-out requests)", "live",
                   caption="Assignments ÷ relative speed — a proxy for how long each GPU would be busy.")
        fig = go.Figure()
        for s in C.STRATEGIES:
            d = gpu[gpu.strategy == s]
            fig.add_bar(name=ui.STRATEGY_LABEL[s], x=[f"GPU {g}" for g in d.gpu_id], y=d.normalized_load,
                        marker_color=ui.STRATEGY_COLORS[s])
        fig.update_layout(barmode="group", yaxis_title="Load proxy (assignments / speed)")
        ui.chart(fig, "Per-GPU normalized load for each placement strategy", height=330)
with ui.more("memory per GPU and the full comparison table"):
    with st.container():
        with st.container(border=True):
            ui.section("Expert-weight residency and memory feasibility", "live",
                       caption="Resident expert weights per GPU. The dashed line is the staging peak during migration.")
            fig = go.Figure()
            for s in C.STRATEGIES:
                d = gpu[gpu.strategy == s]
                fig.add_bar(name=ui.STRATEGY_LABEL[s], x=[f"GPU {g}" for g in d.gpu_id], y=d.weight_gb,
                            marker_color=ui.STRATEGY_COLORS[s])
            if cfg.gpu_memory_gb:
                fig.add_hline(y=cfg.gpu_memory_gb, line_color=ui.RED, annotation_text="memory budget")
            fig.update_layout(barmode="group", yaxis_title="Expert weights (GB)")
            ui.chart(fig, "Expert weight memory per GPU", height=330)

    with st.container(border=True):
        ui.section("Strategy comparison", "live")
        show = summ.copy()
        show["strategy"] = show.strategy.map(ui.STRATEGY_LABEL)
        show["peak vs static"] = [f"{-ui.pct(v, static_peak):+.1f}%" for v in summ.peak_load]
        ui.table(show.round(3), "Placement strategy comparison", column_config={
            "peak_load": st.column_config.NumberColumn("peak load proxy", format="%.1f"),
            "feasible": st.column_config.CheckboxColumn("feasible (slots & memory)")})
        st.caption("Static = round-robin by expert index. Load-balanced = greedy by raw demand, ignores GPU speed. "
                   "Demand-aware = greedy on (load + demand) ÷ speed. Both greedy methods use training demand only.")

# ---------------------------------------------------------------- method
with st.expander("How the experts are distributed — the method, step by step", icon=":material/school:"):
    st.markdown("""
**Goal:** give every GPU an amount of work that matches its speed, so no single GPU becomes the bottleneck.

**Input:** for each of the 1,024 experts, how many times it was chosen on the **training questions (0, 2, 4)** —
its *demand*. The held-out questions (1, 3, 5) are kept aside for checking.

**Demand-aware algorithm (a greedy "biggest job first" rule):**
1. Sort all 1,024 experts from **most used to least used**.
2. Take the next expert. For each GPU that still has a free slot, compute
   **(work already on that GPU + this expert's demand) ÷ GPU speed** — i.e. how busy that GPU *would* become.
3. Put the expert on the GPU where that number is **smallest**. A fast GPU (speed 1.6) can take more work before
   it looks busy than a slow one (speed 0.8).
4. Repeat until every expert is placed. Each GPU holds at most 256 experts (its memory slots).

**The two baselines it is compared with:**
- **Static:** expert number *i* goes to GPU *i mod 4* — dealt out like cards, ignoring demand and speed.
- **Load-balanced:** the same greedy rule, but ignoring GPU speed (it balances raw work, not time).

**Tiny example** (computed live: 2 GPUs with speeds 1.0 and 2.0, six experts):
""")
    ex_d, ex_s = [6, 5, 4, 3, 2, 2], [1.0, 2.0]
    loads, rows_ex = [0.0, 0.0], []
    for i, dmd in enumerate(ex_d):
        cand = [(loads[g] + dmd) / ex_s[g] for g in range(2)]
        g = min(range(2), key=lambda k: (cand[k], k))
        rows_ex.append({"step": i + 1, "expert demand": dmd, "GPU 0 would be (speed 1)": round(cand[0], 2),
                        "GPU 1 would be (speed 2)": round(cand[1], 2), "placed on": f"GPU {g}"})
        loads[g] += dmd
    ui.table(pd.DataFrame(rows_ex), "Worked example of the greedy placement rule")
    st.caption(f"Result: GPU 0 holds work {loads[0]:g} (time {loads[0] / ex_s[0]:g}); GPU 1 holds work {loads[1]:g} "
               f"(time {loads[1] / ex_s[1]:g}). The faster GPU gets about twice the work, so both finish at similar times.")

# ---------------------------------------------------------------- validation vs random
with st.container(border=True):
    ui.section("Is this better than random guessing?", "live",
               caption="We placed the experts randomly 10,000 times (same 256-slot limit) and measured the busiest "
                       "GPU's load on the held-out questions each time. A real method must beat almost all of them.")
    val = ui.validate_vs_random(ui.cfg_key(cfg), demand_source or "train")
    vt = val["table"].set_index("strategy")
    da = vt.loc["demand_aware"]
    v1, v2, v3 = st.columns(3)
    v1.metric("Random placements beaten by demand-aware", f"{da.beats_random_pct:.1f}%", border=True,
              help="Share of the 10,000 random placements whose busiest GPU is MORE loaded than ours.")
    v2.metric("Best of 10,000 random placements", f"{val['random_best']:,.0f}", border=True,
              help=f"Ours: {da.peak_load:,.0f}. Lower is better.")
    v3.metric("Distance from the best possible", f"{da.gap_to_best_pct:.1f}%", border=True,
              help=f"Best possible = {val['lower_bound']:,.0f}: all work split exactly by GPU speed. No placement can "
                   "beat it.")
    fig = go.Figure(go.Histogram(x=val["random"], nbinsx=60, marker_color="#334155", name="random placements"))
    for name, x, color in [("Static", vt.loc["static", "peak_load"], ui.SLATE),
                           ("Load-balanced", vt.loc["balanced", "peak_load"], ui.BLUE),
                           ("Demand-aware (ours)", da.peak_load, ui.TEAL),
                           ("Best possible", val["lower_bound"], ui.GREEN)]:
        side = {"Best possible": "top left", "Demand-aware (ours)": "top right", "Load-balanced": "top left"}.get(name, "top right")
        fig.add_vline(x=x, line_color=color, line_width=3, line_dash="dot" if name == "Best possible" else "solid",
                      annotation_text=name, annotation_position=side, annotation_font_color=color)
    fig.update_layout(xaxis_title="Busiest GPU's load on held-out questions (lower = better)",
                      yaxis_title="Number of random placements", showlegend=False)
    ui.chart(fig, "Our placement is far left of all random placements and close to the best possible", height=340)
    verdict = ("**Not luck:** demand-aware beats every random placement" if da.beats_random_pct >= 99.9 else
               f"Demand-aware beats {da.beats_random_pct:.1f}% of random placements")
    st.markdown(f"{verdict} and lands within **{da.gap_to_best_pct:.1f}%** of the best possible value, on questions "
                f"it never saw. Static ({vt.loc['static', 'beats_random_pct']:.0f}%) and load-balanced "
                f"({vt.loc['balanced', 'beats_random_pct']:.0f}%) are no better than a typical random layout here, "
                "because they ignore GPU speed.")
    st.caption("Other checks that it is correct, not guessed: the planner never sees the held-out questions (a test "
               "changes them and confirms the placement does not change); every expert is placed exactly once within "
               "the slot limit; and the saved results are regenerated exactly by the test suite.")

with ui.more("which GPU every expert is on (assignment matrix)"):
    # ---------------------------------------------------------------- assignment matrix
    with st.container(border=True):
        strat = st.segmented_control("Inspect placement", list(C.STRATEGIES), default="demand_aware",
                                     format_func=ui.STRATEGY_LABEL.get, key="pl_strat") or "demand_aware"
        ui.section(f"Expert → GPU assignment matrix · {ui.STRATEGY_LABEL[strat]}", "live",
                   caption="Rows are layers, columns are experts; colour = simulated GPU.")
        rows = placements[placements.strategy == strat]
        mat = rows.pivot(index="layer_id", columns="expert_id", values="gpu_id")
        colors = ui.GPU_COLORS[:cfg.num_gpus]
        scale = []
        for i, c in enumerate(colors):
            scale += [[i / cfg.num_gpus, c], [(i + 1) / cfg.num_gpus, c]]
        fig = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, colorscale=scale, zmin=-0.5,
                                   zmax=cfg.num_gpus - 0.5, xgap=1, ygap=1,
                                   colorbar=dict(title="GPU", tickvals=list(range(cfg.num_gpus))),
                                   hovertemplate="expert (L%{y}, E%{x}) → GPU %{z}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", title="Layer", dtick=1)
        fig.update_xaxes(title="Expert", dtick=8)
        ui.chart(fig, f"Assignment of 1024 experts to GPUs for {strat}", height=430)

        g1, g2 = st.columns([1, 3])
        with g1:
            gsel = st.selectbox("GPU", range(cfg.num_gpus), key="pl_gpu", format_func=lambda g: f"GPU {g} (speed {cfg.gpu_speeds[g]:g})")
            mine = rows[rows.gpu_id == gsel]
            st.metric("Experts", len(mine), border=True)
            st.metric("Train demand", int(mine.train_demand.sum()), border=True)
            st.metric("Weights", f"{len(mine) * cfg.expert_bytes / C.GB:.2f} GB", border=True)
        with g2:
            lst = mine.assign(expert=[f"({l}, {e})" for l, e in zip(mine.layer_id, mine.expert_id)])
            ui.table(lst[["expert", "layer_id", "expert_id", "train_demand", "test_demand"]]
                     .sort_values("train_demand", ascending=False), f"Experts assigned to GPU {gsel}", height=300)

st.caption("Placement capacity counts expert slots; memory budget counts bytes. The simulator never places weights on real GPUs.")

ui.step_footer("app_pages/placement.py")
