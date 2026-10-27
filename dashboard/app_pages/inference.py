import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import C

ui.page_header("How fast would requests run?",
               "A step-by-step simulation of the 3 test questions running on the simulated GPUs.",
               ("measured", "live"), step="Part 2 · Step 4 of 8")
ui.explain(
    "The simulator replays the **real expert choices** of the test questions on the simulated GPUs. For each of the "
    "16 layers, the work is sent to the GPUs holding the chosen experts, waits in line if a GPU is busy, is "
    "computed, and comes back. The next layer starts only when all experts of the current layer are done.",
    ["**Top numbers:** *p50* = typical request time, *p95* = the slow requests, *makespan* = when the last one "
     "finished. Lower is better. Times are simulated milliseconds.",
     "**GPU timeline:** each row is a GPU, each small bar is one piece of expert work; colour = which question.",
     "**Request progress:** each question moving through its 16 layers (yellow triangle = arrival).",
     "**Waiting time:** how long work waited for a busy GPU.",
     "**Bottom table:** the three placement methods compared. Change the placement in the setup panel above."],
    "With the demand-aware placement, requests finish sooner because work is spread better across the GPUs.")

cfg = ui.scenario()
with st.expander("Change the simulated setup (optional)", icon=":material/tune:"):
    c1, c2, c3, c4 = st.columns(4)
    strategy = c1.selectbox("Placement", C.STRATEGIES, index=2, format_func=ui.STRATEGY_LABEL.get, key="inf_strat")
    req_set = c1.segmented_control("Requests", ["test", "train"], default="test", key="inf_reqs",
                                   format_func=lambda v: "Held-out 1, 3, 5" if v == "test" else "Train 0, 2, 4")
    bw = c2.select_slider("Link bandwidth (GB/s)", [1.0, 3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0, 400.0],
                          value=cfg.bandwidth_gb_per_s if cfg.bandwidth_gb_per_s in
                          [1.0, 3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0, 400.0] else 12.0, key="inf_bw",
                          help="Gigabytes per second per link (not gigabits).")
    link_model = c2.segmented_control("Link model", list(C.LINK_MODELS), default=cfg.link_model, key="inf_link",
                                      help="directed: one queue per (src, dst) pair (original model). shared_port: "
                                           "each GPU has one send and one receive port, so transfers contend more.")
    gap = c3.number_input("Arrival gap (ms)", 0.0, 2000.0, float(cfg.arrival_gap_ms), 25.0, key="inf_gap")
    process = c3.segmented_control("Arrivals", list(C.ARRIVAL_PROCESSES), default=cfg.arrival_process, key="inf_proc")
    seed = c4.number_input("Seed (poisson)", 0, 10_000, int(cfg.seed), key="inf_seed", disabled=process != "poisson")
    compute = c4.number_input("Compute ms / assignment", 0.05, 5.0, float(cfg.compute_ms_per_assignment), 0.05,
                              key="inf_cms", help="Hypothetical cost of one token through one expert on a speed-1.0 GPU")

cfg = ui.set_scenario(bandwidth_gb_per_s=float(bw), link_model=link_model or cfg.link_model, arrival_gap_ms=float(gap),
                      arrival_process=process or cfg.arrival_process, seed=int(seed),
                      compute_ms_per_assignment=float(compute))
if cfg is None:
    st.stop()
ids = cfg.test_ids if req_set != "train" else cfg.train_ids
run_cfg = cfg.with_(source_strategy=strategy)
res = ui.simulate(ui.cfg_key(run_cfg), tuple(ids), False)
s = res.summary

with st.container(horizontal=True):
    st.metric("Simulated p50 latency", f"{s['p50_latency_ms']:.1f} ms", border=True)
    st.metric("Simulated p95 latency", f"{s['p95_latency_ms']:.1f} ms", border=True,
              help="Linear-interpolated 95th percentile of only 3 requests ≈ the slowest request.")
    st.metric("Makespan", f"{s['makespan_ms']:.1f} ms", border=True)
    st.metric("Communication volume", f"{s['total_link_mb']:.2f} MB", border=True,
              help="Activations to and from remote experts (4,096 B per assignment each way)")
    st.metric("Remote expert calls", f"{s['remote_expert_calls']:,}", border=True,
              help=f"Expert invocations not on gateway GPU {cfg.gateway_gpu}")

# ---------------------------------------------------------------- GPU Gantt
comp = res.compute
with st.container(border=True):
    ui.section("GPU compute timeline", "live",
               caption="Each bar = one expert's work for one request in one layer. Colour = request.")
    fig = go.Figure()
    for i, (rid, d) in enumerate(comp.groupby("request_id")):
        fig.add_bar(y=[f"GPU {g}" for g in d.gpu_id], x=d.end_ms - d.start_ms, base=d.start_ms, orientation="h",
                    name=f"request {rid}", marker_color=ui.GPU_COLORS[i % 8], marker_line_width=0,
                    customdata=d[["layer_id", "expert_id", "assignments", "start_ms", "end_ms"]],
                    hovertemplate="req " + str(rid) + " · expert (L%{customdata[0]}, E%{customdata[1]})<br>"
                                  "%{customdata[2]} assignments<br>%{customdata[3]:.2f} → %{customdata[4]:.2f} ms<extra></extra>")
    arrivals = res.requests
    for r in arrivals.itertuples():
        fig.add_vline(x=r.arrival_ms, line_dash="dot", line_color="#475569")
    fig.update_layout(barmode="overlay", xaxis_title="Simulated time (ms)", bargap=0.35)
    fig.update_yaxes(categoryorder="array", categoryarray=[f"GPU {g}" for g in reversed(range(cfg.num_gpus))])
    ui.chart(fig, "Gantt chart of expert compute on each simulated GPU", height=320)

with ui.more("request progress, queues, network links, all placements"):
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            ui.section("Request progress through 16 layers", "live",
                       caption="A layer starts only after every expert of the previous layer has returned.")
            lay = res.layers
            fig = go.Figure(go.Bar(y=[f"req {r}" for r in lay.request_id], x=lay.end_ms - lay.start_ms, base=lay.start_ms,
                                   orientation="h", marker=dict(color=lay.layer_id, colorscale=ui.HEAT, line_width=0.5,
                                                                line_color="#0b1220", colorbar=dict(title="layer")),
                                   customdata=lay[["layer_id", "start_ms", "end_ms"]],
                                   hovertemplate="layer %{customdata[0]}<br>%{customdata[1]:.2f} → %{customdata[2]:.2f} ms<extra></extra>"))
            for r in arrivals.itertuples():
                fig.add_scatter(x=[r.arrival_ms], y=[f"req {r.request_id}"], mode="markers", showlegend=False,
                                marker=dict(symbol="triangle-right", size=12, color=ui.AMBER), hovertext="arrival")
            fig.update_layout(xaxis_title="Simulated time (ms)", showlegend=False)
            ui.chart(fig, "Per-request layer timeline with arrival markers", height=300)
    with c2:
        with st.container(border=True):
            ui.section("Queueing delay per GPU", "live",
                       caption="Time an expert task waited for its GPU after its activations arrived.")
            comp = comp.assign(wait_ms=comp.start_ms - comp.inbound_ms)
            fig = go.Figure()
            for g, d in comp.groupby("gpu_id"):
                fig.add_box(y=d.wait_ms, name=f"GPU {g}", marker_color=ui.GPU_COLORS[g % 8], boxpoints=False)
            fig.update_layout(yaxis_title="Wait (ms)", showlegend=False)
            ui.chart(fig, "Distribution of GPU queueing delay", height=300)

    tr = res.transfers
    with st.container(border=True):
        ui.section("Network link activity", "live",
                   caption="Dispatch (gateway → expert GPU) and return transfers. Local experts on the gateway need no transfer.")
        if tr.empty:
            st.info("No remote transfers in this configuration.")
        else:
            tr = tr.assign(link=[f"{a}→{b}" for a, b in zip(tr.src_gpu, tr.dst_gpu)])
            vol = tr.groupby(["link", "kind"]).bytes.sum().reset_index()
            l1, l2 = st.columns([3, 2])
            with l1:
                fig = go.Figure()
                for kind, color in [("dispatch", ui.BLUE), ("return", ui.TEAL)]:
                    d = tr[tr.kind == kind]
                    fig.add_bar(y=d.link, x=d.end_ms - d.start_ms, base=d.start_ms, orientation="h", name=kind,
                                marker_color=color, marker_line_width=0)
                fig.update_layout(barmode="overlay", xaxis_title="Simulated time (ms)")
                ui.chart(fig, "Timeline of link transfers", height=300)
            with l2:
                fig = go.Figure()
                for kind, color in [("dispatch", ui.BLUE), ("return", ui.TEAL)]:
                    d = vol[vol.kind == kind]
                    fig.add_bar(x=d.link, y=d.bytes / C.MB, name=kind, marker_color=color)
                fig.update_layout(barmode="stack", yaxis_title="MB")
                ui.chart(fig, "Activation volume per link", height=300)

    with st.container(border=True):
        ui.section("All placements under this scenario", "live")
        rows = []
        for st_name in C.STRATEGIES:
            r = ui.simulate(ui.cfg_key(cfg.with_(source_strategy=st_name)), tuple(ids), False).summary
            rows.append({"placement": ui.STRATEGY_LABEL[st_name], **{k: r[k] for k in
                         ["p50_latency_ms", "p95_latency_ms", "makespan_ms", "total_link_mb", "remote_expert_calls"]}})
        ui.table(pd.DataFrame(rows).round(2), "Simulated latency for static, balanced and demand-aware placements")
        st.caption(ui.SIM_DISCLAIMER + " Prefill only; decode steps, attention compute and KV cache are not modelled.")

ui.step_footer("app_pages/inference.py")
