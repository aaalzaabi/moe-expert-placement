from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import C, P, POL

ui.page_header("Should we move experts?",
               "A running system uses the simple layout. Moving to the better layout means copying experts — is it worth it?",
               ("live", "simulated"), step="Part 3 · Step 5 of 8")
ui.explain(
    "Switching from the simple layout to our better one means **copying 776 experts (≈ 9.8 GB)** between GPUs. "
    "While copying, the network is busy and requests slow down. A **controller** decides whether the move is worth "
    "it. It only looks at past questions (0, 2, 4) to decide; questions 1, 3, 5 are used afterwards to check if it "
    "was right.",
    ["**Top numbers:** how many experts must move, how much data, and how long the copy takes.",
     "**Copies grid:** how many experts travel from each GPU to each other GPU.",
     "**Controller box:** its recommendation (APPROVE or REJECT) and the three checks behind it — green PASS / red FAIL.",
     "**Sandbox:** try it yourself. *Follow controller*, *Approve* or *Reject*, then watch the stages "
     "(Proposed → Approved → Transferring → Active). This is only a simulation in your browser.",
     "Open *Change the simulated setup* to try a faster network or to test a copy failure (rollback)."],
    "At the default network speed the controller says REJECT: the move would make the slowest request much slower, "
    "and the check on new questions agrees.")

cfg = ui.scenario()

# ---------------------------------------------------------------- controls
with st.expander("Change the simulated setup (optional)", icon=":material/tune:"):
    st.caption("Network speed, expert size, decision thresholds and failure testing.")
    c1, c2, c3, c4 = st.columns(4)
    bw = c1.number_input("Link bandwidth (GB/s)", 0.5, 1000.0, float(cfg.bandwidth_gb_per_s), 1.0, key="mg_bw",
                         help="GigaBYTES per second per directed link. 100 Gbps Ethernet ≈ 12.5 GB/s.")
    esize = c1.segmented_control("Expert size", ["checkpoint", "legacy"], key="mg_es",
                                 default="checkpoint" if cfg.expert_bytes == C.CHECKPOINT_EXPERT_BYTES else "legacy",
                                 format_func=lambda v: "12.583 MB" if v == "checkpoint" else "8 MB")
    start = c2.number_input("Migration approved at (ms)", 0.0, 2000.0, float(cfg.migration_start_ms), 10.0, key="mg_start")
    staging = c2.number_input("Staging slots per GPU", cfg.capacity_slots, 4 * cfg.capacity_slots,
                              max(cfg.staging_slots, cfg.capacity_slots), key="mg_stage",
                              help="Resident + incoming copies allowed while both layouts coexist")
    min_gain = c3.number_input("Approve if makespan gain ≥ (%)", -50.0, 50.0, float(cfg.min_makespan_gain_pct), 0.5, key="mg_gain")
    max_reg = c3.number_input("…and p95 regression ≤ (%)", -50.0, 200.0, float(cfg.max_p95_regression_pct), 1.0, key="mg_reg")
    inject = c4.toggle("Inject copy failure", value=cfg.fail_after_copies is not None, key="mg_fail_on",
                       help="Fault injection: the migration aborts after N successful copies and rolls back")
    fail_n = c4.number_input("Fail after N copies", 0, 2000, int(cfg.fail_after_copies or 300), key="mg_fail_n",
                             disabled=not inject)

cfg = ui.set_scenario(bandwidth_gb_per_s=float(bw), migration_start_ms=float(start), staging_capacity_slots=int(staging),
                      min_makespan_gain_pct=float(min_gain), max_p95_regression_pct=float(max_reg),
                      expert_bytes=C.LEGACY_EXPERT_BYTES if esize == "legacy" else C.CHECKPOINT_EXPERT_BYTES,
                      fail_after_copies=int(fail_n) if inject else None)
if cfg is None:
    st.stop()
key = ui.cfg_key(cfg)
placements, layouts = ui.layouts_for(cfg)
moved = P.moved_experts(placements, cfg.source_strategy, cfg.target_strategy)
policies, requests, decision, _ = ui.controller(key)

# ---------------------------------------------------------------- current vs proposed
st.subheader("Current vs proposed placement")
with st.container(horizontal=True):
    st.metric("Experts requiring relocation", f"{len(moved)} / 1024", border=True)
    st.metric("Transfer volume", f"{len(moved) * cfg.expert_bytes / C.GB:.2f} GB", border=True,
              help=f"{len(moved)} × {cfg.expert_mb:.3f} MB")
    fm = decision.forecast_migrate
    st.metric("Forecast activation time", f"{(fm['migration_activation_ms'] or np.nan) - cfg.migration_start_ms:.1f} ms",
              border=True, help="From approval until every copy has landed (training-request forecast, with inference traffic)")
    worst = max(1, moved.groupby(["src_gpu", "dst_gpu"]).size().max() if len(moved) else 1)
    st.metric("Busiest link", f"{worst} copies", border=True,
              help=f"≈ {worst * C.transfer_ms(cfg.expert_bytes, cfg.bandwidth_gb_per_s, cfg.link_latency_ms):.1f} ms of serial copying")

with ui.more("load per GPU and copies per link"):
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            ui.section("Load per GPU: current vs proposed", "live")
            fig = go.Figure()
            for s_name in (cfg.source_strategy, cfg.target_strategy):
                m = P.gpu_metrics(placements, s_name, "train_demand", cfg)
                fig.add_bar(name=f"{'current' if s_name == cfg.source_strategy else 'proposed'} · {ui.STRATEGY_LABEL[s_name]}",
                            x=[f"GPU {g}" for g in m.gpu_id], y=m.normalized_load, marker_color=ui.STRATEGY_COLORS[s_name])
            fig.update_layout(barmode="group", yaxis_title="Train load proxy (assignments / speed)")
            ui.chart(fig, "Load proxy per GPU, current versus proposed", height=300)
    with c2:
        with st.container(border=True):
            ui.section("Copies per link (source → destination GPU)", "live")
            mat = (moved.groupby(["src_gpu", "dst_gpu"]).size().unstack(fill_value=0)
                   .reindex(index=range(cfg.num_gpus), columns=range(cfg.num_gpus), fill_value=0))
            fig = go.Figure(go.Heatmap(z=mat.values, x=[f"→ GPU {g}" for g in mat.columns], y=[f"GPU {g} →" for g in mat.index],
                                       colorscale=ui.HEAT, text=mat.values, texttemplate="%{text}",
                                       hovertemplate="%{y} %{x}: %{z} experts<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            ui.chart(fig, "Number of expert copies on each directed link", height=300)

# ---------------------------------------------------------------- controller decision
st.subheader("Cost-aware controller")
st.caption("The controller decides using only the past (training) questions "
           f"{', '.join(map(str, cfg.train_ids))}. The held-out questions {', '.join(map(str, cfg.test_ids))} are used "
           "only afterwards, to check whether the decision was right.")
explanation = ui.explain_decision(key)
d1, d2 = st.columns([3, 2])
with d1:
    ui.show_explanation(explanation)
with d2:
    with st.container(border=True, height="stretch"):
        st.markdown("#### What happens to each question if we move")
        per = explanation.per_request
        fig = go.Figure()
        fig.add_bar(name="don't move", x=[f"Q{r}" for r in per.request_id], y=per.latency_ms_never, marker_color=ui.SLATE)
        fig.add_bar(name="move", x=[f"Q{r}" for r in per.request_id], y=per.latency_ms_migrate, marker_color=ui.AMBER,
                    text=[f"{c:+.0f}%" + (" · during copy" if d else "") for c, d in zip(per.change_pct, per.running_during_copy)],
                    textposition="outside")
        fig.update_layout(barmode="group", yaxis_title="Simulated time per question (ms)",
                          yaxis_range=[0, max(per.latency_ms_migrate.max(), per.latency_ms_never.max()) * 1.25])
        ui.chart(fig, "Per-question time with and without moving, training questions", height=300)
        st.caption("Training questions only (what the controller sees). 'during copy' = the question was running "
                   "while experts were being copied.")
        st.metric("Forecast: all questions finish sooner by", f"{decision.forecast_makespan_gain_pct:+.1f}%", border=True)
        st.metric("Forecast: slowest question changes by", f"{decision.forecast_p95_regression_pct:+.1f}%", border=True,
                  help="Positive = slower")

# hindsight check (evaluation only)
never_row, always_row = policies.set_index("policy").loc["never_migrate"], policies.set_index("policy").loc["always_migrate"]
p95_after = -ui.pct(always_row.p95_latency_ms, never_row.p95_latency_ms)
mk_after = -ui.pct(always_row.makespan_ms, never_row.makespan_ms)
moving_helped = p95_after <= cfg.max_p95_regression_pct and -mk_after >= cfg.min_makespan_gain_pct
right = moving_helped == decision.approve
(st.success if right else st.warning)(
    f"**Checked afterwards on the held-out questions** (not used for the decision): moving would have changed the "
    f"slowest question by {p95_after:+.1f}% and total time by {mk_after:+.1f}%. "
    + ("So the decision was **right** under the same rules." if right else
       "Under the same rules, the **opposite** decision would have been better here — the 3-question forecast was "
       "misleading. This is a known limitation (small sample)."),
    icon=":material/fact_check:")

with ui.more("try it yourself — decision sandbox and policy comparison"):
    # ---------------------------------------------------------------- sandbox
    st.subheader("Migration decision sandbox")
    st.caption(":material/science: Simulation only — no GPU is touched, no transfer runs and no file is written. "
               "Actions here change this browser session only.")

    if "mig_sm" not in st.session_state or st.session_state.get("mig_cfg") != key:
        st.session_state.mig_sm = POL.MigrationStateMachine()
        st.session_state.mig_log = [(datetime.now().strftime("%H:%M:%S"), "proposed", "static → demand-aware proposed")]
        st.session_state.mig_cfg = key
        st.session_state.mig_override = None
    sm: POL.MigrationStateMachine = st.session_state.mig_sm


    def act(action: str):
        sm_ = st.session_state.mig_sm
        log = st.session_state.mig_log
        now = datetime.now().strftime("%H:%M:%S")
        if action == "reset":
            st.session_state.mig_sm = POL.MigrationStateMachine()
            log.append((now, "proposed", "new proposal"))
            return
        approve = decision.approve if action == "controller" else action == "approve"
        who = "controller" if action == "controller" else "operator override" if approve != decision.approve else "operator"
        if not approve:
            sm_.to("rejected", who)
            reason = explanation.headline if who == "controller" else "operator chose not to move (controller said " + \
                decision.decision.upper() + ")"
            log.append((now, "rejected", f"rejected by {who} — {reason}"))
            return
        sm_.to("approved", who)
        log.append((now, "approved", f"approved by {who}"))
        res = ui.simulate(key, tuple(cfg.test_ids), True)
        for s_ in res.states:
            if s_["state"] in ("transferring", "active", "failed") and s_["state"] in POL.TRANSITIONS[sm_.state]:
                sm_.to(s_["state"])
                log.append((now, s_["state"], f"t = {s_['t_ms']:.1f} ms · {s_['note']}"))


    b1, b2, b3, b4 = st.columns(4)
    proposed = sm.state == "proposed"
    b1.button("Follow controller", icon=":material/verified:", on_click=act, args=("controller",), disabled=not proposed,
              width="stretch", type="primary")
    b2.button("Approve (override)" if not decision.approve else "Approve", icon=":material/check_circle:", on_click=act,
              args=("approve",), disabled=not proposed, width="stretch")
    b3.button("Reject", icon=":material/cancel:", on_click=act, args=("reject",), disabled=not proposed, width="stretch")
    b4.button("New proposal", icon=":material/restart_alt:", on_click=act, args=("reset",), disabled=proposed,
              width="stretch")

    s1, s2 = st.columns([3, 2])
    with s1:
        with st.container(border=True):
            cur = sm.state
            lines = ["stateDiagram-v2", "  direction LR", "  [*] --> Proposed", "  Proposed --> Approved",
                     "  Proposed --> Rejected", "  Approved --> Transferring", "  Approved --> Failed: staging memory",
                     "  Transferring --> Active: all copies landed", "  Transferring --> Failed: copy error / rollback",
                     "  Active --> [*]", "  Rejected --> [*]", "  Failed --> [*]",
                     f"  classDef current fill:#0f766e,stroke:#5eead4,color:#ffffff,font-weight:bold",
                     f"  class {cur.capitalize()} current"]
            st.mermaid_chart("\n".join(lines), alt=f"Migration state machine; current state {cur}")
            st.markdown(f"Current state: **{cur.upper()}**")
    with s2:
        with st.container(border=True, height=330):
            st.markdown("**Session event log**")
            for t, s_, note in reversed(st.session_state.mig_log):
                st.markdown(f"`{t}` **{s_}** — {note}")

    # ---------------------------------------------------------------- outcome
    if sm.state in ("active", "failed", "transferring"):
        res = ui.simulate(key, tuple(cfg.test_ids), True)
        base = ui.simulate(key, tuple(cfg.test_ids), False)
        mig = res.migrations
        st.subheader("Simulated execution on held-out requests")
        with st.container(horizontal=True):
            st.metric("Outcome", res.summary["migration_state"].upper(), border=True)
            act_ms = res.summary["migration_activation_ms"]
            st.metric("Activation at", f"{act_ms:.1f} ms" if act_ms is not None else "—", border=True)
            st.metric("p95 latency", f"{res.summary['p95_latency_ms']:.1f} ms",
                      f"{-ui.pct(res.summary['p95_latency_ms'], base.summary['p95_latency_ms']):+.1f}% vs never",
                      delta_color="inverse", border=True)
            st.metric("Makespan", f"{res.summary['makespan_ms']:.1f} ms",
                      f"{-ui.pct(res.summary['makespan_ms'], base.summary['makespan_ms']):+.1f}% vs never",
                      delta_color="inverse", border=True)
        if res.summary["failure_reason"]:
            st.error(f"Migration failed: {res.summary['failure_reason']}. Requests kept the static layout (rollback).",
                     icon=":material/warning:")

        c1, c2 = st.columns([3, 2])
        with c1:
            with st.container(border=True):
                ui.section("Transfer playback", "live", caption="Cumulative expert-weight bytes landed over simulated time.")
                done = np.sort(mig.finish_ms.to_numpy())
                t_end = float(max(done.max() if len(done) else 0, res.summary["makespan_ms"]))
                t = st.slider("Simulated time (ms)", 0.0, t_end, min(t_end, float(act_ms or t_end)), key="mg_t")
                n_done = int((done <= t).sum())
                state_at = [x for x in res.states if x["t_ms"] <= t][-1]["state"]
                st.progress(n_done / max(len(mig), 1), text=f"{n_done}/{len(mig)} copies landed · state at t: **{state_at}**")
                fig = go.Figure(go.Scatter(x=done, y=np.arange(1, len(done) + 1) * cfg.expert_bytes / C.GB, mode="lines",
                                           line=dict(color=ui.AMBER, width=3), name="weights landed"))
                fig.add_vline(x=t, line_color=ui.TEAL)
                for r in res.requests.itertuples():
                    fig.add_vrect(x0=r.arrival_ms, x1=r.finish_ms, fillcolor=ui.BLUE, opacity=0.08, line_width=0,
                                  annotation_text=f"req {r.request_id} ({r.layout})", annotation_position="top left")
                fig.update_layout(xaxis_title="Simulated time (ms)", yaxis_title="GB copied", showlegend=False)
                ui.chart(fig, "Cumulative migration volume with request lifetimes", height=320)
        with c2:
            with st.container(border=True):
                ui.section("Per-request latency", "live", caption="Each request keeps the layout active at its arrival.")
                both = pd.concat([base.requests.assign(policy="never_migrate"), res.requests.assign(policy="executed")])
                fig = go.Figure()
                for pol_name, color in [("never_migrate", ui.SLATE), ("executed", ui.AMBER)]:
                    d = both[both.policy == pol_name]
                    fig.add_bar(name=pol_name.replace("_", " "), x=[f"req {r}" for r in d.request_id], y=d.latency_ms,
                                marker_color=color, text=d.layout, textposition="inside")
                fig.update_layout(barmode="group", yaxis_title="Simulated latency (ms)")
                ui.chart(fig, "Latency per held-out request with and without the migration", height=320)

    with st.container(border=True):
        ui.section("Policy comparison on held-out requests (this scenario)", "live")
        show = policies.copy()
        show["policy"] = show.policy.map(ui.POLICY_LABEL)
        ui.table(show.drop(columns=["failure_reason"]).round(2), "Never, always and cost-aware migration on held-out requests")

ui.step_footer("app_pages/migration.py")
