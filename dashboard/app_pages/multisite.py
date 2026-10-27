import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ms_views as V
import ui
from moeplace import forecast as F
from moeplace import multisite as M
from moeplace import usecases as U

ui.page_header("Multi-site, forecasting & agent",
               "Response to the proposal feedback (points 5, 6, 8, 10, 11, 12): six use cases across sites, site-aware "
               "demand forecasting, placement regret and drift, an operations agent, planner scalability and governance.",
               ("derived", "simulated", "live"))
ui.explain(
    "An **extension** of the main project: instead of one cluster, the experts are spread over **several data-centre "
    "sites** (e.g. regions). Traffic changes during the day, so the system **predicts demand** ahead of time, "
    "re-plans placement, and an **operations agent** proposes changes — with a human approving big or regulated ones.",
    ["First **choose a use case** in the box below (six realistic scenarios, e.g. multi-region, data residency).",
     "**Use case explorer:** who uses it, what is decided, and the demand over the day.",
     "**Site-aware forecasting:** predicted vs actual demand; lower error is better.",
     "**Placement regret & drift:** how much a plan based on the forecast loses versus perfect hindsight.",
     "**Operations agent:** run cycles, see its proposals, approve or reject them.",
     "**Scalability** (how fast the planner is) and **Governance & safety** (rules and approvals)."],
    "Site traffic here is synthetic; the planner can re-plan within milliseconds and the agent keeps humans in charge "
    "of large changes.", expanded=False)
st.warning("Site traffic is **synthetic** (hypothetical rates, mixes and events) over **measured** OLMoE routing profiles. "
           "Utilisation, latency, cost and regret come from an analytic model with hypothetical hardware "
           "(moeplace/multisite.py), not from real multi-site serving. Nothing on this page writes files.",
           icon=":material/warning:")


# ---------------------------------------------------------------- cached data (in memory only)
@st.cache_data(show_spinner=False)
def uc_demand(key: str) -> np.ndarray:
    return U.USE_CASES[key].demand(ui.routing())[0]


@st.cache_resource(show_spinner=False)
def fitted_models(key: str) -> dict:
    return V.fit_models(uc_demand(key), key)


def uc_state(key: str) -> dict:
    """Per-use-case session state, so switching scenarios never mixes results."""
    all_states = st.session_state.setdefault("ms_state", {})
    return all_states.setdefault(key, {"run": None, "cycles": 0, "offset": 8, "decisions": {}, "agent": None})


key = st.selectbox("Use case", V.UC_ORDER, format_func=lambda k: f"{k.split('_')[0].upper()} · {U.USE_CASES[k].title}",
                   key="ms_uc")
uc = U.USE_CASES[key]
prof = V.scenario_profile(key)
state = uc_state(key)
Y = uc_demand(key)
H = U.spec().horizon
saved_dir = V.latest_run("usecases")
saved = lambda rel: V.load_csv(saved_dir, f"{key}/{rel}", "run_usecases.py")

tabs = st.tabs(["Use case explorer", "Site-aware forecasting", "Placement regret & drift", "Operations agent",
                "Scalability benchmarks", "Governance & safety"])

# ================================================================= 1. use case explorer
with tabs[0]:
    st.subheader(prof["title"])
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Users:** {prof['users']}  \n**Decision question:** {prof['question']}  \n"
                    f"**Objective:** {prof['objective']}  \n**KPIs:** {prof['kpis']}")
    with c2:
        st.markdown(f"**Constraints:** {prof['constraints']}  \n**Replica budget:** {prof['replica_budget']} × experts  \n"
                    f"**Residency sites:** {', '.join(prof['residency_sites']) or 'none'}  \n"
                    f"**Events:** {'; '.join(prof['events'])}  \n**Consolidation allowed:** {prof['consolidation']}")
    with st.expander("Infrastructure assumptions (hypothetical)", expanded=False):
        st.json(prof["assumptions"])
        ui.table(prof["sites"], "Sites")
        ui.table(prof["devices"], "Devices")
    ui.section("Demand pattern per site", "derived", caption=V.SYNTHETIC)
    ui.table(prof["traffic"], "Site traffic definition")
    vol = V.site_volume_frame(Y, key)
    fig = go.Figure([go.Scatter(x=vol.window, y=vol[s.name], name=s.name, mode="lines") for s in uc.topology.sites])
    fig.add_vrect(x0=0, x1=uc.train_end, fillcolor="gray", opacity=0.12, annotation_text="training days")
    fig.update_layout(title="Assignments per 15-minute window by site (synthetic)", xaxis_title="window",
                      yaxis_title="assignments")
    V.bottom_legend(fig)
    ui.chart(fig, "Synthetic site demand over the simulated days", height=330)

    ui.section("Live planner for one window", "live", caption="Planner time is measured in this session; "
               "utilisation and latency are predicted by the analytic model.")
    t = st.slider("Window", 0, uc.T - 1, uc.train_end + 40, key="ms_t")
    rf = st.select_slider("Replica budget (× experts)", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0], value=uc.replica_frac, key="ms_rf")
    plan = M.plan_sites(Y[t], uc.topology, replica_frac=rf, constraints=uc.constraints, util_cap=uc.util_cap)
    ev = M.evaluate(plan, Y[t], uc.topology, uc.util_cap)
    with st.container(horizontal=True):
        st.metric("Planner time (measured)", f"{plan.info['planner_ms']:.0f} ms", border=True)
        st.metric("Utilisation J (simulated)", f"{ev['J']:.2f}", border=True)
        st.metric("Added latency / token (simulated)", f"{ev['added_latency_ms_per_token']:.0f} ms", border=True)
        st.metric("Residency violations", f"{ev['residency_violation_assignments']:.0f}", border=True)

    ui.section("Run this scenario", "live", caption="Runs the real pipeline (forecaster fitting, rolling evaluation, "
               "placement regret, agent) for one evaluation day, in memory. Takes roughly 5-20 s.")
    if st.button("Run scenario (1 evaluation day)", key="ms_run", icon=":material/play_arrow:"):
        with st.spinner("Running forecasters, planner and agent..."):
            state["run"] = V.run_scenario(key, Y, eval_windows=U.DAY)
    if state["run"] is not None:
        r = state["run"]
        d = r["agent_decisions"]
        with st.container(horizontal=True):
            st.metric("Agent objective (simulated)", f"{d.objective_actual_agent.mean():.3f}", border=True)
            st.metric("Static plan", f"{d.objective_actual_static.mean():.3f}", border=True)
            st.metric("Perfect-foresight plan (" + V.HINDSIGHT + ")", f"{d.objective_actual_oracle.mean():.3f}", border=True)
            st.metric("Re-plans", int((d.action != "keep").sum()), border=True)
        ui.table(r["forecast_summary"].round(4), "Forecaster comparison from this session's run")
    else:
        st.caption("No session run yet for this use case. Saved two-day results are used on the other tabs.")

# ================================================================= 2. forecasting
with tabs[1]:
    st.markdown(f"**Target:** Y[t, site, expert] = assignments per {U.spec().window_s / 60:.0f}-minute window "
                f"(1,024 layer-specific experts). **Horizon:** {H} windows (1 hour). **Season:** one day. "
                f"**Training:** windows 0-{uc.train_end - 1}; evaluation afterwards. Data: {V.SYNTHETIC}.")
    fitted = fitted_models(key)
    c1, c2 = st.columns(2)
    with c1:
        site = st.selectbox("Site", range(uc.topology.n_sites), format_func=lambda i: uc.topology.sites[i].name, key="ms_fc_site")
    with c2:
        tw = st.slider("Window for the heatmap", 0, uc.T - 1, uc.train_end + 40, key="ms_fc_t")
    ui.chart(V.fig_heatmap(V.demand_heatmap(Y, tw, site), f"Expert demand at {uc.topology.sites[site].name}, "
                           f"window {tw} (synthetic)"), "Site-specific expert demand heatmap", height=380)
    models = st.multiselect("Models", list(F.MODELS), default=["site_ewma", "global_ewma", "seasonal_naive"], key="ms_fc_models")
    pva = V.predicted_vs_actual(Y, {m: fitted[m] for m in models}, key, site)
    ui.chart(V.fig_predicted_vs_actual(pva, models, f"Predicted vs actual site volume, {uc.topology.sites[site].name} "
                                       "(first evaluation day)"), "Predicted versus actual demand", height=360)
    ebs = V.error_by_site(Y, fitted, key)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure([go.Bar(x=g.site, y=g.wape, name=m) for m, g in ebs.groupby("model")])
        fig.update_layout(title="Forecast error (WAPE) by site", barmode="group", yaxis_title="WAPE")
        V.bottom_legend(fig)
        ui.chart(fig, "WAPE by site and model", height=330)
    with c2:
        sub = ebs[ebs.model.isin(["site_ewma", "global_ewma"])]
        fig = go.Figure([go.Bar(x=g.site, y=g.popularity_error, name=m) for m, g in sub.groupby("model")])
        fig.update_layout(title="Global vs site-aware: expert-popularity error", barmode="group",
                          yaxis_title="total-variation distance")
        V.bottom_legend(fig)
        ui.chart(fig, "Popularity error, global vs site-aware", height=330)
    ui.section("Forecasting-model comparison incl. placement regret (saved two-day run)", "simulated")
    fs, msg = saved("forecast_summary.csv")
    if fs is None:
        st.info(msg)
    else:
        ui.table(fs.round(4), "Saved forecaster comparison")

# ================================================================= 3. regret & drift
with tabs[2]:
    st.info("Regret compares the plan built from the forecast with a plan built from the demand that actually "
            "arrived. That second plan uses **hindsight** and exists only for evaluation; the agent and the planner "
            "never see future demand when they decide (tested: `test_agent_decision_ignores_future_demand`).",
            icon=":material/visibility_off:")
    src = st.segmented_control("Data", ["saved two-day run", "this session's run"], default="saved two-day run",
                               key="ms_rg_src") or "saved two-day run"
    if src == "this session's run":
        rows = state["run"]["forecast_points"] if state["run"] is not None else None
        msg = None if rows is not None else "No session run yet: use **Run scenario** on the first tab."
    else:
        rows, msg = saved("forecast_points.csv")
    if rows is None:
        st.info(msg)
    c1, c2 = st.columns(2)
    with c1:
        tau_err = st.number_input("Error threshold (WAPE)", 0.01, 1.0, 0.10, 0.01, key="ms_rg_err")
    with c2:
        tau_reg = st.number_input("Regret threshold", 0.001, 1.0, 0.02, 0.005, format="%.3f", key="ms_rg_reg")
    cl = V.classify(rows, tau_err, tau_reg)
    ui.chart(V.fig_regret_scatter(rows, tau_err, tau_reg), "Forecast error versus placement regret", height=380)
    if not cl.empty:
        counts = cl.groupby(["model", "quadrant"]).size().unstack(fill_value=0).reset_index()
        ui.table(counts, "Harmful versus harmless forecast errors")
        ex = V.small_error_high_regret_example(rows, tau_err, tau_reg)
        if ex:
            st.markdown(f"**Why a small error can hurt:** at window {int(ex['t'])}, `{ex['model']}` had WAPE "
                        f"{ex['wape']:.3f} (small) but regret {ex['regret_rel']:.1%}. Near the {uc.util_cap or 1.0} "
                        "utilisation limit a few per cent of demand decides which replicas fit, so the forecast plan "
                        "drops copies the perfect-foresight plan keeps. Error metrics cannot see this; regret can.")
    ui.section("Capacity violations", "simulated",
               caption="Overload = forecast plan's utilisation >= 1 on the realised demand; above cap = above the "
                       "locality-mode policy cap (not a physical overload).")
    ui.table(V.capacity_violations(rows, uc.util_cap), "Capacity violations")
    alerts, recs = V.drift_alerts(rows, tau_reg)
    ui.section("Decision-sensitive drift alerts and retraining recommendations", "simulated",
               caption="Alert when the rolling mean (3 decisions) of hindsight regret exceeds the threshold.")
    ui.table(recs, "Retraining recommendations")
    if not alerts.empty:
        ui.table(alerts[alerts.alert].round(4), "Drift alerts", height=240)

# ================================================================= 4. agent
with tabs[3]:
    st.graphviz_chart(V.AGENT_WORKFLOW_DOT)
    st.caption("One click = one simulated hourly cycle. No background scheduler runs. Approvals below change only "
               "this browser session; the act tool still requires the deterministic authoriser's token.")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        off = st.slider("Start (windows after training)", 0, uc.T - uc.train_end - 8 * H, state["offset"], H,
                        key="ms_ag_offset")
    if off != state["offset"]:
        state.update(offset=off, cycles=0, decisions={}, agent=None)
    with c2:
        if st.button("Run next cycle", key="ms_ag_step", icon=":material/skip_next:", type="primary"):
            state["cycles"] += 1
    with c3:
        if st.button("Reset agent", key="ms_ag_reset", icon=":material/restart_alt:"):
            state.update(cycles=0, decisions={}, agent=None)
    if state["cycles"] > 0:
        sig = (state["cycles"], state["offset"], tuple(sorted(state["decisions"].items())))
        if state.get("agent_sig") != sig or state.get("agent") is None:
            state["agent"] = V.run_agent_cycles(key, Y, state["cycles"], state["offset"], state["decisions"])
            state["agent_sig"] = sig
        res = state["agent"]
        dec, last_t = res["decisions"], res["last_t"]
        last = dec.iloc[-1]
        st.markdown(f"### Cycle {state['cycles']} · window t = {last_t}")
        c1, c2 = st.columns(2)
        with c1:
            ui.section("Current observation", "derived")
            ui.table(V.observation(Y, key, last_t), "Observation at t")
        with c2:
            ui.section("Forecast for the next hour", "simulated")
            fc = res["agent"].forecaster.predict(last_t, H).mean(0).sum(axis=1)
            ui.table(pd.DataFrame({"site": [s.name for s in uc.topology.sites], "forecast assignments / window": fc}),
                     "Forecast per site")
        for p in res["proposals"]:
            prop = p["proposal"]
            with st.container(border=True):
                st.markdown(f"**Proposal:** {prop.reason} · forecast objective {prop.J_current_forecast:.3f} → "
                            f"{prop.J_new_forecast:.3f} · {prop.migration['new_copies']} new copies "
                            f"({prop.migration['bytes'] / 1e9:.2f} GB, ~{prop.migration['est_time_s']:.0f} s)")
                ui.table(V.placement_summary(prop.plan, prop.topo), "Proposed placement (copies per device)", height=220)
                ui.table(pd.DataFrame([{"check": s, "result": "PASS" if r else "FAIL"} for s, r in p["reasons"]]),
                         "Deterministic authoriser checks")
                st.markdown(f"**Authoriser:** {'approved' if p['authoriser_ok'] else 'rejected'} · "
                            f"**Human approval required:** {'yes' if p['needs_human'] else 'no'} · "
                            f"**Human decision:** {p['human_decision']}")
                why = V.explain_proposal(p)
                if why:
                    st.warning(why, icon=":material/info:")
        if res["pending"]:
            p = res["pending"][0]
            st.warning(f"Human approval pending for '{p['proposal'].reason}' at t = {p['t']}. The change is NOT applied "
                       "until approved. (Simulated approval, this session only.)", icon=":material/gavel:")
            a1, a2 = st.columns(2)
            dkey = f"{p['t']}:{p['proposal'].reason}"
            if a1.button("Approve (simulated)", key="ms_ag_approve"):
                state["decisions"][dkey] = "approve"
                st.rerun()
            if a2.button("Deny (simulated)", key="ms_ag_deny"):
                state["decisions"][dkey] = "deny"
                st.rerun()
        events = res["audit"]
        expl = events[events.t == last_t].event.iloc[-1] if not events.empty and (events.t == last_t).any() else last["reasons"]
        st.success(f"**Final action:** {last['action']} · {expl}", icon=":material/task_alt:")
        trace = res["trace"]
        ui.table(trace[trace.t == last_t], "Tool calls in this cycle", height=240)
        ui.table(res["audit"], "Audit history", height=240)
        ui.table(dec[["t", "action", "human_decision", "needs_human", "new_copies", "migration_gb",
                      "objective_actual_agent", "objective_actual_static"]].round(3),
                 "Decision history (outcome columns use " + V.HINDSIGHT + ")")
    else:
        st.info("Press **Run next cycle** to execute one simulated agent cycle.")

# ================================================================= 5. scalability
with tabs[4]:
    sc_dir = V.latest_run("scalability")
    df, msg = V.load_csv(sc_dir, "scalability.csv", "run_scalability.py")
    meta = V.run_meta(sc_dir)
    if df is None:
        st.info(msg)
    else:
        pp = V.proposal_point(df)
        if pp:
            st.metric(f"Proposal point ({pp['experts']} experts, {pp['sites']} sites) vs 2 s target",
                      f"{pp['planner_ms']:.2f} ms", f"{pp['target_ms'] / pp['planner_ms']:,.0f}× under target",
                      border=True, help=V.platform_label(meta))
        st.caption(f"Planner timings: {V.platform_label(meta)}. Latency and cost: simulated.")
        factor = st.segmented_control("Factor", ["experts", "devices_per_site", "sites", "replica_frac", "constraints"],
                                      default="experts", key="ms_sc_factor") or "experts"
        sv = V.scalability_views(df, meta, factor)
        tb = sv["table"]
        if tb.empty:
            st.info("No rows for this factor in the saved run.")
        else:
            x = tb[sv["x"]].astype(str)
            fig = go.Figure(go.Scatter(x=x, y=tb.planner_ms, mode="lines+markers", name="median planner time"))
            fig.add_hline(y=sv["target_ms"], line_dash="dash", annotation_text="2 s target")
            fig.update_layout(title=sv["planner_title"], yaxis_type="log", yaxis_title="ms (log)", xaxis_title=sv["x"])
            ui.chart(fig, "Measured planner time", height=330)
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(x=x, y=tb.sim_latency_ms))
                fig.update_layout(title=sv["latency_title"], yaxis_title="ms per token", xaxis_title=sv["x"])
                ui.chart(fig, "Simulated latency", height=320)
            with c2:
                fig = go.Figure(go.Bar(x=x, y=tb.sim_cost_usd_per_window))
                fig.update_layout(title=sv["cost_title"], yaxis_title="USD per 15 min", xaxis_title=sv["x"])
                ui.chart(fig, "Simulated cost", height=320)
            ui.table(tb.round(3), "Scalability table")
        fits, _ = V.load_csv(sc_dir, "growth_fits.csv", "run_scalability.py")
        if fits is not None:
            ui.table(fits.round(3), "Growth exponents (planner_ms ∝ x^k)")

# ================================================================= 6. governance
with tabs[5]:
    st.error("These are **documented governance considerations** from docs/GOVERNANCE.md, not a verified legal "
             "compliance assessment. Obligations must be confirmed by legal/compliance for a real deployment.",
             icon=":material/policy:")
    gv = V.governance_views(key)
    if not gv["found"]:
        st.info("docs/GOVERNANCE.md not found.")
    ui.section(f"Compliance considerations for {key.split('_')[0].upper()}", "derived")
    ui.table(gv["compliance"], "Use-case compliance considerations")
    ui.section("Data-residency restrictions", "simulated")
    if prof["residency_sites"]:
        st.markdown(f"Sites with residency: **{', '.join(prof['residency_sites'])}**. Each hosts a copy of every expert "
                    "and serves all of its own demand; any change to its experts needs human approval.")
    else:
        st.markdown("No residency site in this use case (cross-site traffic is allowed and measured).")
    ui.section("Human approval and blocking rules", "derived")
    ui.table(V.approval_rules(), "Approval rules")
    ui.section("Operational risk register (NIST AI RMF)", "derived")
    ui.table(gv["risks"], "Risk register", height=320)
    ui.section("Frameworks", "derived")
    ui.table(gv["frameworks"], "Governance frameworks")
    session_agent = state.get("agent")
    if session_agent is not None:
        audit, dec = session_agent["audit"], session_agent["decisions"]
        origin = "this session's agent cycles"
    else:
        audit, _ = saved("agent_audit.csv")
        dec, _ = saved("agent_decisions.csv")
        origin = "saved two-day run"
    ui.section(f"Audit trail ({origin})", "simulated")
    if audit is None or audit.empty:
        st.info("No audit events yet.")
    else:
        ui.table(audit, "Audit trail", height=260)
    ui.section(f"Blocked migration reasons ({origin})", "simulated")
    ui.table(V.blocked_reasons(dec), "Blocked reasons", height=260)
