"""Shared dashboard helpers: styling, provenance badges, cached data and simulations.

Nothing here imports torch or loads OLMoE weights. Live simulations are cheap
(~10-50 ms each) and cached by their full parameter set.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import streamlit as st  # noqa: E402

from moeplace import config as C  # noqa: E402
from moeplace import data as D  # noqa: E402
from moeplace import placement as P  # noqa: E402
from moeplace import policy as POL  # noqa: E402
from moeplace import simulator as S  # noqa: E402
from moeplace.registry import BY_PATH, KIND_LABEL  # noqa: E402

# ------------------------------------------------------------------ palette
TEAL, BLUE, VIOLET, AMBER, PINK, SLATE, GREEN, RED = (
    "#2dd4bf", "#60a5fa", "#a78bfa", "#f59e0b", "#f472b6", "#94a3b8", "#34d399", "#f87171")
STRATEGY_COLORS = {"static": SLATE, "balanced": BLUE, "demand_aware": TEAL}
POLICY_COLORS = {"never_migrate": SLATE, "always_migrate": AMBER, "cost_aware": TEAL,
                 "static": SLATE, "migrate": AMBER}
CATEGORY_COLORS = {"ai": TEAL, "coding": BLUE, "general": VIOLET}
GPU_COLORS = [TEAL, BLUE, VIOLET, AMBER, PINK, GREEN, RED, SLATE]
STRATEGY_LABEL = {"static": "Static", "balanced": "Load-balanced", "demand_aware": "Demand-aware"}
POLICY_LABEL = {"never_migrate": "Never migrate", "always_migrate": "Always migrate", "cost_aware": "Cost-aware"}
HEAT = [[0, "#0b1a2e"], [0.15, "#0f3a58"], [0.4, "#0f7487"], [0.7, "#2dd4bf"], [1, "#e6fffa"]]

pio.templates["moe_dark"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d5e6", size=12), colorway=GPU_COLORS,
    xaxis=dict(gridcolor="#1c2942", zerolinecolor="#26385a", linecolor="#26385a"),
    yaxis=dict(gridcolor="#1c2942", zerolinecolor="#26385a", linecolor="#26385a"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=70, r=20, t=50, b=60), hoverlabel=dict(bgcolor="#13213a", bordercolor="#2dd4bf"),
    title=dict(font=dict(size=14, color="#e2e8f0"), x=0, xanchor="left"),
))


def chart(fig: go.Figure, alt: str, height: int | None = None, key: str | None = None):
    fig.update_layout(template="moe_dark")
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme=None, key=key, alt=alt,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def table(df: pd.DataFrame, alt: str, **kw):
    st.dataframe(df, width="stretch", hide_index=kw.pop("hide_index", True), alt=alt, **kw)


# ------------------------------------------------------------------ provenance
_BADGE = {
    "measured": (":material/science:", "green", "Real measurement"),
    "checkpoint": (":material/memory:", "gray", "From the model file"),
    "derived": (":material/functions:", "blue", "Calculated from real data"),
    "simulated": (":material/model_training:", "orange", "Simulation (saved result)"),
    "live": (":material/bolt:", "violet", "Simulation (updates with your settings)"),
}
_BADGE_HELP = {
    "measured": "Recorded by running the real OLMoE model on your Mac.",
    "checkpoint": "Read from the model's files on disk (sizes, shapes). The model is not loaded.",
    "derived": "Computed directly from the real measurements; no hardware assumptions.",
    "simulated": "Produced by the simulator with assumed (made-up) GPU and network speeds. Not a real GPU test.",
    "live": "Recomputed right now by the simulator from the settings on this page. Not a real GPU test.",
}

# The main story, in order. (file, short title) — used for the "previous / next step" links.
STORY = [
    ("app_pages/start.py", "Start here"),
    ("app_pages/model.py", "What is inside the model?"),
    ("app_pages/routing.py", "Which experts are busy?"),
    ("app_pages/placement.py", "Where should experts go?"),
    ("app_pages/inference.py", "How fast would requests run?"),
    ("app_pages/migration.py", "Should we move experts?"),
    ("app_pages/network.py", "Does network speed matter?"),
    ("app_pages/evaluation.py", "All results side by side"),
    ("app_pages/findings.py", "What we learned and the limits"),
]


def badge(kind: str, help: str | None = None):
    icon, color, label = _BADGE[kind]
    st.badge(label, icon=icon, color=color, help=help or _BADGE_HELP[kind])


def section(title: str, kind: str, caption: str | None = None, source: str | None = None):
    st.subheader(title)
    with st.container(horizontal=True, gap="small"):
        help_text = None
        if source and source in BY_PATH:
            a = BY_PATH[source]
            help_text = f"outputs/{a.path} — {a.producer}. Parameters: {a.parameters}. {a.notes}"
        badge(kind, help=help_text)
        if source:
            st.caption(f"`outputs/{source}`")
    if caption:
        st.caption(caption)


def page_header(title: str, subtitle: str, kinds: tuple[str, ...] = (), step: str | None = None):
    if step:
        st.caption(f":material/flag: {step}")
    st.title(title)
    st.caption(subtitle)
    if kinds:
        with st.container(horizontal=True, gap="small"):
            for k in kinds:
                badge(k)


def explain(what: str, read: list[str], takeaway: str, expanded: bool = True):
    """Plain-language 'About this page' box shown at the top of every page."""
    with st.expander("About this page — read me first", icon=":material/help:", expanded=expanded):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**What this page shows**")
            st.markdown(what)
        with c2:
            st.markdown("**How to read it**")
            st.markdown("\n".join(f"- {line}" for line in read))
        st.success(f"**Key takeaway:** {takeaway}", icon=":material/lightbulb:")


def step_footer(current: str):
    """'Previous step / next step' links following the main story order."""
    files = [f for f, _ in STORY]
    if current not in files:
        return
    i = files.index(current)
    st.divider()
    left, mid, right = st.columns([1, 1, 1])
    if i > 0:
        left.page_link(STORY[i - 1][0], label=f"Previous: {STORY[i - 1][1]}", icon=":material/arrow_back:")
    mid.caption(f"Step {i} of {len(STORY) - 1}" if i else "Introduction")
    if i < len(STORY) - 1:
        right.page_link(STORY[i + 1][0], label=f"Next: {STORY[i + 1][1]}", icon=":material/arrow_forward:")


SIM_DISCLAIMER = ("Simulated results use hypothetical GPU speeds, compute cost, activation sizes and network "
                  "parameters. They are not measured multi-GPU speedups. Routing inputs are real OLMoE measurements.")


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner=False)
def csv(name: str) -> pd.DataFrame:
    return D.read_csv(name)


@st.cache_data(show_spinner=False)
def routing(source: str | None = None) -> pd.DataFrame:
    return D.load_routing(source)


@st.cache_data(show_spinner=False)
def timings(source: str | None = None) -> pd.DataFrame:
    return D.load_timings(source)


@st.cache_data(show_spinner=False)
def model_config() -> dict:
    return D.model_config()


@st.cache_data(show_spinner=False)
def checkpoint_index_meta() -> dict:
    return D.checkpoint_index().get("metadata", {})


@st.cache_data(show_spinner=False)
def checkpoint_files() -> pd.DataFrame:
    return D.checkpoint_files()


@st.cache_data(show_spinner=False)
def expert_memory():
    return D.expert_memory()


@st.cache_data(show_spinner=False)
def phase4f(kind: str) -> pd.DataFrame:
    return D.phase4f(kind)


def exists(name: str) -> bool:
    return (D.OUTPUTS / name).exists()


# ------------------------------------------------------------------ scenario (session-wide)
SCENARIO_KEY = "scenario"


def scenario() -> C.SimConfig:
    if SCENARIO_KEY not in st.session_state:
        st.session_state[SCENARIO_KEY] = C.DEFAULT.to_dict()
    return C.SimConfig.from_dict(st.session_state[SCENARIO_KEY])


def set_scenario(**changes) -> C.SimConfig | None:
    """Validate and store changes; show the validation error instead of crashing."""
    try:
        cfg = scenario().with_(**changes)
    except C.ConfigError as exc:
        st.error(str(exc), icon=":material/warning:")
        return None
    st.session_state[SCENARIO_KEY] = cfg.to_dict()
    return cfg


def reset_scenario():
    st.session_state[SCENARIO_KEY] = C.DEFAULT.to_dict()


def scenario_summary(cfg: C.SimConfig) -> str:
    speeds = ", ".join(f"{s:g}" for s in cfg.gpu_speeds)
    mem = f"{cfg.gpu_memory_gb:g} GB budget" if cfg.gpu_memory_gb else "slot-limited"
    return (f"{cfg.num_gpus} simulated GPUs (speeds {speeds}) · {cfg.bandwidth_gb_per_s:g} GB/s "
            f"{cfg.link_model} links · {cfg.expert_mb:.3f} MB/expert · {mem}")


def cfg_key(cfg: C.SimConfig) -> str:
    import json
    return json.dumps(cfg.to_dict(), sort_keys=True, default=list)


# ------------------------------------------------------------------ cached simulations
@st.cache_data(show_spinner=False, max_entries=64)
def plan(cfg_json: str, demand_source: str = "train") -> pd.DataFrame:
    import json
    cfg = C.SimConfig.from_dict(json.loads(cfg_json))
    r = routing()
    train = D.demand(r) if demand_source == "all" else D.demand(r, cfg.train_ids)
    return P.plan(train, cfg, eval_demand=D.demand(r, cfg.test_ids))


def layouts_for(cfg: C.SimConfig, demand_source: str = "train"):
    pl = plan(cfg_key(cfg), demand_source)
    return pl, {s: P.layout(pl, s) for s in C.STRATEGIES}


@st.cache_data(show_spinner=False, max_entries=64)
def simulate(cfg_json: str, ids: tuple[int, ...], allow_migration: bool) -> S.SimResult:
    import json
    cfg = C.SimConfig.from_dict(json.loads(cfg_json))
    _, layouts = layouts_for(cfg)
    traces = D.build_traces(routing(), ids)
    return S.run(traces, layouts, cfg, ids, allow_migration=allow_migration, record=True)


@st.cache_data(show_spinner=False, max_entries=64)
def controller(cfg_json: str):
    import json
    cfg = C.SimConfig.from_dict(json.loads(cfg_json))
    _, layouts = layouts_for(cfg)
    policies, requests, decision, migrations, _ = POL.evaluate(routing(), cfg, layouts)
    return policies, requests, decision, migrations


@st.cache_data(show_spinner=False, max_entries=16)
def bandwidth_sweep(cfg_json: str, bandwidths: tuple[float, ...], expert_bytes: tuple[int, ...]) -> pd.DataFrame:
    import json
    base = C.SimConfig.from_dict(json.loads(cfg_json))
    _, layouts = layouts_for(base)
    traces = D.build_traces(routing(), base.test_ids)
    rows = []
    for eb in expert_bytes:
        for bw in bandwidths:
            cfg = base.with_(bandwidth_gb_per_s=float(bw), expert_bytes=int(eb))
            for mig in (False, True):
                s = S.run(traces, layouts, cfg, base.test_ids, allow_migration=mig).summary
                rows.append({"bandwidth_gb_per_s": bw, "expert_mb": eb / C.MB, **s})
    return pd.DataFrame(rows)


def pct(new: float, old: float) -> float:
    return 100.0 * (old - new) / old if old else float("nan")


def crossover(bw: np.ndarray, improvement: np.ndarray) -> float | None:
    """Bandwidth where improvement crosses zero (linear in log-bandwidth between tested points)."""
    order = np.argsort(bw)
    bw, y = np.asarray(bw, float)[order], np.asarray(improvement, float)[order]
    for i in range(len(bw) - 1):
        if y[i] < 0 <= y[i + 1]:
            lx0, lx1 = np.log(bw[i]), np.log(bw[i + 1])
            return float(np.exp(lx0 + (0 - y[i]) * (lx1 - lx0) / (y[i + 1] - y[i])))
    return None


@st.cache_data(show_spinner=False, max_entries=16)
def validate_vs_random(cfg_json: str, demand_source: str = "train") -> dict:
    import json
    cfg = C.SimConfig.from_dict(json.loads(cfg_json))
    return P.validate_vs_random(plan(cfg_json, demand_source), cfg)


@st.cache_data(show_spinner=False, max_entries=32)
def explain_decision(cfg_json: str) -> POL.Explanation:
    """Plain-language reasons for the controller's decision (training requests only) + what would change it."""
    import json
    cfg = C.SimConfig.from_dict(json.loads(cfg_json))
    _, layouts = layouts_for(cfg)
    return POL.explain(D.build_traces(routing(), cfg.train_ids), layouts, cfg)


def show_explanation(ex: POL.Explanation, title: str = None):
    """Render an Explanation: headline, checks with meaning, cause, and what would change the decision."""
    title = title or ("Why was it approved?" if ex.approved else "Why was it rejected?")
    with st.container(border=True):
        st.markdown(f"#### {title}")
        (st.success if ex.approved else st.error)(ex.headline, icon=":material/check_circle:" if ex.approved
                                                  else ":material/cancel:")
        st.markdown("**The three checks**")
        for c in ex.checks:
            icon = {True: ":green[:material/check_circle: PASS]", False: ":red[:material/cancel: FAIL]",
                    None: ":gray[:material/remove_circle: not checked]"}[c["passed"]]
            st.markdown(f"{icon} **{c['check']}** — {c['meaning']}")
        if ex.cause:
            st.markdown("**What is causing it**")
            st.markdown("\n".join(f"- {c}" for c in ex.cause))
        if ex.what_would_change:
            st.markdown("**What would change the decision** (each found by re-running the controller)")
            st.markdown("\n".join(f"- {c}" for c in ex.what_would_change))


@st.cache_data(show_spinner=False, max_entries=16)
def assistant_facts(cfg_json: str) -> dict:
    """Live numbers the offline assistant uses, so its answers always match the pages."""
    import json
    cfg = C.SimConfig.from_dict(json.loads(cfg_json))
    held = csv("heldout_comparison.csv").set_index("strategy")
    inf = csv("simulated_inference_comparison.csv").set_index("strategy")
    cal = phase4f("calibrated").pivot(index="bandwidth_gb_per_s", columns="policy", values="p95_latency_ms")
    imp = 100 * (cal["static"] - cal["migrate"]) / cal["static"]
    tim = timings()
    return {
        "cut": pct(held.loc["demand_aware", "peak_test_load"], held.loc["static", "peak_test_load"]),
        "bal_cut": pct(held.loc["balanced", "peak_test_load"], held.loc["static", "peak_test_load"]),
        "p95_cut": pct(inf.loc["demand_aware", "p95_latency_ms"], inf.loc["static", "p95_latency_ms"]),
        "crossover": crossover(imp.index.to_numpy(), imp.to_numpy()) or float("nan"),
        "saved_decision": csv("policy_decision.csv").iloc[0].decision.upper(),
        "explanation": explain_decision(cfg_json),
        "validation": validate_vs_random(ui_default_key()),
        "n_requests": int(tim.request_id.nunique()), "tokens": int(tim.input_tokens.sum()),
        "scenario": scenario_summary(cfg),
    }


def ui_default_key() -> str:
    return cfg_key(C.DEFAULT)
