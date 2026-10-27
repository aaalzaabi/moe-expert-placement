import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import gpu_viz as V
import live_model as LM
import ui
from ui import C, D

ui.page_header("Try your own prompt",
               "Type any question, run it through the real OLMoE model, see which experts it uses, and watch how the "
               "simulated GPUs would handle it.", ("measured", "live"))
ui.explain(
    "Until now the project used 6 fixed questions. Here you can test **your own prompt**:\n\n"
    "1. The real OLMoE model reads your prompt once. Its router picks **8 of the 64 experts** for every word "
    "(token) in each of the 16 layers. This part is **real** and measured on this Mac.\n"
    "2. Those experts then light up on the **simulated GPUs**, placed by each method from page 3, to show which GPU "
    "would be the bottleneck. This part is **simulated** with assumed GPU speeds.",
    ["The first run loads the model (about 10–60 seconds, about 15 GB of memory). Later prompts take seconds.",
     "**Heatmap:** rows are layers, columns are experts; brighter = chosen more often by your prompt.",
     "**GPU picture:** press ▶ Play to go layer by layer. Red bar = the slowest GPU, which the others wait for.",
     "The placement was planned from the 3 training questions only, so your prompt is a fresh test of it."],
    "Your prompt is a new, unseen test: if demand-aware still gives the lowest total time, the plan generalises "
    "beyond the questions it was built from.")

cfg = ui.scenario()
cfg_json = ui.cfg_key(cfg)
EXAMPLES = ["What is the capital of France?", "Write a Python function that reverses a list.",
            "Explain how vaccines work.", "Translate 'good morning' into Spanish."]


@st.cache_resource(show_spinner=False, max_entries=1)
def real_model() -> LM.Loaded:
    return LM.load("bfloat16")


@st.cache_data(show_spinner=False, max_entries=16)
def traces(cfg_json: str) -> dict[str, pd.DataFrame]:
    cfg = C.SimConfig.from_dict(json.loads(cfg_json))
    pl = ui.plan(cfg_json)
    return {s: V.placement_trace(pl, s, cfg) for s in C.STRATEGIES}


def use_example():
    if st.session_state.get("mp_example"):
        st.session_state.mp_text = st.session_state.mp_example


# ---------------------------------------------------------------- input
with st.container(border=True):
    st.pills("Need an idea? Pick an example", EXAMPLES, key="mp_example", on_change=use_example)
    with st.form("mp_form", border=False):
        text = st.text_area("Your prompt", key="mp_text", max_chars=1000, height=90,
                            placeholder="e.g. How do airplanes stay in the air?")
        submitted = st.form_submit_button("Run on the real model", icon=":material/play_arrow:", type="primary")
    model_note = st.empty()  # filled below, after a run may have loaded the model
loaded = st.session_state.get("mp_loaded", False)

if submitted:
    if not text.strip():
        st.warning("Please type a prompt first.", icon=":material/edit:")
    elif not LM.checkpoint_present():
        st.error("The OLMoE checkpoint folder is missing, so the real model cannot run here.", icon=":material/error:")
    else:
        with st.status("Running the real OLMoE model…", expanded=True) as status:
            if not loaded:
                status.write("Loading the model from disk (first time only)…")
            m = real_model()
            st.session_state.mp_loaded = True
            status.write(f"Model ready on **{m.device}** ({m.dtype}). Reading your prompt…")
            result = LM.route(m, text)
            status.update(label=f"Done: {len(result.tokens)} tokens in {result.forward_seconds:.2f} s",
                          state="complete", expanded=False)
        st.session_state.mp_result = result

model_note.caption(("Model is loaded: runs take a few seconds." if st.session_state.get("mp_loaded") else
                    "The model is not loaded yet. The first run loads it (about 10–60 seconds, about 15 GB of memory).")
                   + f" Prompts longer than {LM.MAX_TOKENS} tokens are cut. Nothing is saved to disk.")

result: LM.Routed | None = st.session_state.get("mp_result")
if result is None:
    st.info("Type a prompt above and press **Run on the real model** to see the results here.",
            icon=":material/keyboard:")
    st.stop()

# ---------------------------------------------------------------- 1. real routing
r = result.routing
used = r[r.token_assignments > 0]
train = D.demand(ui.routing(), cfg.train_ids)
unseen = int((train.reindex(pd.MultiIndex.from_frame(used[["layer_id", "expert_id"]])).to_numpy() == 0).sum())

st.subheader(f"Your prompt: “{result.prompt[:120]}{'…' if len(result.prompt) > 120 else ''}”")
with st.container(horizontal=True):
    st.metric("Tokens (word pieces)", len(result.tokens), border=True)
    st.metric("Model run time", f"{result.forward_seconds:.2f} s", border=True,
              help=f"Measured: one forward pass on this Mac ({result.device}, {result.dtype}). Single device, "
                   "not a multi-GPU time.")
    st.metric("Different experts used", f"{len(used)} of 1,024", border=True)
    st.metric("Experts the plan never saw used", unseen, border=True,
              help="Experts your prompt used that the 3 training questions never used. The placement treated them "
                   "as idle, so they are a test of how well the plan generalises.")

with st.container(border=True):
    ui.section("Which experts your prompt used", "measured",
               caption="Real router choices from OLMoE. Each token picks 8 of 64 experts in every layer.")
    mat = r.pivot(index="layer_id", columns="expert_id", values="token_assignments")
    fig = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, colorscale=ui.HEAT,
                               colorbar=dict(title="times<br>chosen"),
                               hovertemplate="layer %{y}, expert %{x}: chosen %{z}×<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Layer", dtick=1)
    fig.update_xaxes(title="Expert", dtick=8)
    ui.chart(fig, "Heatmap of experts chosen by your prompt in each layer", height=380)

with ui.more("which experts each word used"):
    labels = [f"{i}: {t!r}" for i, t in enumerate(result.tokens)]
    pick = st.selectbox("Word piece (token)", range(len(labels)), format_func=lambda i: labels[i], key="mp_tok")
    st.caption(f"The 8 experts the router picked for token {labels[pick]} in each layer.")
    ui.table(pd.DataFrame(result.choices[:, pick, :], columns=[f"choice {k + 1}" for k in range(C.TOP_K)])
             .rename_axis("layer").reset_index(), f"Experts chosen for token {pick} per layer", height=300)
    st.download_button("Download this routing as CSV", r.to_csv(index=False), "my_prompt_routing.csv", "text/csv",
                       icon=":material/download:")

# ---------------------------------------------------------------- 2. simulated GPUs
with st.container(border=True):
    ui.section("How the simulated GPUs would handle it", "live",
               caption="Real expert choices, simulated GPUs: work = tokens ÷ assumed GPU speed; each layer waits for "
                       "its slowest GPU. Computation only (no network or queues).")
    strategy = st.segmented_control(
        "Placement method", list(C.STRATEGIES), default="demand_aware", key="mp_strategy",
        format_func=lambda s: {"static": "Static (simple)", "balanced": "Load-balanced",
                               "demand_aware": "Demand-aware (ours)"}[s]) or "demand_aware"
    tr = traces(cfg_json)
    pl = ui.plan(cfg_json)
    work = {s: V.question_work(r, LM.PROMPT_ID, pl, s, cfg) for s in C.STRATEGIES}
    ymax = max(per.to_numpy().max() for _, per in work.values())
    active, per = work[strategy]
    ui.chart(V.fig_question_animation(tr[strategy], active, per, cfg, f"Your prompt · {ui.STRATEGY_LABEL[strategy]}",
                                      ymax=ymax),
             f"Animation of your prompt flowing through 16 layers on the {ui.STRATEGY_LABEL[strategy]} placement",
             height=640, key=f"mp_anim_{strategy}")

    summ = {s: V.question_summary(p) for s, (_, p) in work.items()}
    base = summ["static"]["total"]
    with st.container(horizontal=True):
        for s, m in summ.items():
            st.metric(f"{ui.STRATEGY_LABEL[s]} · total time", f"{m['total']:,.1f}",
                      None if s == "static" else f"{100 * (m['total'] / base - 1):+.1f}% vs static",
                      delta_color="inverse", border=True,
                      help=f"Sum over 16 layers of the slowest GPU's work ÷ speed. GPUs wait "
                           f"{m['waiting_pct']:.0f}% of the time.")
    best = min(summ, key=lambda s: summ[s]["total"])
    gain = 100 * (1 - summ["demand_aware"]["total"] / base)
    if best == "demand_aware":
        st.success(f"For your prompt, **demand-aware is fastest**: {gain:.1f}% less time than static. The plan, made "
                   "from 3 other questions, still works on this new one.", icon=":material/check_circle:")
    else:
        st.warning(f"For your prompt, **{ui.STRATEGY_LABEL[best]}** is fastest (demand-aware: {-gain:+.1f}% vs static). "
                   "A single short prompt can differ from the 3 questions the plan was built from — this is the "
                   "small-sample limitation; more profiled prompts would make the plan more robust.",
                   icon=":material/info:")

st.caption("Real: which experts your prompt used and the model run time on this Mac. Simulated: the GPUs, their "
           "speeds and every time on the GPU picture. The model runs in BF16 here (the saved Phase 2 data used FP32), "
           "so in rare near-ties an expert choice can differ: on question 1 it matched 891 of the 896 saved choices.")

c1, c2 = st.columns(2)
with c1:
    if st.session_state.get("mp_loaded") and st.button("Unload the model (free memory)", icon=":material/eject:"):
        real_model.clear()
        LM.unload()
        st.session_state.mp_loaded = False
        st.rerun()
with c2:
    st.page_link("app_pages/gpu_sim.py", label="Compare with the saved questions on the visual simulator",
                 icon=":material/play_circle:")
