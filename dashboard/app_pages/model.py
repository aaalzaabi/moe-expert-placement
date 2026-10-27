import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import C

ui.page_header("What is inside the model?",
               "The real OLMoE model: its layers, its 1,024 experts and how big they are. Read from the model's files "
               "only — the model itself is not loaded.", ("checkpoint", "measured"), step="Part 1 · Step 1 of 8")
ui.explain(
    "OLMoE has **16 layers**. Each layer has **64 experts** (small networks), and for every word a *router* picks "
    "**8 of them**. That makes **1,024 experts**, each about **12.6 MB**. Experts make up about **93%** of the model's "
    "size — which is why deciding *where they live* matters.",
    ["The **top row of numbers** gives the model's size facts.",
     "The **diagram** shows one layer: attention → router → 8 chosen experts → combined result.",
     "The **expert inspector** lets you pick any expert, e.g. layer 3, expert 17, and see its size, how often it "
     "was used, and which simulated GPU it would be placed on.",
     "The **donut chart** shows that experts dominate the model's memory."],
    "Experts are most of the model, so placing them well across GPUs is the key infrastructure decision.")

mc = ui.model_config()
meta = ui.checkpoint_index_meta()
mem_summary, tensors = ui.expert_memory()
expert_total = float(mem_summary.expert_bytes.sum())
ckpt_total = float(meta.get("total_size", 0))

with st.container(horizontal=True):
    st.metric("Checkpoint tensors", f"{ckpt_total / C.GB:.2f} GB", border=True, help="model.safetensors.index.json total_size")
    st.metric("Expert weights", f"{expert_total / C.GB:.3f} GB", border=True,
              help=f"{100 * expert_total / ckpt_total:.1f}% of the checkpoint")
    st.metric("Per expert (BF16)", f"{C.CHECKPOINT_EXPERT_BYTES / C.MB:.3f} MB", border=True,
              help="gate_proj + up_proj + down_proj = 3 × 2048 × 1024 values × 2 bytes")
    st.metric("Per layer (64 experts)", f"{mem_summary.expert_bytes.iloc[0] / C.MB:.0f} MB", border=True)
    st.metric("Non-expert weights", f"{(ckpt_total - expert_total) / C.GB:.3f} GB", border=True,
              help="Attention, router, embeddings, norms, LM head — replicated on every GPU in the simulator's mental model")

with st.container():
    with st.container(border=True):
        ui.section("Architecture", "checkpoint", "One OLMoE decoder layer: attention, then a router choosing 8 of 64 "
                   "expert MLPs per token. Repeated 16 times.")
        st.mermaid_chart("""
flowchart LR
  T["token hidden state<br/>d=2048"] --> AT["self-attention<br/>16 heads"] --> R{"router<br/>64 logits"}
  R -->|"top-8"| E1["expert (L, e1)<br/>SwiGLU 2048→1024→2048"]
  R --> E2["expert (L, e2)"]
  R --> E8["... expert (L, e8)"]
  E1 --> S(("weighted<br/>sum"))
  E2 --> S
  E8 --> S
  S --> N["next layer L+1"]
""", alt="One MoE layer: attention, router, top-8 of 64 experts, weighted sum")
with ui.more("configuration, expert inspector, memory"):
    with st.container(border=True):
        ui.section("Model configuration", "checkpoint", source=None)
        keys = ["architectures", "model_type", "num_hidden_layers", "num_experts", "num_experts_per_tok",
                "hidden_size", "intermediate_size", "num_attention_heads", "vocab_size", "max_position_embeddings",
                "torch_dtype", "norm_topk_prob", "router_aux_loss_coef"]
        ui.table(pd.DataFrame({"field": keys, "value": [str(mc.get(k)) for k in keys]}), "OLMoE config.json fields",
                 height=320)

    # ---------------------------------------------------------------- expert inspector
    st.subheader("Layer-specific expert inspector")
    st.caption("An expert's identity is the pair (layer_id, expert_id). There are 1,024 distinct experts.")
    c1, c2 = st.columns([1, 3])
    with c1:
        layer = st.selectbox("Layer", range(C.NUM_LAYERS), key="mx_layer")
        expert = st.number_input("Expert", min_value=0, max_value=C.EXPERTS_PER_LAYER - 1, value=0, step=1, key="mx_expert")
        st.markdown(f"### Expert `({layer}, {expert})`")
        st.caption(f"Global slot index {layer * C.EXPERTS_PER_LAYER + expert} of 1024")

    routing = ui.routing()
    with c2:
        t = tensors[(tensors.layer_id == layer) & (tensors.expert_id == expert)]
        with st.container(border=True):
            ui.section("Weight tensors", "checkpoint", source="phase4f/memory/expert_tensor_metadata.csv")
            ui.table(t[["tensor_name", "shape", "dtype", "size_mb"]], f"Tensors of expert ({layer}, {expert})")
            st.caption(f"Total {t.bytes.sum() / C.MB:.3f} MB in BF16. The profiling scripts load the model in FP32, "
                       f"which would occupy {2 * t.bytes.sum() / C.MB:.3f} MB in memory.")

        pl = ui.csv("heterogeneous_placements.csv")
        here = pl[(pl.layer_id == layer) & (pl.expert_id == expert)].set_index("strategy")
        req = routing[(routing.layer_id == layer) & (routing.expert_id == expert)]
        m1, m2, m3 = st.columns(3)
        m1.metric("Measured assignments (6 prompts)", int(req.token_assignments.sum()), border=True)
        m2.metric("Rank within layer", int(routing[routing.layer_id == layer].groupby("expert_id").token_assignments.sum()
                                             .rank(ascending=False, method="min").loc[expert]), border=True)
        m3.metric("Simulated GPU (static → demand-aware)",
                  f"{here.loc['static', 'gpu_id']} → {here.loc['demand_aware', 'gpu_id']}", border=True,
                  help="From the saved Phase 3B placement. A simulated assignment only.")

    with st.container(border=True):
        ui.section(f"Routing to expert ({layer}, {expert}) by request", "measured", source="workload_routing.csv")
        fig = go.Figure(go.Bar(x=[f"req {r} · {c}" for r, c in zip(req.request_id, req.category)], y=req.token_assignments,
                               marker_color=[ui.CATEGORY_COLORS.get(c, ui.SLATE) for c in req.category]))
        fig.update_layout(yaxis_title="Token assignments", showlegend=False)
        ui.chart(fig, f"Token assignments to expert ({layer},{expert}) per request", height=260)

    # ---------------------------------------------------------------- memory
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            ui.section("Checkpoint memory composition", "checkpoint")
            fig = go.Figure(go.Pie(labels=["Expert MLPs (1,024)", "Everything else"],
                                   values=[expert_total, ckpt_total - expert_total], hole=0.6,
                                   marker=dict(colors=[ui.TEAL, ui.SLATE])))
            ui.chart(fig, "Expert weights are about 93 percent of checkpoint bytes", height=280)
    with c2:
        with st.container(border=True):
            ui.section("Checkpoint files", "checkpoint")
            files = ui.checkpoint_files()
            ui.table(files.round(3), "Files in the local OLMoE checkpoint folder", height=280)

st.info("This page describes the real model. The dashboard does not move, load or manage experts on GPUs; GPU "
        "assignments shown elsewhere are simulation plans.", icon=":material/info:")

ui.step_footer("app_pages/model.py")
