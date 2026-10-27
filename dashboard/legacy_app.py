
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# Original Phase 1-4A dashboard, kept for reference. New app: dashboard/app.py
ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="MoE Expert Placement Planner",
    layout="wide"
)

st.title("MoE Expert Placement Planner")
st.caption("OLMoE-1B-7B — Measured Expert Routing")

df = pd.read_csv(ROOT / "outputs/expert_routing.csv")

# Check that the routing data has the expected structure
required = {"layer_id", "expert_id", "token_assignments"}
assert required.issubset(df.columns)

total = int(df["token_assignments"].sum())
active = int((df["token_assignments"] > 0).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("MoE Layers", df["layer_id"].nunique())
c2.metric("Expert Slots", len(df))
c3.metric("Active Expert Slots", active)
c4.metric("Total Assignments", total)

st.subheader("Expert Routing Heatmap")

matrix = df.pivot(
    index="layer_id",
    columns="expert_id",
    values="token_assignments"
).fillna(0)

fig = px.imshow(
    matrix,
    labels={
        "x": "Expert ID",
        "y": "Layer ID",
        "color": "Assignments"
    },
    aspect="auto",
    color_continuous_scale="Viridis"
)

fig.update_layout(height=550)
st.plotly_chart(fig, width="stretch")

st.subheader("Expert Demand by Layer")

layer = st.selectbox(
    "Select layer",
    sorted(df["layer_id"].unique())
)

filtered = df[df["layer_id"] == layer]

fig2 = px.bar(
    filtered,
    x="expert_id",
    y="token_assignments",
    title=f"Layer {layer}: Expert Assignments"
)

st.plotly_chart(fig2, width="stretch")

st.subheader("Most Selected Experts")

st.dataframe(
    df.sort_values(
        "token_assignments",
        ascending=False
    ).head(20),
    width="stretch"
)

st.info(
    "These measurements come from a single prompt. "
    "They are not yet representative of production traffic."
)


st.divider()
st.subheader("Workload Demand Comparison")

try:
    category_df = pd.read_csv(
        ROOT / "outputs/expert_demand_by_category.csv"
    )

    selected_layer = st.selectbox(
        "Layer for workload comparison",
        sorted(category_df.layer_id.unique()),
        key="workload_layer",
    )

    comparison = category_df[
        category_df.layer_id == selected_layer
    ]

    fig3 = px.bar(
        comparison,
        x="expert_id",
        y="assignments_per_token",
        color="category",
        barmode="group",
        title=f"Expert Demand by Workload — Layer {selected_layer}",
        labels={
            "expert_id": "Expert ID",
            "assignments_per_token": "Assignments per input token",
        },
    )

    st.plotly_chart(fig3, width="stretch")

except FileNotFoundError:
    st.info("Run the multi-workload profiling scripts first.")


st.divider()
st.header("Phase 3 — GPU Placement Comparison")

try:
    placements = pd.read_csv(ROOT / "outputs/gpu_placements.csv")
    metrics = pd.read_csv(ROOT / "outputs/placement_comparison.csv")

    st.subheader("Placement Strategy Metrics")
    st.dataframe(metrics, width="stretch")

    st.subheader("Expert Demand Assigned to Each GPU")

    gpu_load = (
        placements.groupby(["strategy", "gpu_id"], as_index=False)
        ["demand"].sum()
    )

    fig = px.bar(
        gpu_load,
        x="gpu_id",
        y="demand",
        color="strategy",
        barmode="group",
        title="GPU Load by Placement Strategy",
        labels={
            "gpu_id": "GPU ID",
            "demand": "Total Expert Assignments"
        }
    )

    st.plotly_chart(fig, width="stretch")

    selected_strategy = st.selectbox(
        "Inspect placement",
        sorted(placements.strategy.unique())
    )

    st.dataframe(
        placements[
            placements.strategy == selected_strategy
        ],
        width="stretch"
    )

except FileNotFoundError:
    st.info("Run optimize_placement.py first.")


st.divider()
st.header("Phase 4 — Distributed Inference Simulation")

try:
    simulation = pd.read_csv(
        ROOT / "outputs/simulated_inference_comparison.csv"
    )

    st.dataframe(simulation, width="stretch")

    fig_latency = px.bar(
        simulation,
        x="strategy",
        y="p95_latency_ms",
        title="Simulated p95 Latency",
        labels={
            "strategy": "Placement Strategy",
            "p95_latency_ms": "p95 Latency (ms)"
        }
    )

    st.plotly_chart(
        fig_latency,
        width="stretch"
    )

    fig_comm = px.bar(
        simulation,
        x="strategy",
        y="communication_mb",
        title="Simulated Communication Volume",
        labels={
            "strategy": "Placement Strategy",
            "communication_mb": "Communication (MB)"
        }
    )

    st.plotly_chart(
        fig_comm,
        width="stretch"
    )

    st.caption(
        "Simulation results use hypothetical GPU speeds, "
        "compute costs and network parameters. "
        "They are not real multi-GPU benchmark results."
    )

except FileNotFoundError:
    st.info("Run simulate_inference.py first.")

