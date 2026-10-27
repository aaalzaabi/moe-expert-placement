"""View models for the visual GPU simulator page (app_pages/gpu_sim.py).

Two pictures, both built from the same placements the rest of the dashboard uses:

* placement_trace / fig_placement_animation: the placement rule placing experts one by one
  into the GPUs' slots, in the exact order moeplace.placement.assign uses.
* question_work / fig_question_animation: one real question (measured routing) flowing through the
  16 layers; per layer every GPU does its share and the layer finishes when the slowest GPU does.

"Work" is token assignments ÷ assumed relative GPU speed: the same load proxy as the placement page,
not measured GPU time. Pure pandas/plotly so the functions can be tested without a browser.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ui
from ui import C

EMPTY, IDLE = "#26385a", "#334155"
# Demand colours start at a visible blue-grey so an unused expert (demand 0) still looks occupied.
DEMAND_SCALE = [[0, "#3b4f6e"], [0.15, "#0f5f7a"], [0.4, "#0f8a9a"], [0.7, "#2dd4bf"], [1, "#e6fffa"]]


# ------------------------------------------------------------------ geometry
def grid_side(cfg: C.SimConfig) -> int:
    return math.ceil(math.sqrt(cfg.capacity_slots))


def _xy(gpu: np.ndarray, slot: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    return gpu * (side + 3) + slot % side, slot // side


def _marker_size(cfg: C.SimConfig) -> int:
    cols = cfg.num_gpus * (grid_side(cfg) + 3)
    return int(max(3, min(11, 720 // cols)))


def _gpu_labels(cfg: C.SimConfig) -> list[str]:
    return [f"GPU {g}" for g in range(cfg.num_gpus)]


def _grid_axes(fig: go.Figure, cfg: C.SimConfig, row: int = 1):
    side = grid_side(cfg)
    fig.update_xaxes(visible=False, range=[-1.5, cfg.num_gpus * (side + 3) - 2.5], row=row, col=1)
    fig.update_yaxes(visible=False, autorange=False, range=[side + 0.5, -2.8], row=row, col=1)
    fastest = max(cfg.gpu_speeds)
    for g, s in enumerate(cfg.gpu_speeds):
        tag = " (fastest)" if s == fastest else (" (slowest)" if s == min(cfg.gpu_speeds) else "")
        fig.add_annotation(x=g * (side + 3) + (side - 1) / 2, y=-1.6, text=f"<b>GPU {g}</b> · speed {s:g}×{tag}",
                           showarrow=False, font=dict(color=ui.GPU_COLORS[g % len(ui.GPU_COLORS)], size=12),
                           row=row, col=1)


def _empty_slots(cfg: C.SimConfig) -> go.Scatter:
    side = grid_side(cfg)
    g = np.repeat(np.arange(cfg.num_gpus), cfg.capacity_slots)
    s = np.tile(np.arange(cfg.capacity_slots), cfg.num_gpus)
    x, y = _xy(g, s, side)
    return go.Scatter(x=x, y=y, mode="markers", hoverinfo="skip", showlegend=False,
                      marker=dict(symbol="square-open", size=_marker_size(cfg) - 1, color=EMPTY))


# ------------------------------------------------------------------ 1. placing the experts
def placement_trace(placements: pd.DataFrame, strategy: str, cfg: C.SimConfig) -> pd.DataFrame:
    """One row per placement decision, in the order the algorithm makes them.

    Columns: step, layer_id, expert_id, train_demand, gpu_id, slot (position on that GPU),
    work_after / time_after (that GPU's total work and work ÷ speed after this step).
    """
    rows = placements[placements.strategy == strategy].reset_index(drop=True)
    d = rows.train_demand.to_numpy(float)
    # Same order as placement.assign: static = canonical order; greedy = busiest first (stable for ties).
    order = np.arange(len(rows)) if strategy == "static" else np.argsort(-d, kind="stable")
    t = rows.iloc[order].reset_index(drop=True)
    t.insert(0, "step", np.arange(1, len(t) + 1))
    t["slot"] = t.groupby("gpu_id").cumcount()
    t["work_after"] = t.groupby("gpu_id").train_demand.cumsum()
    t["time_after"] = t.work_after / np.asarray(cfg.gpu_speeds, float)[t.gpu_id]
    return t


def busy_time(trace: pd.DataFrame, cfg: C.SimConfig, upto: int | None = None) -> np.ndarray:
    """Work ÷ speed per GPU after the first `upto` placements (all if None)."""
    t = trace if upto is None else trace[trace.step <= upto]
    work = t.groupby("gpu_id").train_demand.sum().reindex(range(cfg.num_gpus), fill_value=0).to_numpy(float)
    return work / np.asarray(cfg.gpu_speeds, float)


RULE = {
    "static": "Static deals experts out in order like cards: expert number *i* goes to GPU *i mod {n}*. "
              "It does not look at how busy an expert is or how fast a GPU is.",
    "balanced": "Load-balanced takes the busiest remaining expert and gives it to the GPU with the **least work so far**. "
                "It ignores GPU speed.",
    "demand_aware": "Demand-aware takes the busiest remaining expert and gives it to the GPU whose **busy time would be "
                    "smallest after adding it** — (work already there + this expert's demand) ÷ GPU speed.",
}


def decision_table(trace: pd.DataFrame, step: int, cfg: C.SimConfig) -> tuple[pd.Series, pd.DataFrame]:
    """The numbers the rule looked at for one decision, one row per GPU."""
    row = trace.iloc[step - 1]
    before = trace[trace.step < step]
    speeds = np.asarray(cfg.gpu_speeds, float)
    out = []
    for g in range(cfg.num_gpus):
        mine = before[before.gpu_id == g]
        work, used = float(mine.train_demand.sum()), len(mine)
        full = used >= cfg.capacity_slots
        out.append({"GPU": f"GPU {g}", "speed": speeds[g], "experts already there": used,
                    "work already there": work, "busy time now": work / speeds[g],
                    "busy time if placed here": None if full else (work + row.train_demand) / speeds[g],
                    "free slots": cfg.capacity_slots - used, "chosen": g == row.gpu_id})
    return row, pd.DataFrame(out)


def fig_placement_animation(trace: pd.DataFrame, cfg: C.SimConfig, title: str, ymax: float | None = None,
                            n_frames: int = 32) -> go.Figure:
    side = grid_side(cfg)
    n = len(trace)
    x, y = _xy(trace.gpu_id.to_numpy(), trace.slot.to_numpy(), side)
    demand = trace.train_demand.to_numpy(float)
    cmax = max(1.0, float(np.quantile(demand, 0.99)))
    hover = [f"Layer {l}, expert {e}<br>demand {d:g}<br>→ GPU {g} (placed {s} of {n})"
             for l, e, d, g, s in zip(trace.layer_id, trace.expert_id, demand, trace.gpu_id, trace.step)]
    labels = _gpu_labels(cfg)
    colors = [ui.GPU_COLORS[g % len(ui.GPU_COLORS)] for g in range(cfg.num_gpus)]
    final = busy_time(trace, cfg)
    ideal = float(demand.sum() / sum(cfg.gpu_speeds))
    ymax = ymax or float(final.max())

    def experts(k: int) -> go.Scatter:
        return go.Scatter(x=x[:k], y=y[:k], mode="markers", showlegend=False, text=hover[:k],
                          hovertemplate="%{text}<extra></extra>",
                          marker=dict(symbol="square", size=_marker_size(cfg), color=demand[:k], cmin=0, cmax=cmax,
                                      colorscale=DEMAND_SCALE, showscale=True,
                                      colorbar=dict(title="demand", len=0.6, y=1, yanchor="top", thickness=12)))

    def bars(k: int) -> go.Bar:
        t = busy_time(trace, cfg, k)
        return go.Bar(x=labels, y=t, marker_color=colors, showlegend=False, text=[f"{v:,.0f}" for v in t],
                      textposition="inside", insidetextanchor="end", textfont=dict(color="#0b1220", size=13),
                      hovertemplate="%{x}: busy time %{y:,.0f} work units<extra></extra>")

    def caption(k: int) -> str:
        if k == 0:
            return f"{title} · press ▶ Play to watch the experts being placed"
        last = trace.iloc[k - 1]
        return (f"{title} · placed {k:,} of {n:,} · latest: layer {last.layer_id}, expert {last.expert_id} "
                f"(demand {last.train_demand:g}) → GPU {last.gpu_id}")

    fig = make_subplots(rows=2, cols=1, row_heights=[0.66, 0.34], vertical_spacing=0.1)
    fig.add_trace(_empty_slots(cfg), row=1, col=1)
    fig.add_trace(experts(n), row=1, col=1)
    fig.add_trace(bars(n), row=2, col=1)
    fig.add_hline(y=ideal, line_dash="dot", line_color=ui.GREEN, row=2, col=1,
                  annotation_text="perfect balance (finished)", annotation_font_color=ui.GREEN,
                  annotation_position="top left")
    _grid_axes(fig, cfg)
    fig.update_yaxes(title="Busy time (work ÷ speed)", range=[0, ymax * 1.25], row=2, col=1)

    ks = sorted({int(round(v)) for v in np.linspace(0, n, n_frames + 1)})
    # Experts are placed in trace order, so each frame only needs the first k positions; hover text and
    # colours come from the full arrays on the initial trace (keeps the figure small).
    fig.frames = [go.Frame(name=str(k), data=[go.Scatter(x=x[:k], y=y[:k]), bars(k)], traces=[1, 2],
                           layout=dict(title_text=caption(k))) for k in ks]
    fig.update_layout(title_text=caption(n), margin=dict(l=70, r=20, t=50, b=40), **_player(ks, "experts placed"))
    return fig


def _player(names, label: str, labels=None) -> dict:
    play = dict(frame=dict(duration=350, redraw=True), transition=dict(duration=0), fromcurrent=False, mode="immediate")
    return dict(
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.06, xanchor="left", yanchor="top",
                          pad=dict(t=0, r=10), showactive=False, bgcolor="#13213a", font=dict(color="#e2e8f0"),
                          buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=len(names) - 1, x=0.2, len=0.8, y=-0.04, yanchor="top", pad=dict(t=0),
                      currentvalue=dict(prefix=f"{label}: ", font=dict(color="#c9d5e6")),
                      font=dict(color="#94a3b8"), bgcolor="#13213a",
                      steps=[dict(label=str(lab), method="animate",
                                  args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                             for k, lab in zip(names, labels or names)])])


# ------------------------------------------------------------------ 2. a question flowing through the GPUs
def question_work(routing: pd.DataFrame, request_id: int, placements: pd.DataFrame, strategy: str,
                  cfg: C.SimConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-expert and per-(layer, GPU) work for one question.

    Returns (active, per_layer): `active` has one row per expert the question used (token count, GPU);
    `per_layer` is layers × GPUs of work ÷ speed.
    """
    r = routing[(routing.request_id == request_id) & (routing.token_assignments > 0)]
    lay = placements[placements.strategy == strategy][["layer_id", "expert_id", "gpu_id"]]
    m = r[["layer_id", "expert_id", "token_assignments"]].merge(lay, on=["layer_id", "expert_id"])
    m["work"] = m.token_assignments / np.asarray(cfg.gpu_speeds, float)[m.gpu_id]
    per = (m.groupby(["layer_id", "gpu_id"]).work.sum().unstack(fill_value=0.0)
           .reindex(index=range(C.NUM_LAYERS), columns=range(cfg.num_gpus), fill_value=0.0))
    return m, per


def question_summary(per_layer: pd.DataFrame) -> dict:
    """Layer-by-layer finish: each layer waits for its slowest GPU (simplified: compute only)."""
    layer_time = per_layer.max(axis=1)
    total = float(layer_time.sum())
    busy = float(per_layer.to_numpy().sum())
    n = per_layer.shape[1]
    bottleneck = per_layer.idxmax(axis=1)
    return {"total": total, "waiting_pct": 100 * (1 - busy / (n * total)) if total else 0.0,
            "layer_time": layer_time, "bottleneck_counts": bottleneck.value_counts().reindex(range(n), fill_value=0)}


def fig_question_animation(trace: pd.DataFrame, active: pd.DataFrame, per_layer: pd.DataFrame, cfg: C.SimConfig,
                           title: str, ymax: float | None = None) -> go.Figure:
    side = grid_side(cfg)
    size = _marker_size(cfg)
    pos = trace.set_index(["layer_id", "expert_id"])[["gpu_id", "slot"]]
    ax, ay = _xy(trace.gpu_id.to_numpy(), trace.slot.to_numpy(), side)
    labels = _gpu_labels(cfg)
    colors = [ui.GPU_COLORS[g % len(ui.GPU_COLORS)] for g in range(cfg.num_gpus)]
    tok_max = max(1.0, float(active.token_assignments.max()))
    ymax = ymax or float(per_layer.to_numpy().max())

    def lit(layer: int) -> go.Scatter:
        a = active[active.layer_id == layer]
        p = pos.loc[list(zip(a.layer_id, a.expert_id))]
        x, y = _xy(p.gpu_id.to_numpy(), p.slot.to_numpy(), side)
        return go.Scatter(
            x=x, y=y, mode="markers", showlegend=False,
            text=[f"Layer {layer}, expert {e}<br>{t} token(s) → GPU {g}" for e, t, g in
                  zip(a.expert_id, a.token_assignments, a.gpu_id)],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(symbol="square", size=size + 4 * (a.token_assignments.to_numpy() / tok_max),
                        color=[colors[g % len(colors)] for g in a.gpu_id], line=dict(color="#ffffff", width=1)))

    def bars(layer: int) -> go.Bar:
        t = per_layer.loc[layer].to_numpy()
        slow = int(t.argmax())
        return go.Bar(x=labels, y=t, showlegend=False, text=[f"{v:,.1f}" for v in t], textposition="outside",
                      cliponaxis=False, hovertemplate="%{x}: %{y:,.1f} work units this layer<extra></extra>",
                      marker=dict(color=[ui.RED if g == slow else colors[g] for g in range(cfg.num_gpus)]))

    def caption(layer: int) -> str:
        t = per_layer.loc[layer]
        slow = int(t.idxmax())
        return (f"{title} · layer {layer + 1} of {C.NUM_LAYERS}: {len(active[active.layer_id == layer])} experts used · "
                f"GPU {slow} is slowest (red), the other GPUs wait for it")

    fig = make_subplots(rows=2, cols=1, row_heights=[0.66, 0.34], vertical_spacing=0.1)
    fig.add_trace(_empty_slots(cfg), row=1, col=1)
    fig.add_trace(go.Scatter(x=ax, y=ay, mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="square", size=size, color=IDLE)), row=1, col=1)
    fig.add_trace(lit(0), row=1, col=1)
    fig.add_trace(bars(0), row=2, col=1)
    _grid_axes(fig, cfg)
    fig.update_yaxes(title="Work this layer (÷ speed)", range=[0, ymax * 1.3], row=2, col=1)
    fig.frames = [go.Frame(name=str(layer), data=[lit(layer), bars(layer)], traces=[2, 3],
                           layout=dict(title_text=caption(layer))) for layer in range(C.NUM_LAYERS)]
    player = _player(list(range(C.NUM_LAYERS)), "layer", [l + 1 for l in range(C.NUM_LAYERS)])
    player["sliders"][0]["active"] = 0
    fig.update_layout(title_text=caption(0), margin=dict(l=70, r=20, t=50, b=40), **player)
    return fig
