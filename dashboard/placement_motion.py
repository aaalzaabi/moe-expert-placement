"""Smooth, self-playing animation of the placement rule (Guided story, scene 3).

Each expert flies from the model into its slot on a simulated GPU in the exact order the rule places it
(placement_trace), and each GPU's busy-time bar grows as experts land. The numbers are the same as the
Plotly step-through (gpu_viz.fig_placement_animation); this module only adds motion.

Rendered as a small canvas page through streamlit.components.v1.html. No external scripts.
"""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

import ui
from gpu_viz import DEMAND_SCALE, EMPTY, busy_time, grid_side
from ui import C

HEIGHT = 500  # iframe height in px; the canvas fits itself inside it


def motion_data(trace: pd.DataFrame, cfg: C.SimConfig, ymax: float | None = None) -> dict:
    """Everything the page needs, as plain JSON-able lists (one entry per placement step)."""
    demand = trace.train_demand.to_numpy(float)
    final = busy_time(trace, cfg)
    return {
        "gpu": trace.gpu_id.astype(int).tolist(),
        "slot": trace.slot.astype(int).tolist(),
        "layer": trace.layer_id.astype(int).tolist(),
        "expert": trace.expert_id.astype(int).tolist(),
        "demand": [round(float(d), 3) for d in demand],
        "speeds": [float(s) for s in cfg.gpu_speeds],
        "side": grid_side(cfg),
        "cmax": max(1.0, float(np.quantile(demand, 0.99))),
        "ideal": float(demand.sum() / sum(cfg.gpu_speeds)),
        "ymax": float(ymax or final.max()),
        "colors": [ui.GPU_COLORS[g % len(ui.GPU_COLORS)] for g in range(cfg.num_gpus)],
        "scale": DEMAND_SCALE,
        "empty": EMPTY,
        "green": ui.GREEN,
    }


def placement_motion_html(trace: pd.DataFrame, cfg: C.SimConfig, title: str, order_note: str,
                          ymax: float | None = None) -> str:
    data = json.dumps(motion_data(trace, cfg, ymax), separators=(",", ":"))
    return (_PAGE.replace("__DATA__", data)
                 .replace("__TITLE__", html.escape(title))
                 .replace("__ORDER__", html.escape(order_note)))


_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  :root { --bg:#0b1220; --panel:#111b2e; --line:#1e2b45; --text:#d6e0ee; --muted:#94a3b8; --accent:#2dd4bf; }
  html, body { margin:0; background:var(--bg); color:var(--text);
               font:13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  #wrap { display:flex; flex-direction:column; height:100vh; }
  canvas { display:block; width:100%; flex:1; min-height:0; }
  .bar { display:flex; align-items:center; gap:8px; flex-wrap:wrap; padding:6px 2px 2px; }
  button { background:#13213a; color:var(--text); border:1px solid var(--line); border-radius:6px;
           padding:4px 10px; font:inherit; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button[aria-pressed="true"] { border-color:var(--accent); color:var(--accent); }
  #cap { color:var(--muted); flex:1; min-width:220px; font-variant-numeric:tabular-nums; }
  #cap b { color:var(--text); font-weight:600; }
</style></head>
<body><div id="wrap">
  <canvas id="c" role="img" aria-label="__TITLE__: all experts flying into four simulated GPUs, with each GPU's busy time growing below"></canvas>
  <div class="bar">
    <button id="play" type="button">❚❚ Pause</button>
    <button id="replay" type="button">↺ Replay</button>
    <span role="group" aria-label="Speed">
      <button class="sp" data-s="1" type="button" aria-pressed="true">1×</button>
      <button class="sp" data-s="2" type="button">2×</button>
      <button class="sp" data-s="4" type="button">4×</button>
    </span>
    <span id="cap" aria-live="off"></span>
  </div>
</div>
<script>
const D = __DATA__;
const N = D.gpu.length, G = D.speeds.length, SIDE = D.side;
const TOTAL = 9.0, FLIGHT = 0.65, FLASH = 0.35;      // seconds of animation time
// Slow start (the busiest experts are placed first, so you can follow them), then faster.
const launch = Array.from({length: N}, (_, i) => TOTAL * Math.pow(i / N, 0.6));
const END = launch[N - 1] + FLIGHT;
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const cv = document.getElementById('c'), cx = cv.getContext('2d');
const cap = document.getElementById('cap'), playBtn = document.getElementById('play');
let t = reduce ? END : 0, speed = 1, playing = !reduce, started = reduce, last = null, shown = new Array(G).fill(0);
let L = null;  // layout

// ---------------------------------------------------------------- colours
function hex(h) { return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); }
const STOPS = D.scale.map(([p, c]) => [p, hex(c)]);
function demandColor(d) {
  const v = Math.min(1, Math.max(0, d / D.cmax));
  for (let k = 1; k < STOPS.length; k++) {
    const [p1, c1] = STOPS[k], [p0, c0] = STOPS[k - 1];
    if (v <= p1) { const f = (v - p0) / (p1 - p0 || 1);
      return `rgb(${c0.map((c, j) => Math.round(c + (c1[j] - c) * f)).join(',')})`; }
  }
  return `rgb(${STOPS[STOPS.length - 1][1].join(',')})`;
}
const COL = D.demand.map(demandColor);

// ---------------------------------------------------------------- layout
function layout() {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr; cx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const pad = 10, top = 58, barsH = Math.max(80, Math.min(140, H * 0.3)), below = 22;
  const byW = (W - 2 * pad) / (G * SIDE + 2 * (G - 1));
  const byH = (H - top - barsH - below - 18) / SIDE;
  const cell = Math.max(2, Math.min(14, byW, byH));
  const gw = SIDE * cell, gap = 2 * cell, total = G * gw + (G - 1) * gap;
  const x0 = (W - total) / 2, gridBottom = top + gw;
  L = { W, H, cell, gw, gap, x0, top, gridBottom, barTop: gridBottom + 22, barBottom: H - below,
        src: [W / 2, 22], gx: g => x0 + g * (gw + gap) };
}
function slotXY(i) {
  const g = D.gpu[i], s = D.slot[i];
  return [L.gx(g) + (s % SIDE) * L.cell, L.top + Math.floor(s / SIDE) * L.cell];
}

// ---------------------------------------------------------------- drawing
const easeOut = x => 1 - Math.pow(1 - x, 3);
function draw() {
  const { W, H, cell, gw } = L;
  cx.clearRect(0, 0, W, H);
  cx.textAlign = 'center'; cx.textBaseline = 'middle';

  // empty slots
  for (let g = 0; g < G; g++) {
    const x = L.gx(g);
    cx.strokeStyle = D.empty; cx.lineWidth = 1;
    if (cell >= 4) for (let s = 0; s < SIDE * SIDE; s++)
      cx.strokeRect(x + (s % SIDE) * cell + 0.5, L.top + Math.floor(s / SIDE) * cell + 0.5, cell - 1.5, cell - 1.5);
    else { cx.fillStyle = '#16233b'; cx.fillRect(x, L.top, gw, gw); }
  }

  // landed and flying experts
  const work = new Array(G).fill(0); let landed = 0, latest = -1;
  const flying = [];
  for (let i = 0; i < N; i++) {
    const age = t - launch[i];
    if (age < 0) break;
    if (age >= FLIGHT) {
      const [x, y] = slotXY(i);
      cx.fillStyle = COL[i]; cx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      work[D.gpu[i]] += D.demand[i]; landed++; latest = i;
      const since = age - FLIGHT;
      if (since < FLASH) {                       // landing flash: strong for the first (busiest) experts, faint later
        const a = (1 - since / FLASH) * (i < 64 ? 0.9 : 0.25);
        cx.strokeStyle = `rgba(230,255,250,${a})`; cx.lineWidth = 1;
        const r = cell * (0.6 + 1.0 * (1 - a));
        cx.strokeRect(x + cell / 2 - r, y + cell / 2 - r, 2 * r, 2 * r);
      }
    } else flying.push([i, age / FLIGHT]);
  }
  for (const [i, f] of flying) {
    const [tx, ty] = slotXY(i), e = easeOut(f);
    const [sx, sy] = L.src, mx = (sx + tx) / 2, my = Math.min(sy, ty) - 30;      // arc above
    const x = (1 - e) * (1 - e) * sx + 2 * (1 - e) * e * mx + e * e * tx;
    const y = (1 - e) * (1 - e) * sy + 2 * (1 - e) * e * my + e * e * ty;
    const size = cell * (1 + 1.2 * (1 - e));
    cx.shadowColor = COL[i]; cx.shadowBlur = 10 * (1 - e);
    cx.fillStyle = COL[i]; cx.fillRect(x + cell / 2 - size / 2, y + cell / 2 - size / 2, size, size);
    cx.shadowBlur = 0;
  }

  // labels on top of the flying experts so they stay readable: the model they come from, then each GPU
  cx.font = '600 12px -apple-system, sans-serif';
  const label = 'OLMoE · 1,024 experts · __ORDER__', lw = cx.measureText(label).width + 22;
  cx.fillStyle = '#13213a'; cx.strokeStyle = '#2dd4bf55'; cx.lineWidth = 1;
  roundRect(L.src[0] - lw / 2, L.src[1] - 11, lw, 22, 11); cx.fill(); cx.stroke();
  cx.fillStyle = '#d6e0ee'; cx.fillText(label, L.src[0], L.src[1]);
  const fastest = Math.max(...D.speeds), slowest = Math.min(...D.speeds);
  cx.font = `600 ${Math.max(10, Math.min(12, gw / 9))}px -apple-system, sans-serif`;
  for (let g = 0; g < G; g++) {
    const tag = D.speeds[g] === fastest ? ' fastest' : D.speeds[g] === slowest ? ' slowest' : '';
    const txt = `GPU ${g} · ${D.speeds[g]}×${tag}`, tw = cx.measureText(txt).width + 10, x = L.gx(g) + gw / 2;
    cx.fillStyle = '#0b1220cc'; cx.fillRect(x - tw / 2, L.top - 20, tw, 16);
    cx.fillStyle = D.colors[g]; cx.fillText(txt, x, L.top - 12);
  }

  // busy-time bars (work ÷ speed), easing toward their target
  const ymax = D.ymax * 1.15, bh = L.barBottom - L.barTop, yOf = v => L.barBottom - bh * Math.min(1, v / ymax);
  cx.strokeStyle = '#26385a'; cx.lineWidth = 1;
  cx.beginPath(); cx.moveTo(L.x0, L.barBottom + 0.5); cx.lineTo(L.x0 + G * gw + (G - 1) * L.gap, L.barBottom + 0.5); cx.stroke();
  for (let g = 0; g < G; g++) {
    const target = work[g] / D.speeds[g];
    shown[g] += (target - shown[g]) * (playing ? 0.18 : 1);
    if (Math.abs(target - shown[g]) < 0.05) shown[g] = target;
    const bw = gw * 0.55, x = L.gx(g) + (gw - bw) / 2, y = yOf(shown[g]);
    cx.fillStyle = D.colors[g]; cx.globalAlpha = latest >= 0 && D.gpu[latest] === g && t < END ? 1 : 0.8;
    cx.fillRect(x, y, bw, L.barBottom - y); cx.globalAlpha = 1;
    cx.fillStyle = '#d6e0ee'; cx.font = '600 12px -apple-system, sans-serif';
    cx.fillText(Math.round(shown[g]).toLocaleString(), x + bw / 2, y - 9);
  }
  const iy = yOf(D.ideal);
  cx.setLineDash([4, 4]); cx.strokeStyle = D.green; cx.beginPath();
  cx.moveTo(L.x0, iy); cx.lineTo(L.x0 + G * gw + (G - 1) * L.gap, iy); cx.stroke(); cx.setLineDash([]);
  cx.fillStyle = D.green; cx.textAlign = 'left'; cx.font = '11px -apple-system, sans-serif';
  cx.fillText('perfect balance', L.x0, iy - 8);
  cx.fillStyle = '#94a3b8'; cx.textAlign = 'center';
  cx.fillText('busy time per GPU (work ÷ speed)', W / 2, L.barBottom + 12);

  caption(landed, latest, work);
}

let lastCap = '';
function caption(landed, latest, work) {
  let s;
  if (landed === 0) s = `<b>__TITLE__</b> · starting…`;
  else if (landed < N) {
    s = `<b>__TITLE__</b> · placed <b>${landed.toLocaleString()}</b> of ${N.toLocaleString()} · latest: layer ${D.layer[latest]}, ` +
        `expert ${D.expert[latest]} (demand ${D.demand[latest]}) → <b>GPU ${D.gpu[latest]}</b>`;
  } else {
    const busy = work.map((w, g) => w / D.speeds[g]), top = Math.max(...busy);
    s = `<b>__TITLE__</b> · all ${N.toLocaleString()} placed · busiest GPU <b>${Math.round(top).toLocaleString()}</b> ` +
        `vs perfect balance ${Math.round(D.ideal).toLocaleString()}`;
  }
  if (s !== lastCap) { cap.innerHTML = s; lastCap = s; }
}
function roundRect(x, y, w, h, r) {
  cx.beginPath(); cx.moveTo(x + r, y); cx.arcTo(x + w, y, x + w, y + h, r); cx.arcTo(x + w, y + h, x, y + h, r);
  cx.arcTo(x, y + h, x, y, r); cx.arcTo(x, y, x + w, y, r); cx.closePath();
}

// ---------------------------------------------------------------- loop and controls
function frame(now) {
  if (last !== null && playing && started) t = Math.min(END + FLASH, t + (now - last) / 1000 * speed);
  last = now; draw();
  if (t >= END + FLASH && playing) setPlaying(false, true);
  requestAnimationFrame(frame);
}
function setPlaying(p, done) {
  playing = p;
  playBtn.textContent = p ? '❚❚ Pause' : (done ? '▶ Play' : '▶ Resume');
}
playBtn.onclick = () => { if (t >= END + FLASH) t = 0; started = true; setPlaying(!playing); };
document.getElementById('replay').onclick = () => { t = 0; shown.fill(0); started = true; setPlaying(true); };
document.querySelectorAll('.sp').forEach(b => b.onclick = () => {
  speed = +b.dataset.s; document.querySelectorAll('.sp').forEach(o => o.setAttribute('aria-pressed', o === b));
});
// Start when the animation scrolls into view, so it is not over before anyone sees it.
if (!reduce && 'IntersectionObserver' in window) {
  new IntersectionObserver((es, ob) => { if (es.some(e => e.isIntersecting)) { started = true; ob.disconnect(); } },
                           { threshold: 0.4 }).observe(cv);
} else started = true;
if (reduce) setPlaying(false, true);
new ResizeObserver(layout).observe(cv); layout();
requestAnimationFrame(frame);
</script></body></html>"""
