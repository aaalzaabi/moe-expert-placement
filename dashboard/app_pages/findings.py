import streamlit as st

import ui
from ui import C

ui.page_header("What we learned and the limits",
               "The research question, the method, the findings, and what is still uncertain.",
               ("measured", "simulated"), step="Part 4 · Step 8 of 8")
ui.explain(
    "A summary written for a professor or reviewer. The numbers are read live from the saved result files, so they "
    "always match the other pages.",
    ["**Research question** at the top, then **method** (left) and **how it was tested** (right).",
     "**Current findings:** four headline numbers and four numbered conclusions.",
     "**Limitations** and **threats to validity:** what could make the results wrong or not general.",
     "**Next experiments:** what to do next to make the results stronger."],
    "The approach works in simulation, but the sample is small and the hardware is assumed — real GPU tests are the "
    "next step.", expanded=False)

held = ui.csv("heldout_comparison.csv").set_index("strategy")
inf = ui.csv("simulated_inference_comparison.csv").set_index("strategy")
pol = ui.csv("policy_comparison.csv").set_index("policy")
dec = ui.csv("policy_decision.csv").iloc[0]
cal = ui.phase4f("calibrated").pivot(index="bandwidth_gb_per_s", columns="policy", values="p95_latency_ms")
cal_imp = 100 * (cal["static"] - cal["migrate"]) / cal["static"]
tim = ui.timings()

with st.container(border=True):
    st.markdown("#### Research question")
    st.markdown("> Can measured expert-routing demand from a real MoE model be used to (1) place layer-specific experts "
                "on heterogeneous GPUs better than demand-agnostic baselines, and (2) decide **when** relocating experts "
                "is worth its weight-transfer cost?")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True, height="stretch"):
        st.markdown("#### Baselines and approach")
        st.markdown("""
- **Static** — round-robin by expert index (ignores demand and speed).
- **Load-balanced** — greedy by measured demand, ignores GPU speed.
- **Demand-aware (proposed)** — greedy, highest demand first, onto the GPU minimising *(load + demand) ÷ speed*, under slot capacity.
- **Migration policies** — never, always, and **cost-aware**: approve only if a training-request forecast shows makespan gain ≥ threshold and p95 regression ≤ limit, and staging memory suffices.
""")
with c2:
    with st.container(border=True, height="stretch"):
        st.markdown("#### Evaluation methodology")
        st.markdown(f"""
- Routing **measured** on OLMoE-1B-7B (16 × 64 experts, top-8) for {len(tim)} prompts on Apple MPS.
- Split by request: train {{0, 2, 4}} plans placements and drives the controller; held-out {{1, 3, 5}} only evaluates.
- Discrete-event simulator replays measured routing: per-GPU FIFO compute, FIFO links, layer barriers, placement pinned at request arrival, atomic activation after all copies.
- Expert size calibrated from checkpoint headers: {C.CHECKPOINT_EXPERT_BYTES / C.MB:.3f} MB (BF16). GPU speeds and network are **hypothetical**.
""")

st.markdown("#### Current findings")
f1, f2, f3, f4 = st.columns(4)
f1.metric("Held-out peak load, demand-aware vs static",
          f"{-ui.pct(held.loc['demand_aware', 'peak_test_load'], held.loc['static', 'peak_test_load']):+.1f}%",
          border=True, help="Load proxy (Phase 3B). Load-balanced: "
          f"{-ui.pct(held.loc['balanced', 'peak_test_load'], held.loc['static', 'peak_test_load']):+.1f}%")
f2.metric("Simulated p95, demand-aware vs static (4A batch)",
          f"{-ui.pct(inf.loc['demand_aware', 'p95_latency_ms'], inf.loc['static', 'p95_latency_ms']):+.1f}%", border=True)
f3.metric("Always-migrate held-out p95 (8 MB, 12 GB/s)",
          f"{-ui.pct(pol.loc['always_migrate', 'p95_latency_ms'], pol.loc['never_migrate', 'p95_latency_ms']):+.1f}%",
          border=True, help=f"Controller decision: {dec.decision} (forecast p95 {dec.forecast_p95_regression_pct:+.1f}%)")
f4.metric("Migrate p95 at 100 GB/s vs 3 GB/s (12.583 MB)", f"{-cal_imp.loc[100]:+.1f}% / {-cal_imp.loc[3]:+.0f}%",
          border=True, help="Negative = migration lowers p95")

st.markdown(f"""
1. **Placement.** On held-out prompts, demand-aware placement lowers the peak per-GPU load proxy by
   {ui.pct(held.loc['demand_aware', 'peak_test_load'], held.loc['static', 'peak_test_load']):.1f}% vs static and brings imbalance
   from {held.loc['static', 'test_imbalance']:.2f} to {held.loc['demand_aware', 'test_imbalance']:.2f}. Most of the gain over load-balancing comes from
   using GPU speed; with identical GPUs the two greedy methods coincide.
2. **Simulated latency.** With all requests at once (Phase 4A) demand-aware cuts simulated p95 by
   {ui.pct(inf.loc['demand_aware', 'p95_latency_ms'], inf.loc['static', 'p95_latency_ms']):.1f}% and makespan by
   {ui.pct(inf.loc['demand_aware', 'makespan_ms'], inf.loc['static', 'makespan_ms']):.1f}%, at
   {-ui.pct(inf.loc['demand_aware', 'communication_mb'], inf.loc['static', 'communication_mb']):+.1f}% communication.
   With staggered arrivals (Phase 4D/4E) the steady-state benefit is smaller (~1.3% makespan) because GPUs are rarely saturated.
3. **Migration is bandwidth-bound.** Moving 776 experts (6.2 GB at 8 MB, 9.8 GB at 12.583 MB) contends with inference traffic.
   Held-out p95 gets worse below ~15–25 GB/s and better above it (+{cal_imp.loc[100]:.1f}% at 100 GB/s, calibrated size).
4. **The controller is conservative.** It rejected migration in the saved Phase 4E scenario, which was correct on held-out p95
   ({-ui.pct(pol.loc['always_migrate', 'p95_latency_ms'], pol.loc['never_migrate', 'p95_latency_ms']):+.1f}%). In the
   calibrated sweep it also rejects at 25–100 GB/s where held-out p95 improves, because its 3-request training forecast is
   dominated by the one request in flight during copying, and the p95 limit is 0%.
""")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True, height="stretch"):
        st.markdown("#### Known limitations")
        st.markdown("""
- **6 prompts, 6–10 tokens each**, prefill only (44 tokens in total); about a quarter of experts get zero assignments within each 3-prompt split, so demand estimates are noisy.
- **3 held-out requests**: p95 ≈ max; no confidence intervals; percentages of a few points are not significant.
- **Hypothetical hardware**: GPU speeds, 0.5 ms/assignment, 4 KB/assignment, link bandwidth and latency are not measured.
- **Load proxy ≠ latency**: token counts ÷ speed ignore batching efficiency, kernel launch overhead and memory bandwidth.
- **Simplified network**: independent FIFO per directed link by default (no NVLink/PCIe topology, no switch contention); the `shared_port` option is a sensitivity check, not a calibrated model.
- **No decode phase, attention, KV cache or activation memory** in the simulator; expert weights only.
- Measured MPS timings are single-device FP32 and include first-call warm-up for request 0.
""")
with c2:
    with st.container(border=True, height="stretch"):
        st.markdown("#### Threats to validity")
        st.markdown("""
- **Internal:** resources are reserved in event order without back-filling; replay of measured traces instead of new inference.
- **Train/test leakage:** none in the controller or the held-out placement (verified by tests); Phase 3A in-sample placement is labelled as such.
- **Construct:** the cost-aware rule optimises a forecast from 3 requests; thresholds were not tuned on held-out data, and should not be.
- **External:** one model, one prompt mix, one hypothetical 4-GPU cluster; results may not transfer to other MoE models or real interconnects.
- **Statistical:** single deterministic run per scenario; the runner supports Poisson arrivals with seeds for replicates.
""")

with st.container(border=True):
    st.markdown("#### Next experiments")
    st.markdown("""
1. Profile ≥ 100 prompts (`scripts/profile_routing.py --prompt-set extended`, with warm-up) and include generated tokens (decode routing).
2. Replicate simulations with Poisson arrivals over many seeds; report means and 95% CIs (`scripts/run_experiment.py --arrival-process poisson --seed 1 2 3 …`).
3. Make the controller forecast robust: rolling window of recent requests, p95 over more requests, or expected-cost formulation with a payback horizon.
4. Incremental / layer-by-layer migration to cap link contention instead of one atomic 776-expert move.
5. Calibrate compute and network parameters on real NVIDIA GPUs (see docs/FUTURE_GPU_VALIDATION.md).
""")
st.info(ui.SIM_DISCLAIMER, icon=":material/model_training:")

ui.step_footer("app_pages/findings.py")
