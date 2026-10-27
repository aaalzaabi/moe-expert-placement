# Assumptions

All parameters live in `moeplace/config.py` (`SimConfig`). **Units:** bandwidth in **GB/s = 10⁹ bytes/s**
(never Gbps; 100 Gbps Ethernet ≈ 12.5 GB/s), time in simulated milliseconds, sizes in bytes (MB = 10⁶ B).

## Facts (not assumptions)

| Quantity | Value | Source |
|---|---|---|
| MoE layers / experts per layer / top-k | 16 / 64 / 8 | `config.json` |
| Layer-specific experts | 1,024 | 16 × 64 |
| Expert weights | 3 × 2048 × 1024 BF16 = **12,582,912 B** (12.583 MB) | safetensors headers |
| Total expert weights | 12.885 GB of 13.838 GB checkpoint | `phase4f/memory`, index.json |
| Routing counts, input tokens, MPS forward time | measured | `workload_routing.csv`, `workload_timings.csv` |

## Hypothetical simulation parameters

| Parameter | Default | Notes |
|---|---|---|
| `gpu_speeds` | (1.0, 1.3, 0.8, 1.6) | Relative throughput multipliers; not real GPUs |
| `gateway_gpu` | 0 | Holds attention/router; dispatches activations |
| `placement_capacity_slots` | 256 | Experts per GPU in a placement (1024 / 4) |
| `staging_capacity_slots` | 512 | Resident + incoming copies during migration |
| `gpu_memory_gb` | None | Optional expert-weight byte budget (excludes ~0.95 GB shared weights, activations, KV cache) |
| `compute_ms_per_assignment` | 0.5 ms | One token through one expert on a speed-1.0 GPU |
| `activation_bytes_per_assignment` | 4,096 B | Sent each way per assignment (a 2048-dim BF16 hidden state) |
| `bandwidth_gb_per_s` | 12 | Per directed link (scenario axis: 3, 12, 25, 100) |
| `link_latency_ms` | 0.1 ms | Added to every transfer |
| `link_model` | `directed` | Independent FIFO per (src, dst). `shared_port` = one send + one receive port per GPU |
| `expert_bytes` | 12,582,912 | Legacy experiments used an 8,000,000 B placeholder |
| `arrival_gap_ms` / `arrival_process` | 200 ms / fixed | `poisson` uses `seed` |
| `migration_start_ms` | 100 ms | Approval time in the integrated simulation |
| `min_makespan_gain_pct` / `max_p95_regression_pct` | 2% / 0% | Controller thresholds, fixed a priori |

## Multi-site model (moeplace/multisite.py, usecases.py) — all HYPOTHETICAL

| Parameter | Default | Notes |
|---|---|---|
| `window_s` | 900 s | Planning window W (15 min); horizon = 4 windows |
| `assignments_per_s` | 5 × 10⁴ | Token-expert assignments per second on a speed-1.0 device |
| `activation_bytes` | 4,096 B | Per assignment per direction |
| `inter_site_gb_per_s` / `inter_site_rtt_ms` | 1.25 GB/s / 60 ms | Per site pair and direction; RTT paid once per layer with a remote expert |
| `intra_site_gb_per_s` / `storage_gb_per_s` | 12 / 2 GB/s | Migration time only |
| `util_cap` | 0.7 | Locality mode: minimise latency while utilisation ≤ cap; penalty 10,000 per unit above |
| Site traffic | peak 2,500–3,000 req / 15 min, 400 tokens/request, trough 25% | Daily cosine cycle in local time, gamma volume noise (CV 8%), Dirichlet mix noise |
| Agent | gain ≥ 2%, transfer ≤ 5% of horizon, approval > 50 GB or residency change, refit if realised regret > 2%, stickiness 5% | `AgentConfig` |

Latency model: added ms per token = 16 × RTT × (1 − (1 − r)^8), assuming the 8 experts of a layer are remote
independently with probability r (the site's remote share).

## Modelling simplifications

1. **Prefill only.** Each request is one forward pass over its prompt; no decode tokens, no KV cache.
2. **Only expert MLPs are timed.** Attention, router and embeddings cost zero simulated time.
3. **Per-layer aggregation.** All tokens of a request routed to one expert in one layer form one task.
4. **Linear compute cost.** Time ∝ assignments ÷ speed; no batching efficiency, kernel launch or memory-bandwidth effects.
5. **FIFO reservation in event order.** A GPU or link cannot back-fill an idle gap with a task processed later.
6. **Network.** Point-to-point links with a fixed bandwidth and latency; no topology, switches, NVLink/PCIe distinction or congestion control.
7. **Migration.** All moved experts are copied (old copies stay resident); activation is a single atomic switch after the last copy; requests keep the layout active at their arrival; old copies are never freed in the simulation.
8. **Failures.** Only staging-capacity failure and injected copy failure are modelled; link time reserved by aborted copies is not reclaimed.
9. **Precision.** Migration sizes assume BF16 serving; the profiling ran in FP32.

## What the dashboard is not

It does not load, move or execute experts on GPUs. "GPU" always means a simulated device. Approve/reject
actions change only browser-session state.
