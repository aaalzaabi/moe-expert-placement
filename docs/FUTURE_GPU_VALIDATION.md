# Future validation on NVIDIA GPUs

Status: nothing in this project has been run on NVIDIA hardware. The steps below turn each hypothetical
parameter into a measurement and then test the planner's predictions against real serving.

## 1. Calibrate the simulator's parameters

| Parameter (now hypothetical) | How to measure | Target hardware |
|---|---|---|
| `compute_ms_per_assignment`, `gpu_speeds` | Time one OLMoE expert MLP (2048→1024→2048 SwiGLU, BF16) for 1…512 tokens with CUDA events; fit time vs tokens per GPU type | A100 / H100 / L40S mix for heterogeneity |
| `activation_bytes_per_assignment` | 2048 × 2 B = 4,096 B per token per direction in BF16 (verify serving dtype) | — |
| `bandwidth_gb_per_s`, `link_latency_ms` | `nccl-tests` all-to-all and send/recv per GPU pair; record NVLink vs PCIe vs InfiniBand | 4–8 GPU node, 2 nodes |
| `link_model` | Compare concurrent-transfer throughput with `directed` vs `shared_port` predictions | same |
| Expert copy time | `cudaMemcpyPeerAsync` / NCCL send of 12.583 MB tensors, alone and under inference load | same |
| Memory | `torch.cuda.memory_allocated` with resident + staged experts, plus KV cache | same |

Write the measured values into a new `SimConfig` preset (e.g. `h100_nvlink`) and re-run the sweep with
`scripts/run_experiment.py`.

## 2. Collect representative routing

* ≥ 100 prompts per category, plus decode tokens (generate 64–256 tokens with `output_router_logits`).
* Run `scripts/profile_routing.py` (CUDA is auto-detected) with warm-up; keep the train / held-out split by request.

## 3. Validate predictions

1. Implement custom expert placement in an expert-parallel serving stack (e.g. vLLM or SGLang with expert parallelism) driven by `moeplace.placement`.
2. Replay held-out traces under static vs demand-aware placement; measure p50/p95 latency and throughput.
3. Compare measured vs simulated per-GPU load and latency; report error (MAPE) per metric.
4. Execute a live migration (stage → atomic switch → free old copies) under load; measure activation time and p95 impact; test rollback by killing a transfer.
5. Evaluate the cost-aware controller online with a rolling forecast window; compare decisions to the hindsight-optimal choice.

## 4. Acceptance criteria (proposed)

* Simulated vs measured p95 within ±20% across bandwidth settings.
* Demand-aware placement reduces measured peak-GPU busy time on held-out traffic.
* No request observes a partially migrated layout; failed migrations leave serving unaffected.

Until these steps are done, all latency, makespan and migration numbers must be presented as simulated.
