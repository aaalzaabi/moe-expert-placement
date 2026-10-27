# Experiment methodology

## 1. Measurement (real)

* Model: `allenai/OLMoE-1B-7B-0125-Instruct`, loaded locally in FP32 with eager attention on Apple MPS.
* Workload: 6 prompts, 2 each of AI / coding / general; raw tokenizer (no chat template); 6–10 input tokens.
* One forward pass per prompt with `output_router_logits=True`; per layer, top-8 of the 64 router logits per
  token; counts aggregated per (request, layer, expert). Invariant (verified): each (request, layer) sums to
  `tokens × 8`.
* Timing: wall clock around the forward pass with `torch.mps.synchronize()`; no warm-up (request 0 includes it).
* Prefill only: no generated tokens, so decode-phase routing is not represented.

## 2. Train / held-out protocol

| Split | Requests | Used for |
|---|---|---|
| Train | 0 (ai), 2 (coding), 4 (general) | Placement demand; controller forecast |
| Held-out | 1 (ai), 3 (coding), 5 (general) | Evaluation only |

Guarantees enforced in code and tests: overlapping ids raise `ConfigError`; placements and controller
decisions are unchanged when held-out routing is corrupted (`test_placement_ignores_test_demand`,
`test_decision_never_uses_heldout_requests`). Phase 3A (`gpu_placements.csv`) is the exception: it uses all
six requests and is labelled in-sample.

## 3. Metrics

| Metric | Definition | Caveat |
|---|---|---|
| Normalized load (load proxy) | Σ token assignments on a GPU ÷ its relative speed | Not utilisation, not time |
| Imbalance | max ÷ mean normalized load across GPUs | |
| p50 / p95 latency | `numpy.percentile` (linear) of request completion − arrival | n = 3 → p95 ≈ max |
| Makespan | Last request finish time (absolute simulated ms) | |
| Communication | Bytes on links (activations both ways + migration) | |
| Remote expert calls | Expert invocations on a GPU other than the gateway | |
| Activation time | Last copy finish − approval time | |
| Forecast makespan gain / p95 regression | Train-request simulations, migrate vs not | In-sample w.r.t. placement |

## 4. Controller rule

Approve iff (a) the staging check passes, (b) forecast makespan gain ≥ `min_makespan_gain_pct` (2%),
(c) forecast p95 regression ≤ `max_p95_regression_pct` (0%). Thresholds are fixed a priori and must not be tuned
on held-out results.

**Observed behaviour (calibrated 12.583 MB, bandwidth sweep, `outputs/runs/calibrated_bw_sweep-*`):**

| GB/s | Forecast makespan gain | Forecast p95 change | Decision | Held-out p95 change if migrated |
|---|---|---|---|---|
| 3 | −42.2% | +194.1% | reject | +173.4% (worse) |
| 12 | +4.1% | +47.8% | reject | +23.6% (worse) |
| 25 | +4.1% | +26.8% | reject | −2.2% (better) |
| 100 | +4.1% | +6.5% | reject | −12.3% (better) |

The rule never approves in this scenario because the 3-request forecast's p95 is set by the single training
request that is in flight while the copies saturate its links. This is a property of the forecast design and
the sample size, not of the bandwidth. It motivates a larger forecast window and an expected-cost formulation.

## 5. Reproducibility

* Every saved simulated CSV is regenerated bit-for-bit by `moeplace` (`tests/test_reproduction.py`), with the
  parameters listed in `moeplace/registry.py`.
* The simulator is deterministic; Poisson arrivals use `numpy.random.default_rng(seed)`.
* `scripts/run_experiment.py` writes each run to `outputs/runs/<name>-<hash>/` where the hash covers the full
  configuration and the routing file's SHA-256. Re-running identical parameters is skipped (`--overwrite` to redo).
  `metadata.json` records config, command, routing hash, package versions and the assumptions statement.
* Sweeps write `summary.csv` + `sweep.json`; failed points are recorded and the sweep continues.

### Reproducing the saved results

```bash
.venv/bin/python -m unittest tests.test_reproduction -v            # all saved CSVs, in ~1 s
.venv/bin/python scripts/run_experiment.py --preset phase4e_legacy  # Phase 4E as a new run
.venv/bin/python scripts/run_experiment.py --preset phase4f_network --name net8mb --bandwidth 3 12 25 100
.venv/bin/python scripts/run_experiment.py --preset phase4f_calibrated --name cal --bandwidth 3 12 25 100
```

### Suggested next runs

```bash
.venv/bin/python scripts/run_experiment.py --arrival-process poisson --seed $(seq 1 30) --bandwidth 12 25 100
.venv/bin/python scripts/run_experiment.py --link-model shared_port --bandwidth 3 12 25 100
.venv/bin/python scripts/run_experiment.py --max-p95-regression-pct 10 --bandwidth 12 25 100
.venv/bin/python scripts/profile_routing.py --prompt-set extended --warmup 1   # then pass --routing <dir>
```

## 6. Is the placement better than chance? (validation against random placements)

`moeplace.placement.validate_vs_random()` places the 1,024 experts at random 10,000 times (seed 0, same 256-slot
limit per GPU) and measures the busiest GPU's held-out load proxy each time. It also computes the theoretical best
(total held-out work ÷ total GPU speed; no placement can beat it) and a hindsight plan made with the held-out demand.

| Placement | Held-out peak load | Random placements beaten | Gap to best possible |
|---|---|---|---|
| Static | 896 | 36% | 49.6% |
| Load-balanced | 869 | 57% | 45.0% |
| **Demand-aware** | **632** | **100%** (best random: 716) | **5.6%** |
| Best possible (bound) | 599 | — | 0% |

Demand-aware placement beats every random layout on questions it never saw, so its advantage is not luck. Static and
load-balanced are no better than a typical random layout here because they ignore GPU speed. Shown on dashboard
page 3 and checked by `tests/test_placement.py::TestBetterThanRandom`.

## 7. Explaining controller decisions

`moeplace.policy.explain()` turns every decision into: a headline; the three checks with plain meanings (timing
checks are marked "not checked" when the move is impossible); the cause (e.g. which training question was running
during the copy and how much slower it became); and what would change the decision. Each counterfactual is found by
re-running the controller (relaxed p95 limit, faster networks up to 1,000 GB/s, other start times, more staging room),
and uses training questions only. Dashboard page 5 also shows an after-the-fact check on held-out questions, clearly
marked as not used for the decision. The multi-site agent's rejected or human-denied proposals are explained the same
way (`dashboard/ms_views.py::explain_proposal`).

Saved Phase 4E scenario, as explained: REJECTED because question 0 was already running when copying began and took
+35% longer; question 4, which arrived after the switch, got 16% faster. Allowing a 33% p95 regression would approve
it; a faster network alone would not (even at 1,000 GB/s the in-flight question is still +2.4% slower).
