# Forecasting, placement regret and decision-aware drift

Code: `moeplace/workload.py`, `moeplace/forecast.py`. Tests: `tests/test_forecast.py`.
Run: `scripts/run_usecases.py` → `outputs/runs/usecases-<id>/<use case>/forecast_*.csv, drift.csv`.

## 1. What is forecast

**Target.** A time-indexed, site-aware expert-demand tensor

    Y[t, s, e] = token-to-expert assignments that originate at site s and are routed to the
                 layer-specific expert e (e = layer × 64 + expert, 0…1023) during window t

* **Unit:** assignments per window. Divide by the window length for activations per second.
* **Why this quantity:** it carries everything the planner needs. Summed over e it gives each site's
  traffic volume (tokens × 16 layers × 8). Normalised over e it gives each site's expert-popularity
  distribution. Kept per site, it captures the case the feedback described: expert 3 popular in one region,
  expert 7 in another. A single global popularity vector loses that, and is kept only as a baseline
  (`global_ewma`).
* **Window:** W = 15 min. **Season:** P = 96 windows (one day). **Horizon:** H = 4 windows (1 hour), the time
  a placement stays active before the next decision. Placement uses the mean forecast over the horizon.
* **Data:** per-category routing profiles P[c, e] are measured on OLMoE (6 prompts; additive smoothing of 0.5
  assignments per expert so unseen experts keep a small probability). Site traffic (request rate with a daily
  cycle in local time, category mix, window-to-window noise, events) is synthetic and defined per use case.
  So the tensor is synthetic traffic over measured routing.
* **Training / evaluation split:** days 1–2 (windows 0–191) are used to choose smoothing parameters; days 3–4 are
  evaluation only (rolling origin, a decision every hour). Parameters are never tuned on evaluation days
  (`test_fit_ignores_data_after_train_end`).

## 2. Models

| Model | Forecast of Y[t+h] made at t | Parameters (grid-searched on training WAPE) |
|---|---|---|
| naive_last | Y[t] | — |
| seasonal_naive | Y[t+h−P] | — |
| ewma | exponentially weighted level of Y | α |
| **site_ewma** | V[t+h−P, s] · R[t, s] · π̂[t, s, e]: per-site volume from yesterday scaled by an EWMA(β) of the day-over-day ratio, times a per-site EWMA(α) of expert shares | α, β |
| global_ewma | same volume model, but one pooled share vector for all sites | α, β |

All forecasts are causal (`test_causal`). On noise-free periodic demand the seasonal models are exact
(`test_seasonal_models_exact_on_periodic_noise_free_demand`).

## 3. Accuracy metrics

* **WAPE** = Σ|Y − Ŷ| / ΣY over sites and experts (selection loss; robust to many low-demand experts)
* **Popularity error** = mean over sites of the total-variation distance between actual and forecast expert
  shares (0 = identical, 1 = disjoint)
* **Volume error** = mean over sites of |V − V̂| / V
* **Decision-weighted WAPE** = WAPE restricted to fragile experts (below)

## 4. Placement regret

At each decision time t, with the same planner and constraints:

    P̂  = plan(mean_h Ŷ[t+h])          plan from the forecast
    P* = plan(mean_h Y[t+h])          plan with perfect foresight
    regret_abs = mean_h [ Φ(P̂; Y[t+h]) − Φ(P*; Y[t+h]) ]
    regret_rel = regret_abs / max(mean_h Φ(P*; Y[t+h]), floor)
    aggregate regret = Σ regret_abs / Σ Φ(P*)        (reported in summaries)

Φ is the planner's objective (`moeplace/multisite.py`):
* balance mode: Φ = J, the bottleneck utilisation (max over devices and inter-site links);
* locality mode (geo use cases): Φ = added cross-site latency per token (ms) + 10,000 × max(0, J − 0.7).

The floor (10 ms in locality mode, 0 in balance mode) stops night windows, where the oracle needs almost no
cross-site traffic, from turning tiny absolute differences into huge percentages.

Regret is **planner-relative**. It isolates the damage caused by forecast error from the planner's own
optimality gap. It can be slightly negative because the greedy planner is not exactly optimal.

Two further decision diagnostics:
* **Agreement / host Jaccard:** how many experts keep the same device, and the overlap of copies, between P̂ and P*.
  They are low (≈0.13 / 0.18) even when regret is near zero: many layouts are almost equally good.
* **Fragile experts:** experts whose best and second-best device scores in the oracle plan are within 1%.
  Errors on these flip decisions. `fragile_demand_share` is the share of demand they carry.

## 5. Decision-aware drift

Each decision point is classified by (WAPE > 10%) × (regret_rel > 2%):

| | regret ≤ 2% | regret > 2% |
|---|---|---|
| **WAPE ≤ 10%** | fine | **harmful small error**: an error-based monitor misses it |
| **WAPE > 10%** | **harmless error**: an error-based monitor raises a false alarm | caught by both |

The operations agent re-fits its forecaster when the **realised regret** of its previous decision exceeds
2% (`AgentConfig.tau_regret`). Realised regret becomes observable one horizon later, so detection lags by one
hour by design.

## 6. Results (simulated; `usecases-f497898bea`)

Multi-region use case (UC1), 47 decision points:

| Model | WAPE | Popularity error | Aggregate regret |
|---|---|---|---|
| site_ewma (α 0.05, β 0.1) | 0.047 | 0.004 | 23.0% |
| global_ewma | 0.105 | 0.048 | 29.6% |
| seasonal_naive | 0.048 | 0.006 | 29.7% |
| ewma | 0.093 | 0.008 | 50.9% |
| naive_last | 0.095 | 0.009 | 55.0% |

* Site-aware popularity is about 11× more accurate than one global vector, and gives the lowest regret.
* Under the coding surge (UC4) the ranking changes: seasonal models assume yesterday repeats and fail
  (seasonal-naive regret 102%); the plain EWMA adapts fastest (7.6%), and site_ewma reaches 15.2%.
* Error is a poor alarm. For site_ewma in UC1, 15 of 47 points were harmful small errors and none were harmless
  errors; the WAPE–regret correlation was 0.62 for site_ewma, 0.55 for global_ewma and only 0.13–0.22 for the other three.

Even the best regret (≈23%) is large in locality mode. The objective is sensitive near the utilisation cap,
where a small demand difference changes which replicas fit. That is a real property of capped locality
placement, and the main reason the agent re-plans hourly with hysteresis.
