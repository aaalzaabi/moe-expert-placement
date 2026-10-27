"""Site-aware demand forecasting, placement regret and decision-aware drift detection.

Forecast specification (proposal feedback 11)
---------------------------------------------
target      Y[t, s, e]: assignments per window from site s to expert e (see workload.py)
window      W = 900 s (15 min);   season P = 96 windows (one day)
horizon     H = 4 windows (1 hour): the period one placement stays active before the next decision
decision    at the end of window t, forecast Y[t+1..t+H] and plan on their mean
training    windows [0, train_end): used to choose smoothing parameters (grid search on WAPE)
loss        WAPE = sum |Y - Yhat| / sum Y   (scale-free; robust to the many low-demand experts)
also        MAE, popularity error (total-variation distance between per-site expert shares),
            volume error (|V - Vhat| / V per site)

Models
------
naive_last        Yhat[t+h] = Y[t]
seasonal_naive    Yhat[t+h] = Y[t+h-P]
ewma              exponentially weighted level of Y (alpha); no seasonality
site_ewma         SITE-AWARE decomposition: volume V[t,s] = sum_e Y forecast as V[t+h-P,s] * R[t,s]
                  (R = EWMA(beta) of V[tau]/V[tau-P]); shares pi[t,s,e] = Y/V forecast by EWMA(alpha)
global_ewma       same volume model, but ONE expert-popularity vector pooled over all sites (the "global
                  popularity" design the feedback warned about) -- kept as a comparison baseline

Placement regret (feedback 12)
------------------------------
    P_hat  = plan(mean_h Yhat[t+h]),   P_star = plan(mean_h Y[t+h])   (same planner, perfect foresight)
    regret_abs = mean_h [ Phi(P_hat; Y[t+h]) - Phi(P_star; Y[t+h]) ]  Phi = the planner's objective
    regret_rel = regret_abs / max(mean_h Phi(P_star; Y[t+h]), floor)   (J, or latency with a utilisation cap)
floor = 0 in balance mode; 10 ms in locality mode, so that night-time windows where the oracle needs almost no
cross-site traffic (Phi close to 0) do not turn tiny absolute differences into huge percentages. Summaries also
report the aggregate ratio sum(regret_abs) / sum(Phi_oracle), which is robust to small denominators.
Regret is planner-relative: it isolates the damage caused by forecast error from the planner's own
optimality gap. Small negative values are possible because the greedy planner is not exactly optimal.

Decision-aware drift
--------------------
A forecast error matters only if it changes the decision enough to raise J. The analysis classifies each
decision point by (WAPE > tau_err) x (regret_rel > tau_regret): "harmless error" (high error, no regret)
and "harmful small error" (low error, real regret). Re-planning / re-fitting is triggered on regret, not error.
Fragile experts are those whose best and second-best devices score within 1% in the oracle plan; the
decision-weighted WAPE measures error on exactly those experts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools

import numpy as np
import pandas as pd

from . import multisite as M

REGRET_FLOOR_LOCALITY_MS = 10.0


def regret_floor(util_cap) -> float:
    return 1e-12 if util_cap is None else REGRET_FLOOR_LOCALITY_MS


def hosts_jaccard(a: M.Plan, b: M.Plan) -> float:
    inter = (a.hosts & b.hosts).sum()
    union = (a.hosts | b.hosts).sum()
    return float(inter / union) if union else 1.0


MODELS = ("naive_last", "seasonal_naive", "ewma", "site_ewma", "global_ewma")


@dataclass
class ForecastSpec:
    window_s: float = 900.0
    season: int = 96
    horizon: int = 4


def _ewma_series(X: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(X, dtype=float)
    out[0] = X[0]
    for t in range(1, len(X)):
        out[t] = alpha * X[t] + (1 - alpha) * out[t - 1]
    return out


class Forecaster:
    """Causal forecaster: predict(t, H) uses only Y[0..t]."""

    def __init__(self, model: str, spec: ForecastSpec, alpha: float = 0.3, beta: float = 0.3):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
        self.model, self.spec, self.alpha, self.beta = model, spec, alpha, beta
        self.Y = None

    @property
    def params(self) -> dict:
        if self.model == "ewma":
            return {"alpha": self.alpha}
        if self.model in ("site_ewma", "global_ewma"):
            return {"alpha": self.alpha, "beta": self.beta}
        return {}

    def prepare(self, Y: np.ndarray) -> "Forecaster":
        self.Y = np.asarray(Y, float)
        P = self.spec.season
        if self.model == "ewma":
            self.level = _ewma_series(self.Y, self.alpha)
        elif self.model in ("site_ewma", "global_ewma"):
            V = self.Y.sum(axis=2)                                   # (T, S)
            self.V = V
            ratio = np.ones_like(V)
            ratio[P:] = V[P:] / np.maximum(V[:-P], 1e-9)
            self.ratio = _ewma_series(ratio, self.beta)
            if self.model == "site_ewma":
                shares = self.Y / np.maximum(V[:, :, None], 1e-9)
            else:
                pooled = self.Y.sum(axis=1) / np.maximum(V.sum(axis=1)[:, None], 1e-9)   # (T, E)
                shares = np.repeat(pooled[:, None, :], self.Y.shape[1], axis=1)
            self.share = _ewma_series(shares, self.alpha)
        return self

    def predict(self, t: int, H: int) -> np.ndarray:
        """(H, S, E) forecasts of Y[t+1..t+H] made at the end of window t."""
        Y, P = self.Y, self.spec.season
        hs = np.arange(1, H + 1)
        if self.model == "naive_last":
            return np.repeat(Y[t][None], H, axis=0)
        if self.model == "seasonal_naive":
            idx = t + hs - P
            if (idx < 0).any():
                raise ValueError("seasonal_naive needs one full season of history")
            return Y[idx]
        if self.model == "ewma":
            return np.repeat(self.level[t][None], H, axis=0)
        idx = t + hs - P
        if (idx < 0).any():
            raise ValueError(f"{self.model} needs one full season of history")
        vol = self.V[idx] * self.ratio[t][None, :]                   # (H, S)
        return vol[:, :, None] * self.share[t][None, :, :]


# ---------------------------------------------------------------------------- metrics
def wape(Y, Yh) -> float:
    Y, Yh = np.asarray(Y, float), np.asarray(Yh, float)
    s = np.abs(Y).sum()
    return float(np.abs(Y - Yh).sum() / s) if s > 0 else 0.0


def mae(Y, Yh) -> float:
    return float(np.abs(np.asarray(Y, float) - np.asarray(Yh, float)).mean())


def popularity_error(Y, Yh) -> float:
    """Mean over sites of the total-variation distance between actual and forecast expert shares (0..1)."""
    Y, Yh = np.asarray(Y, float), np.asarray(Yh, float)
    p = Y / np.maximum(Y.sum(axis=-1, keepdims=True), 1e-12)
    q = Yh / np.maximum(Yh.sum(axis=-1, keepdims=True), 1e-12)
    return float(0.5 * np.abs(p - q).sum(axis=-1).mean())


def volume_error(Y, Yh) -> float:
    V, Vh = np.asarray(Y, float).sum(axis=-1), np.asarray(Yh, float).sum(axis=-1)
    return float((np.abs(V - Vh) / np.maximum(V, 1e-12)).mean())


# ---------------------------------------------------------------------------- fitting
GRID_ALPHA = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
GRID_BETA = (0.1, 0.3, 0.6)


def decision_points(start: int, end: int, H: int) -> list[int]:
    """Decision times t (end of window t) with a full horizon inside [start, end)."""
    return list(range(start, end - H, H))


def fit(model: str, Y: np.ndarray, spec: ForecastSpec, train_end: int) -> Forecaster:
    """Choose smoothing parameters by grid search on WAPE over training decision points only."""
    H, P = spec.horizon, spec.season
    pts = decision_points(P, train_end, H)
    if model in ("naive_last", "seasonal_naive"):
        return Forecaster(model, spec).prepare(Y)
    grid = [(a, 0.3) for a in GRID_ALPHA] if model == "ewma" else list(itertools.product(GRID_ALPHA, GRID_BETA))
    best, best_f = np.inf, None
    Ytrain = Y[:train_end]
    for a, b in grid:
        f = Forecaster(model, spec, a, b).prepare(Ytrain)
        err = np.mean([wape(Ytrain[t + 1:t + H + 1].mean(0), f.predict(t, H).mean(0)) for t in pts])
        if err < best:
            best, best_f = err, (a, b)
    return Forecaster(model, spec, *best_f).prepare(Y)


# ---------------------------------------------------------------------------- regret evaluation
def evaluate_point(t: int, Y: np.ndarray, forecaster: Forecaster, topo: M.Topology, H: int,
                   oracle: M.Plan | None = None, **plan_kw) -> tuple[dict, M.Plan, M.Plan]:
    fut = Y[t + 1:t + H + 1]
    pred = forecaster.predict(t, H)
    ybar, yhat = fut.mean(0), pred.mean(0)
    p_hat = M.plan_sites(yhat, topo, **plan_kw)
    p_star = oracle if oracle is not None else M.plan_sites(ybar, topo, **plan_kw)
    cap = plan_kw.get("util_cap")
    ev_hat = [M.evaluate(p_hat, fut[h], topo, cap) for h in range(H)]
    ev_star = [M.evaluate(p_star, fut[h], topo, cap) for h in range(H)]
    j_hat = np.array([e["objective"] for e in ev_hat])
    j_star = np.array([e["objective"] for e in ev_star])
    frag = p_star.info.get("fragile", np.zeros(Y.shape[2], bool))
    tot = ybar.sum(axis=0)
    dw = (np.abs(ybar - yhat).sum(axis=0)[frag].sum() / tot[frag].sum()) if tot[frag].sum() > 0 else 0.0
    row = {
        "t": t, "wape": wape(ybar, yhat), "mae": mae(ybar, yhat),
        "popularity_error": popularity_error(ybar, yhat), "volume_error": volume_error(ybar, yhat),
        "objective_forecast_plan": float(j_hat.mean()), "objective_oracle_plan": float(j_star.mean()),
        "J_forecast_plan": float(np.mean([e["J"] for e in ev_hat])),
        "latency_ms_forecast_plan": float(np.mean([e["added_latency_ms_per_token"] for e in ev_hat])),
        "regret_abs": float((j_hat - j_star).mean()),
        "regret_rel": float((j_hat - j_star).mean() / max(j_star.mean(), regret_floor(cap))),
        "agreement": M.agreement(p_hat, p_star, topo),
        "hosts_jaccard": hosts_jaccard(p_hat, p_star),
        "fragile_demand_share": float(p_star.info.get("fragile_demand_share", 0.0)),
        "decision_weighted_wape": float(dw),
        "planner_ms": float(p_hat.info["planner_ms"]),
    }
    return row, p_hat, p_star


def rolling_evaluation(Y: np.ndarray, topo: M.Topology, spec: ForecastSpec, train_end: int,
                       models=MODELS, eval_end: int | None = None, **plan_kw) -> pd.DataFrame:
    """Fit each model on [0, train_end), then evaluate forecast error and placement regret at every decision
    point in [train_end, eval_end)."""
    H = spec.horizon
    eval_end = eval_end or Y.shape[0]
    fitted = {m: fit(m, Y, spec, train_end) for m in models}
    rows = []
    for t in decision_points(train_end, eval_end, H):
        ybar = Y[t + 1:t + H + 1].mean(0)
        oracle = M.plan_sites(ybar, topo, **plan_kw)
        for m, f in fitted.items():
            row, _, _ = evaluate_point(t, Y, f, topo, H, oracle=oracle, **plan_kw)
            rows.append({"model": m, **{k: v for k, v in f.params.items()}, **row})
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    cols = ["wape", "popularity_error", "volume_error", "decision_weighted_wape", "regret_abs", "regret_rel",
            "agreement", "hosts_jaccard",
            "objective_forecast_plan", "objective_oracle_plan", "J_forecast_plan", "latency_ms_forecast_plan"]
    g = rows.groupby("model")[cols]
    out = g.mean().add_prefix("mean_")
    sums = rows.groupby("model")[["regret_abs", "objective_oracle_plan"]].sum()
    out["aggregate_regret_rel"] = sums.regret_abs / sums.objective_oracle_plan.where(sums.objective_oracle_plan > 0)
    out["p90_regret_rel"] = rows.groupby("model").regret_rel.quantile(0.9)
    out["max_regret_rel"] = rows.groupby("model").regret_rel.max()
    out["decision_points"] = rows.groupby("model").size()
    return out.reset_index().sort_values("mean_regret_abs")


def drift_table(rows: pd.DataFrame, tau_err: float = 0.10, tau_regret: float = 0.02) -> pd.DataFrame:
    """Classify decision points: does forecast error predict decision damage?"""
    r = rows.copy()
    r["high_error"] = r.wape > tau_err
    r["harmful"] = r.regret_rel > tau_regret
    out = []
    for m, g in r.groupby("model"):
        tp = int((g.high_error & g.harmful).sum())
        fp = int((g.high_error & ~g.harmful).sum())
        fn = int((~g.high_error & g.harmful).sum())
        tn = int((~g.high_error & ~g.harmful).sum())
        out.append({"model": m, "tau_err": tau_err, "tau_regret": tau_regret,
                    "harmful_and_high_error": tp, "harmless_high_error": fp,
                    "harmful_low_error": fn, "harmless_low_error": tn,
                    "error_trigger_precision": tp / (tp + fp) if tp + fp else float("nan"),
                    "error_trigger_recall": tp / (tp + fn) if tp + fn else float("nan"),
                    "corr_wape_regret": float(np.corrcoef(g.wape, g.regret_rel)[0, 1]) if len(g) > 2 and g.wape.std() > 0 and g.regret_rel.std() > 0 else float("nan")})
    return pd.DataFrame(out)
