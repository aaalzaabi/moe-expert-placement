"""Time-indexed, site-aware expert-demand tensors.

Forecast target (proposal feedback 11)
--------------------------------------
    Y[t, s, e] = number of token-to-expert assignments that originate at site s and are routed to the
                 layer-specific expert e (e = layer * 64 + expert) during window t of length W seconds.

Units: assignments per window (divide by W for activations per second). Summed over e, Y[t, s, :] equals
tokens(t, s) * 16 layers * 8 experts, so the tensor carries both the traffic volume of each site and the
expert-popularity distribution of each site. Popularity probabilities are Y / Y.sum(axis=-1).

Data provenance
---------------
* Per-category routing profiles P[c, e] (expected assignments per token) are MEASURED on OLMoE
  (outputs/workload_routing.csv, or a newer outputs/measurements/ run).
* Site traffic (request rates, daily cycles, category mixes, events) is SYNTHETIC and HYPOTHETICAL, defined per
  use case in usecases.py. The tensor is therefore "synthetic traffic over measured routing profiles".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


def category_profiles(routing: pd.DataFrame, smoothing: float = 0.5) -> tuple[list[str], np.ndarray]:
    """Expected assignments per token for each category: P[c, e], rows sum to TOP_K * NUM_LAYERS = 128.

    `smoothing` adds a pseudo-count of assignments to every expert before normalising, so that experts never
    seen in the small measured sample keep a small non-zero probability (documented modelling choice)."""
    cats = sorted(routing.category.unique())
    E = C.TOTAL_EXPERT_SLOTS
    out = np.zeros((len(cats), E))
    per_token = C.TOP_K * C.NUM_LAYERS
    for i, cat in enumerate(cats):
        sub = routing[routing.category == cat]
        counts = sub.groupby(["layer_id", "expert_id"]).token_assignments.sum()
        vec = np.zeros(E)
        idx = counts.index.get_level_values(0).to_numpy() * C.EXPERTS_PER_LAYER + counts.index.get_level_values(1).to_numpy()
        vec[idx] = counts.to_numpy(float)
        vec = vec.reshape(C.NUM_LAYERS, C.EXPERTS_PER_LAYER) + smoothing
        vec = vec / vec.sum(axis=1, keepdims=True) * C.TOP_K  # each layer: 8 assignments per token
        out[i] = vec.ravel()
    assert np.allclose(out.sum(axis=1), per_token)
    return cats, out


@dataclass(frozen=True)
class Shift:
    """A change in a site's traffic between two windows (campaign, outage elsewhere, product launch)."""
    start: int
    end: int
    mix: dict | None = None        # category shares during the shift (None = unchanged)
    rate_multiplier: float = 1.0


@dataclass(frozen=True)
class SiteTraffic:
    peak_requests_per_window: float
    mix: dict                      # category -> share
    utc_offset_h: float = 0.0
    trough_ratio: float = 0.25     # night traffic as a fraction of the daily peak
    peak_hour_local: float = 14.0
    tokens_per_request: float = 400.0  # HYPOTHETICAL (prefill + decode); measured prompts were 6-10 tokens
    shifts: tuple = ()
    volume_cv: float = 0.08        # window-to-window variability of request volume (gamma noise)
    mix_concentration: float = 300.0  # Dirichlet concentration of the per-window category mix (higher = steadier)


def diurnal(t: np.ndarray, window_s: float, utc_offset_h: float, trough: float, peak_hour: float) -> np.ndarray:
    hour = (t * window_s / 3600.0 + utc_offset_h) % 24.0
    return trough + (1 - trough) * 0.5 * (1 + np.cos(2 * np.pi * (hour - peak_hour) / 24.0))


def expected_mix(traffic: SiteTraffic, cats: list[str], T: int) -> tuple[np.ndarray, np.ndarray]:
    base = np.array([traffic.mix.get(c, 0.0) for c in cats], float)
    base = base / base.sum()
    mix = np.repeat(base[None, :], T, axis=0)
    mult = np.ones(T)
    for sh in traffic.shifts:
        sl = slice(max(sh.start, 0), min(sh.end, T))
        if sh.mix is not None:
            m = np.array([sh.mix.get(c, 0.0) for c in cats], float)
            mix[sl] = m / m.sum()
        mult[sl] *= sh.rate_multiplier
    return mix, mult


def synthesize(traffic: list[SiteTraffic], cats: list[str], P: np.ndarray, T: int, window_s: float = 900.0,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y, expected) with shape (T, S, E). Y is sampled; expected is the noise-free mean."""
    rng = np.random.default_rng(seed)
    S, E = len(traffic), P.shape[1]
    t = np.arange(T)
    Y = np.zeros((T, S, E))
    EX = np.zeros((T, S, E))
    for s, tr in enumerate(traffic):
        mix, mult = expected_mix(tr, cats, T)
        lam = tr.peak_requests_per_window * diurnal(t, window_s, tr.utc_offset_h, tr.trough_ratio, tr.peak_hour_local) * mult
        tokens = lam * tr.tokens_per_request
        EX[:, s, :] = tokens[:, None] * (mix @ P)
        k = 1.0 / tr.volume_cv ** 2
        vol = tokens * rng.gamma(k, 1.0 / k, size=T)
        noisy_mix = np.vstack([rng.dirichlet(np.maximum(m * tr.mix_concentration, 1e-3)) for m in mix])
        Y[:, s, :] = rng.poisson(vol[:, None] * (noisy_mix @ P))
    return Y, EX


def synthetic_demand(n_experts: int, n_sites: int, total_per_window: float = 3.0e8, seed: int = 0,
                     routing: pd.DataFrame | None = None, site_skew: float = 0.3) -> np.ndarray:
    """Synthetic (S, E) demand for scalability experiments with an arbitrary number of experts.

    Expert popularity is resampled from the measured per-expert demand distribution when `routing` is given
    (else a Zipf(1.1) law), with lognormal noise; each site perturbs popularity by `site_skew` (lognormal sigma)."""
    rng = np.random.default_rng(seed)
    if routing is not None:
        base = routing.groupby(["layer_id", "expert_id"]).token_assignments.sum().to_numpy(float) + 0.5
        pop = rng.choice(base, size=n_experts, replace=True) * rng.lognormal(0, 0.25, n_experts)
    else:
        pop = 1.0 / np.arange(1, n_experts + 1) ** 1.1
        rng.shuffle(pop)
    pop = pop / pop.sum()
    site_share = rng.dirichlet(np.full(n_sites, 5.0))
    Y = np.zeros((n_sites, n_experts))
    for s in range(n_sites):
        p = pop * rng.lognormal(0, site_skew, n_experts)
        Y[s] = total_per_window * site_share[s] * p / p.sum()
    return Y
