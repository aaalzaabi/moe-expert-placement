"""Multi-site, replica- and constraint-aware expert placement with an analytic cost model.

Why this module exists (proposal feedback 10-12): the original planner placed 1,024 experts on 4 GPUs of one
cluster. The proposal targeted heterogeneous, *geographically distributed* infrastructure, where demand differs
by site, cross-site links are slow, some sites may not send tokens abroad, and hot experts may be replicated.

Everything about hardware here is a HYPOTHETICAL parameter (see CostModel). The demand tensor that drives it is
derived from measured OLMoE routing profiles (see workload.py). The cost model is analytic (utilisation of the
bottleneck resource per planning window); the discrete-event simulator in simulator.py remains the tool for
request-level latency inside one cluster.

Notation (used in docs/FORECASTING_AND_REGRET.md and docs/SCALABILITY.md)
  E experts (layer-specific), G devices, S sites, sigma(g) = site of device g
  Y[s, e]  token-to-expert assignments originating at site s for expert e in one window of length W seconds
  serve(s, e) = device that serves site s's demand for e (local replica if one exists, else the primary)
  load_g   = sum_{s,e : serve(s,e)=g} Y[s, e]
  rho_c(g) = load_g / (speed_g * a_s * W)               compute utilisation (a_s = assignments/s at speed 1)
  B(s,s')  = b * sum_{e : s != sigma(serve(s,e)) ...}   bytes per direction on the link between sites s and s'
  rho_w(s,s') = B(s,s') / (bw_inter * 1e9 * W)          WAN utilisation
  J        = max( max_g rho_c(g), max_{s<s'} rho_w(s,s') )   bottleneck utilisation (lower is better, >1 = overload)
  r_s      = remotely served share of site s's assignments
  L        = sum_s V_s * 16 * RTT * (1 - (1 - r_s)^8) / sum_s V_s   added latency per token (ms), assuming the 8
             experts of a layer are remote independently with probability r_s
  Objective Phi: J in "balance" mode (util_cap=None); L + 10000 * max(0, J - util_cap) in "locality" mode
             (the penalty makes exceeding the utilisation cap dominate any latency saving)
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import config as C


class InfeasiblePlan(ValueError):
    """Raised when constraints (capacity, memory, residency) cannot be satisfied."""


@dataclass(frozen=True)
class Site:
    name: str
    region: str = ""
    utc_offset_h: float = 0.0
    residency: bool = False  # tokens originating here must be served by experts hosted here


@dataclass(frozen=True)
class Device:
    name: str
    site: int
    speed: float = 1.0           # relative throughput (HYPOTHETICAL)
    slots: int = 256             # expert slots
    memory_gb: float | None = None  # memory available for expert weights (GB); None = slots only


@dataclass(frozen=True)
class CostModel:
    """HYPOTHETICAL parameters of the analytic cost model. Units are in the names."""
    window_s: float = 900.0                  # planning window W (15 min)
    assignments_per_s: float = 5.0e4         # token-expert assignments/s on a speed-1.0 device
    activation_bytes: int = 4096             # bytes per assignment per direction (2048-dim BF16)
    inter_site_gb_per_s: float = 1.25        # per site pair and direction (10 Gbps)
    inter_site_rtt_ms: float = 60.0          # round trip between sites, paid per MoE layer with a remote expert
    intra_site_gb_per_s: float = 12.0        # used only for migration time
    storage_gb_per_s: float = 2.0            # reload from checkpoint storage after a device loss
    expert_bytes: int = C.CHECKPOINT_EXPERT_BYTES


@dataclass(frozen=True)
class Constraints:
    capacity: bool = True    # experts per device <= slots
    memory: bool = True      # experts per device <= floor(memory_gb / expert size)
    residency: bool = True   # residency sites host a copy of every expert and serve all their demand locally

    @property
    def label(self) -> str:
        on = [n for n in ("capacity", "memory", "residency") if getattr(self, n)]
        return "+".join(on) if on else "none"


@dataclass(frozen=True)
class Topology:
    sites: tuple[Site, ...]
    devices: tuple[Device, ...]
    cost: CostModel = field(default_factory=CostModel)

    def validate(self) -> "Topology":
        errs = []
        if not self.sites:
            errs.append("at least one site is required")
        if not self.devices:
            errs.append("at least one device is required")
        for d in self.devices:
            if not 0 <= d.site < len(self.sites):
                errs.append(f"device {d.name} refers to unknown site {d.site}")
            if d.speed <= 0 or d.slots <= 0:
                errs.append(f"device {d.name}: speed and slots must be positive")
        if len({d.name for d in self.devices}) != len(self.devices):
            errs.append("device names must be unique")
        c = self.cost
        if min(c.window_s, c.assignments_per_s, c.inter_site_gb_per_s, c.intra_site_gb_per_s, c.expert_bytes) <= 0:
            errs.append("cost-model parameters must be positive")
        if errs:
            raise C.ConfigError("Invalid Topology:\n  - " + "\n  - ".join(errs))
        return self

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def n_devices(self) -> int:
        return len(self.devices)

    @property
    def site_of(self) -> np.ndarray:
        return np.array([d.site for d in self.devices], dtype=int)

    @property
    def speeds(self) -> np.ndarray:
        return np.array([d.speed for d in self.devices], dtype=float)

    def capacity(self, constraints: Constraints) -> np.ndarray:
        cap = np.array([d.slots if constraints.capacity else 10**9 for d in self.devices], dtype=np.int64)
        if constraints.memory:
            for i, d in enumerate(self.devices):
                if d.memory_gb is not None:
                    cap[i] = min(cap[i], int(d.memory_gb * C.GB // self.cost.expert_bytes))
        return cap

    def without(self, names) -> "Topology":
        names = set(names)
        return replace(self, devices=tuple(d for d in self.devices if d.name not in names)).validate()

    def residency_sites(self) -> list[int]:
        return [i for i, s in enumerate(self.sites) if s.residency]


@dataclass
class Plan:
    hosts: np.ndarray          # bool (E, G): device g holds a copy of expert e
    primary: np.ndarray        # int (E,): primary device of each expert
    serve: np.ndarray          # int (S, E): device serving site s's demand for expert e
    info: dict = field(default_factory=dict)

    @property
    def n_experts(self) -> int:
        return self.hosts.shape[0]

    @property
    def copies(self) -> int:
        return int(self.hosts.sum())


# ---------------------------------------------------------------------------- evaluation
def evaluate(plan: Plan, Y: np.ndarray, topo: Topology, util_cap: float | None = None) -> dict:
    """Predicted inference metrics of `plan` under demand Y (S, E) for one window. Analytic, HYPOTHETICAL."""
    Y = np.asarray(Y, dtype=float)
    S, E = Y.shape
    c = topo.cost
    G = topo.n_devices
    site_of = topo.site_of
    load = np.zeros(G)
    np.add.at(load, plan.serve.ravel(), Y.ravel())
    rho_c = load / (topo.speeds * c.assignments_per_s * c.window_s)
    serve_site = site_of[plan.serve]                      # (S, E)
    remote = serve_site != np.arange(S)[:, None]
    B = np.zeros((S, S))
    if remote.any():
        src = np.repeat(np.arange(S)[:, None], E, axis=1)[remote]
        np.add.at(B, (src, serve_site[remote]), Y[remote] * c.activation_bytes)
    B = B + B.T  # requests one way + responses the other way: symmetric bytes per direction
    rho_w = B / (c.inter_site_gb_per_s * C.GB * c.window_s)
    J = float(max(rho_c.max() if G else 0.0, rho_w.max() if S > 1 else 0.0))
    total = Y.sum()
    viol = 0.0
    for r in topo.residency_sites():
        if r < S:
            viol += float(Y[r][serve_site[r] != r].sum())
    counts = plan.hosts.sum(axis=0)
    V = Y.sum(axis=1)
    r_s = np.where(V > 0, (Y * remote).sum(axis=1) / np.maximum(V, 1e-12), 0.0)
    p_layer = 1 - (1 - r_s) ** C.TOP_K
    lat_s = C.NUM_LAYERS * c.inter_site_rtt_ms * p_layer
    latency = float((V * lat_s).sum() / total) if total > 0 else 0.0
    return {
        "objective": objective_value(J, latency, util_cap),
        "J": J,
        "rho_compute_max": float(rho_c.max()),
        "rho_compute_mean": float(rho_c.mean()),
        "rho_wan_max": float(rho_w.max()) if S > 1 else 0.0,
        "imbalance": float(rho_c.max() / rho_c.mean()) if rho_c.mean() > 0 else 1.0,
        "remote_share": float(Y[remote].sum() / total) if total > 0 else 0.0,
        "added_latency_ms_per_token": latency,
        "wan_bytes": float(np.triu(B, 1).sum()),
        "queue_factor": float(1.0 / (1.0 - J)) if J < 1 else math.inf,
        "overloaded": bool(J >= 1.0),
        "residency_violation_assignments": viol,
        "copies": int(counts.sum()),
        "max_experts_per_device": int(counts.max()) if G else 0,
    }


PENALTY = 10_000.0  # ms-equivalent per unit of utilisation above the cap


def objective_value(J: float, latency_ms: float, util_cap: float | None) -> float:
    return J if util_cap is None else latency_ms + PENALTY * max(0.0, J - util_cap)


def J_of(plan: Plan, Y: np.ndarray, topo: Topology) -> float:
    return evaluate(plan, Y, topo)["J"]


def objective_of(plan: Plan, Y: np.ndarray, topo: Topology, util_cap: float | None = None) -> float:
    return evaluate(plan, Y, topo, util_cap)["objective"]


# ---------------------------------------------------------------------------- planner
def plan_sites(Yhat: np.ndarray, topo: Topology, replica_frac: float = 0.0, constraints: Constraints = Constraints(),
               fragile_gap: float = 0.01, util_cap: float | None = None,
               incumbent: np.ndarray | None = None, stickiness: float = 0.0) -> Plan:
    """Greedy min-bottleneck placement with optional replicas and constraints.

    1. Residency: every residency site r hosts each expert once (experts sorted by r's demand, each to the
       least-utilised device in r). Their demand is served locally.
    2. Primary: remaining experts sorted by remaining demand (descending, stable); each goes to the feasible device
       minimising max(compute utilisation after adding it, utilisation of the WAN links it would load).
       Ties: lower compute utilisation, then lower device id. With one site, no replicas and no residency this is
       exactly placement.assign(..., "demand_aware") (tested).
    3. Replicas: up to round(replica_frac * E) extra copies. Candidates are (site, expert) pairs ordered by the
       demand that site currently sends abroad for that expert; a copy goes to the least-utilised feasible device
       in that site. "balance" mode (util_cap=None) keeps it only if J does not increase; "locality" mode keeps it
       while J stays <= max(util_cap, J before), trading pooled capacity for lower cross-site latency.
    Hysteresis: with an `incumbent` host matrix (E, G) and stickiness k > 0, a device that already holds the
    expert scores (1 - k) times its real score, and replicas reuse an existing local copy when possible. This
    trades a little optimality for far fewer weight transfers between consecutive plans.
    Work: O(E log E + E (G + S^2) + R (G + S^2)).
    """
    t0 = time.perf_counter()
    topo.validate()
    Yhat = np.asarray(Yhat, dtype=float)
    S, E = Yhat.shape
    if S != topo.n_sites:
        raise ValueError(f"demand has {S} sites, topology has {topo.n_sites}")
    c = topo.cost
    G = topo.n_devices
    site_of = topo.site_of
    speeds = topo.speeds
    denom = speeds * c.assignments_per_s * c.window_s           # assignments that saturate each device
    wan_denom = c.inter_site_gb_per_s * C.GB * c.window_s
    a = float(c.activation_bytes)
    cap = topo.capacity(constraints)
    counts = np.zeros(G, dtype=np.int64)
    load = np.zeros(G)
    W = np.zeros((S, S))                                        # bytes per direction between sites (symmetric)
    hosts = np.zeros((E, G), dtype=bool)
    serve = np.full((S, E), -1, dtype=int)
    evaluated = 0

    res_sites = topo.residency_sites() if constraints.residency else []
    free_sites = [s for s in range(S) if s not in res_sites]

    # 1. residency
    for r in res_sites:
        devs = np.flatnonzero(site_of == r)
        if cap[devs].sum() < E:
            raise InfeasiblePlan(f"residency site {topo.sites[r].name} needs {E} expert slots, has {int(cap[devs].sum())}")
        for e in np.argsort(-Yhat[r], kind="stable"):
            ok = devs[counts[devs] < cap[devs]]
            u = (load[ok] + Yhat[r, e]) / denom[ok]
            g = int(ok[np.lexsort((ok, u))[0]])
            evaluated += len(devs)
            hosts[e, g] = True
            counts[g] += 1
            load[g] += Yhat[r, e]
            serve[r, e] = g

    # 2. primary placement of the demand from the other sites
    rem = Yhat[free_sites].sum(axis=0) if free_sites else np.zeros(E)
    order = np.argsort(-rem, kind="stable")
    primary = np.full(E, -1, dtype=int)
    gap = np.full(E, np.inf)
    fs = np.array(free_sites, dtype=int)
    for e in order:
        d_sites = Yhat[fs, e] if len(fs) else np.zeros(0)
        feasible = (counts < cap) | hosts[e]
        if not feasible.any():
            raise InfeasiblePlan(f"no device has a free slot for expert {e} (capacity/memory constraint)")
        cand = np.flatnonzero(feasible)
        evaluated += len(cand)
        d = float(d_sites.sum())
        u = (load[cand] + d) / denom[cand]
        if S > 1 and len(fs):
            # WAN utilisation of the links this expert would load if hosted in each candidate site
            wan_site = np.zeros(S)
            for k in range(S):
                add = 0.0
                for j, s in enumerate(fs):
                    if s != k and d_sites[j] > 0:
                        add = max(add, (W[s, k] + a * d_sites[j]) / wan_denom)
                wan_site[k] = add
            score = np.maximum(u, wan_site[site_of[cand]])
        else:
            score = u
        if incumbent is not None and stickiness > 0:
            score = score * (1.0 - stickiness * incumbent[e, cand])
        idx = np.lexsort((cand, u, score))
        g = int(cand[idx[0]])
        if len(idx) > 1:
            best, second = score[idx[0]], score[idx[1]]
            gap[e] = (second - best) / max(best, 1e-12)
        primary[e] = g
        if not hosts[e, g]:
            hosts[e, g] = True
            counts[g] += 1
        load[g] += d
        for j, s in enumerate(fs):
            serve[s, e] = g
            if s != site_of[g] and d_sites[j] > 0:
                W[s, site_of[g]] += a * d_sites[j]
                W[site_of[g], s] += a * d_sites[j]
    # experts with no free-site demand but residency hosts: primary = first residency host
    for e in np.flatnonzero(primary < 0):
        primary[e] = int(np.flatnonzero(hosts[e])[0])

    # 3. replicas
    budget = int(round(replica_frac * E))
    added = 0
    if budget > 0 and S > 1 and len(fs) > 1:
        def current_J():
            return max((load / denom).max(), (W / wan_denom).max())
        pairs = []
        for s in fs:
            m = site_of[serve[s]] != s
            for e in np.flatnonzero(m & (Yhat[s] > 0)):
                pairs.append((-float(Yhat[s, e]), int(s), int(e)))
        pairs.sort()
        for neg, s, e in pairs:
            if added >= budget:
                break
            d_se = -neg
            if site_of[serve[s, e]] == s:
                continue
            devs = np.flatnonzero((site_of == s) & (counts < cap) & ~hosts[e])
            evaluated += int((site_of == s).sum())
            if len(devs) == 0:
                continue
            inc = devs[incumbent[e, devs]] if incumbent is not None and stickiness > 0 else devs[:0]
            pool = inc if len(inc) else devs
            g = int(pool[np.lexsort((pool, (load[pool] + d_se) / denom[pool]))[0]])
            old_g = serve[s, e]
            before = current_J()
            load[g] += d_se
            load[old_g] -= d_se
            k = site_of[old_g]
            W[s, k] -= a * d_se
            W[k, s] -= a * d_se
            limit = before + 1e-12 if util_cap is None else max(util_cap, before) + 1e-12
            if current_J() <= limit:
                hosts[e, g] = True
                counts[g] += 1
                serve[s, e] = g
                added += 1
            else:  # revert
                load[g] -= d_se
                load[old_g] += d_se
                W[s, k] += a * d_se
                W[k, s] += a * d_se

    if (serve < 0).any():
        raise InfeasiblePlan("internal error: some demand has no serving device")
    fragile = np.isfinite(gap) & (gap < fragile_gap)
    info = {
        "planner_ms": (time.perf_counter() - t0) * 1e3,
        "candidates_evaluated": int(evaluated),
        "replicas_added": int(added),
        "replica_budget": int(budget),
        "fragile": fragile,
        "fragile_share": float(fragile.mean()) if E else 0.0,
        "fragile_demand_share": float(Yhat.sum(axis=0)[fragile].sum() / Yhat.sum()) if Yhat.sum() > 0 else 0.0,
        "constraints": constraints.label,
        "util_cap": util_cap,
    }
    return Plan(hosts=hosts, primary=primary, serve=serve, info=info)


def plan_from_primary(primary: np.ndarray, topo: Topology) -> Plan:
    """Plan with one copy per expert on the given devices (all sites served by the primary)."""
    primary = np.asarray(primary, dtype=int)
    E = len(primary)
    hosts = np.zeros((E, topo.n_devices), dtype=bool)
    hosts[np.arange(E), primary] = True
    serve = np.repeat(primary[None, :], topo.n_sites, axis=0)
    return Plan(hosts, primary, serve, {})


def exact_small(Y: np.ndarray, topo: Topology, constraints: Constraints = Constraints(capacity=True, memory=False,
                                                                                       residency=False)) -> Plan:
    """Brute-force optimum of J over single-copy placements (tiny instances only: G**E <= 2e5)."""
    S, E = Y.shape
    G = topo.n_devices
    if G ** E > 200_000:
        raise ValueError("instance too large for exhaustive search")
    cap = topo.capacity(constraints)
    best, best_p = math.inf, None
    for combo in itertools.product(range(G), repeat=E):
        p = np.array(combo)
        if (np.bincount(p, minlength=G) > cap).any():
            continue
        j = J_of(plan_from_primary(p, topo), Y, topo)
        if j < best - 1e-15:
            best, best_p = j, p
    if best_p is None:
        raise InfeasiblePlan("no feasible placement")
    return plan_from_primary(best_p, topo)


# ---------------------------------------------------------------------------- migration cost
def incumbent_matrix(plan: Plan, plan_topo: Topology, topo: Topology) -> np.ndarray:
    """Host matrix of `plan` re-indexed to the devices of `topo` (matched by name; lost devices drop out)."""
    out = np.zeros((plan.hosts.shape[0], topo.n_devices), dtype=bool)
    idx = {d.name: i for i, d in enumerate(plan_topo.devices)}
    for j, d in enumerate(topo.devices):
        if d.name in idx:
            out[:, j] = plan.hosts[:, idx[d.name]]
    return out


def migration_cost(old: Plan, new: Plan, old_topo: Topology, new_topo: Topology | None = None) -> dict:
    """Bytes and time to go from `old` to `new`. Devices are matched by name, so a topology that lost a device
    is handled: copies that existed only on a lost device are reloaded from checkpoint storage."""
    new_topo = new_topo or old_topo
    c = new_topo.cost
    old_names = [d.name for d in old_topo.devices]
    new_names = [d.name for d in new_topo.devices]
    old_site = {d.name: d.site for d in old_topo.devices}
    E = new.hosts.shape[0]
    old_hosts_by_name = {n: old.hosts[:, i] for i, n in enumerate(old_names)}
    pair_bytes: dict[tuple, float] = {}
    n_new = 0
    for j, name in enumerate(new_names):
        had = old_hosts_by_name.get(name, np.zeros(E, dtype=bool))
        need = np.flatnonzero(new.hosts[:, j] & ~had)
        dst_site = new_topo.devices[j].site
        for e in need:
            n_new += 1
            src_names = [n for n in new_names if n in old_hosts_by_name and old_hosts_by_name[n][e]]
            same = [n for n in src_names if old_site[n] == dst_site]
            if same:
                key = ("intra", dst_site, dst_site)
            elif src_names:
                key = ("inter", old_site[src_names[0]], dst_site)
            else:
                key = ("storage", dst_site, dst_site)
            pair_bytes[key] = pair_bytes.get(key, 0.0) + c.expert_bytes
    rate = {"intra": c.intra_site_gb_per_s, "inter": c.inter_site_gb_per_s, "storage": c.storage_gb_per_s}
    times = [b / (rate[k[0]] * C.GB) for k, b in pair_bytes.items()]
    removed = 0
    for j, name in enumerate(old_names):
        keep = new.hosts[:, new_names.index(name)] if name in new_names else np.zeros(E, dtype=bool)
        removed += int((old.hosts[:, j] & ~keep).sum())
    return {
        "new_copies": n_new,
        "removed_copies": removed,
        "bytes": float(sum(pair_bytes.values())),
        "wan_bytes": float(sum(b for k, b in pair_bytes.items() if k[0] == "inter")),
        "storage_bytes": float(sum(b for k, b in pair_bytes.items() if k[0] == "storage")),
        "est_time_s": float(max(times)) if times else 0.0,
    }


def agreement(a: Plan, b: Plan, topo_a: Topology, topo_b: Topology | None = None) -> float:
    """Share of experts whose primary device (by name) is the same in both plans."""
    topo_b = topo_b or topo_a
    na = np.array([topo_a.devices[i].name for i in a.primary])
    nb = np.array([topo_b.devices[i].name for i in b.primary])
    return float((na == nb).mean())


# ---------------------------------------------------------------------------- topology helpers
def make_topology(site_specs, devices_per_site=4, speeds=None, slots=256, memory_gb=None,
                  cost: CostModel | None = None) -> Topology:
    """site_specs: list of Site. speeds: list per device within a site (cycled)."""
    speeds = list(speeds or [1.0])
    devices = []
    for s, site in enumerate(site_specs):
        for k in range(devices_per_site):
            devices.append(Device(name=f"{site.name}-g{k}", site=s, speed=speeds[k % len(speeds)], slots=slots,
                                  memory_gb=memory_gb))
    return Topology(tuple(site_specs), tuple(devices), cost or CostModel()).validate()
