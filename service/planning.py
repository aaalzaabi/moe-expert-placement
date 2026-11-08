"""Domain logic of the API: scenario -> topology + demand -> plan -> authoriser verdict.

Thin adapter over the tested core; nothing here re-implements planning or authorisation:
    planner      moeplace.multisite.plan_sites   (greedy min-bottleneck, replicas, residency, hysteresis)
    metrics      moeplace.multisite.evaluate     (analytic model, HYPOTHETICAL hardware)
    migration    moeplace.multisite.migration_cost
    rules        moeplace.agent.Authoriser       (the same deterministic rules the operations agent uses)

Demand: site s sends requests_per_window * tokens_per_request tokens, each routed to 8 experts in each of the 16
layers following the MEASURED OLMoE profile of its category mix (moeplace.workload.category_profiles).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from moeplace import agent as A
from moeplace import config as C
from moeplace import data as D
from moeplace import multisite as M
from moeplace import workload as WL

from .schemas import Scenario
from .settings import Settings


class ScenarioError(ValueError):
    """The scenario is well-formed but cannot be planned (unknown category, incompatible with the active plan)."""


@lru_cache(maxsize=1)
def measured_profiles() -> tuple[tuple[str, ...], np.ndarray]:
    cats, P = WL.category_profiles(D.load_routing())
    return tuple(cats), P


def topology(sc: Scenario) -> M.Topology:
    cost = M.CostModel(inter_site_gb_per_s=sc.inter_site_gb_per_s, inter_site_rtt_ms=sc.inter_site_rtt_ms,
                       intra_site_gb_per_s=sc.intra_site_gb_per_s)
    sites = tuple(M.Site(s.name, region=s.region, residency=s.residency) for s in sc.sites)
    devices = tuple(M.Device(d.name, i, d.speed, d.slots, d.memory_gb)
                    for i, s in enumerate(sc.sites) for d in s.devices)
    return M.Topology(sites, devices, cost).validate()


def demand(sc: Scenario) -> np.ndarray:
    """Expected assignments per window, shape (sites, 1024)."""
    cats, P = measured_profiles()
    Y = np.zeros((len(sc.sites), P.shape[1]))
    for s, site in enumerate(sc.sites):
        unknown = sorted(set(site.mix) - set(cats))
        if unknown:
            raise ScenarioError(f"site {site.name}: unknown categories {unknown}; measured categories are {list(cats)}")
        mix = np.array([site.mix.get(c, 0.0) for c in cats])
        Y[s] = site.requests_per_window * sc.tokens_per_request * (mix / mix.sum()) @ P
    return Y


def _finite(d: dict) -> dict:
    """JSON has no inf/nan: report them as null."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            v = float(v)
            out[k] = v if math.isfinite(v) else None
        elif isinstance(v, (np.integer, np.bool_)):
            out[k] = v.item()
        elif isinstance(v, (int, bool, str)) or v is None:
            out[k] = v
    return out


@dataclass
class Computed:
    scenario: Scenario
    topo: M.Topology
    Y: np.ndarray
    plan: M.Plan
    metrics: dict
    devices: list[dict]


def compute(sc: Scenario, incumbent: np.ndarray | None = None, stickiness: float = 0.0) -> Computed:
    """Run the planner. Raises M.InfeasiblePlan or C.ConfigError (both mean: this scenario cannot be served)."""
    topo = topology(sc)
    Y = demand(sc)
    plan = M.plan_sites(Y, topo, replica_frac=sc.replica_frac, util_cap=sc.util_cap,
                        incumbent=incumbent, stickiness=stickiness if incumbent is not None else 0.0)
    metrics = _finite(M.evaluate(plan, Y, topo, sc.util_cap))
    metrics.update(_finite({k: plan.info[k] for k in ("planner_ms", "candidates_evaluated", "replicas_added")}))
    load = np.zeros(topo.n_devices)
    np.add.at(load, plan.serve.ravel(), Y.ravel())
    util = load / (topo.speeds * topo.cost.assignments_per_s * topo.cost.window_s)
    counts = plan.hosts.sum(axis=0)
    devices = [{"name": d.name, "site": topo.sites[d.site].name, "experts": int(counts[i]),
                "utilisation": float(util[i])} for i, d in enumerate(topo.devices)]
    return Computed(sc, topo, Y, plan, metrics, devices)


def to_assignment(plan: M.Plan) -> dict:
    return {"hosts": [np.flatnonzero(row).tolist() for row in plan.hosts],
            "primary": plan.primary.tolist(), "serve": plan.serve.tolist()}


def from_assignment(a: dict, topo: M.Topology) -> M.Plan:
    E = len(a["primary"])
    hosts = np.zeros((E, topo.n_devices), dtype=bool)
    for e, gs in enumerate(a["hosts"]):
        hosts[e, gs] = True
    return M.Plan(hosts, np.asarray(a["primary"], dtype=int), np.asarray(a["serve"], dtype=int), {})


def _empty_plan(topo: M.Topology, E: int) -> M.Plan:
    """'Nothing loaded yet': every copy of the first plan is read from checkpoint storage."""
    return M.Plan(np.zeros((E, topo.n_devices), dtype=bool), np.full(E, -1), np.full((topo.n_sites, E), -1), {})


@dataclass
class Active:
    plan: M.Plan
    topo: M.Topology


@dataclass
class Assessment:
    reason: str                    # initial | rebalance | restore
    objective_current: float | None
    objective_new: float
    util_new: float
    migration: dict
    checks: list[dict]
    approved: bool                 # authoriser verdict
    needs_human: bool


def check_compatible(sc: Scenario, active: Active | None) -> None:
    if active is None:
        return
    have = [s.name for s in active.topo.sites]
    want = [s.name for s in sc.sites]
    if have != want:
        raise ScenarioError(f"scenario sites {want} differ from the active plan's sites {have}; "
                            "adding or removing a site is a new deployment, not a re-placement")


def incumbent(new_topo: M.Topology, active: Active | None) -> np.ndarray | None:
    return None if active is None else M.incumbent_matrix(active.plan, active.topo, new_topo)


def reindex(plan: M.Plan, old_topo: M.Topology, new_topo: M.Topology) -> M.Plan:
    """The same placement expressed on new_topo's device indices (matched by name; every old device must exist).
    Lets the active plan be scored on today's hardware (e.g. a device that became slower)."""
    idx = {d.name: j for j, d in enumerate(new_topo.devices)}
    to_new = np.array([idx[d.name] for d in old_topo.devices])
    return M.Plan(M.incumbent_matrix(plan, old_topo, new_topo), to_new[plan.primary], to_new[plan.serve], {})


def assess(new: Computed, active: Active | None, settings: Settings) -> Assessment:
    """Compare the new plan with the active one and apply the authoriser's rules."""
    E = new.plan.n_experts
    names = {d.name for d in new.topo.devices}
    lost = active is not None and any(d.name not in names for d in active.topo.devices)
    mandatory = active is None or lost
    reason = "initial" if active is None else ("restore" if lost else "rebalance")
    old_plan, old_topo = (active.plan, active.topo) if active else (_empty_plan(new.topo, E), new.topo)
    j_cur = None if mandatory else float(M.evaluate(reindex(old_plan, old_topo, new.topo), new.Y, new.topo,
                                                    new.scenario.util_cap)["objective"])
    migration = M.migration_cost(old_plan, new.plan, old_topo, new.topo)
    j_new = float(new.metrics["objective"])
    util_new = float(new.metrics["J"])
    proposal = A.Proposal(t=0, reason=reason, plan=new.plan, topo=new.topo,
                          J_current_forecast=math.inf if j_cur is None else j_cur, J_new_forecast=j_new,
                          migration=migration, mandatory=mandatory, util_new_forecast=util_new)
    authoriser = A.Authoriser(settings.agent_config(), settings.horizon_windows * new.topo.cost.window_s)
    touched = A.residency_touched(new.topo, old_plan, old_topo, new.plan, new.topo)
    ok, needs_human, reasons = authoriser.check(proposal, touched, new.metrics["residency_violation_assignments"])
    migration = _finite(migration) | {"gb": migration["bytes"] / C.GB, "residency_site_touched": touched}
    return Assessment(reason, j_cur, j_new, util_new, migration,
                      [{"check": s, "passed": bool(r)} for s, r in reasons], bool(ok), bool(needs_human))
