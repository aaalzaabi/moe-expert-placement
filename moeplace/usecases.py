"""Use cases (proposal feedback 6): six scenarios that exercise different strengths of the planner.

All sites, rates, mixes, device speeds and events are HYPOTHETICAL. Expert-routing profiles per category
(ai / coding / general) are MEASURED on OLMoE. Each use case names its primary users, the decision it
supports, its KPIs and the governance obligations it raises (detailed in docs/GOVERNANCE.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import multisite as M
from . import workload as WL
from .forecast import ForecastSpec

DAY = 96  # 15-minute windows per day

SITES = (
    M.Site("abudhabi", region="UAE", utc_offset_h=4),
    M.Site("frankfurt", region="EU", utc_offset_h=1),
    M.Site("singapore", region="APAC", utc_offset_h=8),
)
SPEEDS = (1.0, 1.3, 0.8, 1.6)  # mixed GPU generations inside each site (relative throughput)

TRAFFIC = (
    WL.SiteTraffic(peak_requests_per_window=3000, mix={"general": 0.5, "ai": 0.3, "coding": 0.2}, utc_offset_h=4),
    WL.SiteTraffic(peak_requests_per_window=2500, mix={"coding": 0.55, "ai": 0.25, "general": 0.2}, utc_offset_h=1),
    WL.SiteTraffic(peak_requests_per_window=2800, mix={"ai": 0.5, "coding": 0.3, "general": 0.2}, utc_offset_h=8),
)


@dataclass(frozen=True)
class UseCase:
    key: str
    title: str
    users: str
    question: str
    kpis: str
    governance: str
    topology: M.Topology
    traffic: tuple
    days: int = 4
    train_days: int = 2
    replica_frac: float = 2.0          # up to 2E extra copies, i.e. every expert may exist in all 3 sites
    util_cap: float | None = 0.7       # locality mode: minimise cross-site latency while utilisation <= 0.7
    constraints: M.Constraints = M.Constraints()
    events: tuple = ()                 # (window, "fail", device_name)
    consolidation: bool = False
    seed: int = 7

    @property
    def T(self) -> int:
        return self.days * DAY

    @property
    def train_end(self) -> int:
        return self.train_days * DAY

    def demand(self, routing: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cats, P = WL.category_profiles(routing)
        Y, _ = WL.synthesize(list(self.traffic), cats, P, self.T, self.topology.cost.window_s, seed=self.seed)
        return Y, cats

    def plan_kw(self) -> dict:
        return {"replica_frac": self.replica_frac, "constraints": self.constraints, "util_cap": self.util_cap}


def _topo(sites=SITES, per_site=4, speeds=SPEEDS, slots=256, memory_gb=None) -> M.Topology:
    return M.make_topology(list(sites), devices_per_site=per_site, speeds=speeds, slots=slots, memory_gb=memory_gb)


def _shifted(traffic, site, start, end, mix, mult):
    tr = list(traffic)
    tr[site] = replace(tr[site], shifts=tr[site].shifts + (WL.Shift(start, end, mix, mult),))
    return tuple(tr)


USE_CASES: dict[str, UseCase] = {}


def _add(uc: UseCase):
    USE_CASES[uc.key] = uc


_add(UseCase(
    key="uc1_multi_region",
    title="Multi-region enterprise assistant",
    users="Inference platform engineers; placement operations agent",
    question="Where should each expert live when three regions peak at different hours and ask different kinds of questions?",
    kpis="Bottleneck utilisation J, cross-site traffic share, forecast WAPE, placement regret",
    governance="Cross-border transfer of prompts/activations (UAE PDPL, EU GDPR Chapter V); service availability",
    topology=_topo(), traffic=TRAFFIC,
))
_add(UseCase(
    key="uc2_sovereign_residency",
    title="Sovereign / regulated tenant with data residency",
    users="Compliance officer; platform engineers; operations agent",
    question="What does it cost in capacity and latency to keep every UAE token inside the UAE site?",
    kpis="Residency violations (must be 0), J vs unconstrained plan, extra expert copies",
    governance="Data-residency obligations for government / health workloads; auditability of every placement change",
    topology=M.Topology(tuple(replace(s, residency=(s.name == "abudhabi")) for s in SITES), _topo().devices).validate(),
    traffic=TRAFFIC,
))
_add(UseCase(
    key="uc3_heterogeneous_cluster",
    title="Single cluster with mixed GPU generations (original project setting)",
    users="Inference platform engineers",
    question="How should experts be spread across fast and slow GPUs in one data centre?",
    kpis="Bottleneck utilisation J, imbalance; request-level p95 from the event simulator (Phase 3-4 results)",
    governance="Change management for production model-serving infrastructure",
    topology=M.make_topology([M.Site("cluster", region="single site", utc_offset_h=4)], devices_per_site=4, speeds=SPEEDS),
    traffic=(WL.SiteTraffic(peak_requests_per_window=3000, mix={"general": 0.4, "ai": 0.3, "coding": 0.3}, utc_offset_h=4),),
    replica_frac=0.0, util_cap=None,
))
_add(UseCase(
    key="uc4_workload_shift",
    title="Workload shift: product launch drives a coding surge in Frankfurt",
    users="Operations agent; SRE on call",
    question="Does the system notice that its forecast no longer supports good placements, and react?",
    kpis="Regret-triggered refits, regret before/after, WAPE vs regret relationship",
    governance="Model-risk monitoring and drift management (NIST AI RMF MEASURE/MANAGE)",
    topology=_topo(),
    traffic=_shifted(TRAFFIC, 1, 2 * DAY + 30, 3 * DAY + 60, {"coding": 0.85, "ai": 0.1, "general": 0.05}, 1.6),
))
_add(UseCase(
    key="uc5_device_failure",
    title="Device failure in Abu Dhabi",
    users="SRE on call; operations agent",
    question="How quickly can a feasible placement be restored after a GPU is lost, and at what transfer cost?",
    kpis="Time-to-feasible plan, reload bytes from storage, J after failure",
    governance="Incident management, business continuity, audit trail of automatic actions",
    topology=_topo(), traffic=TRAFFIC,
    events=((2 * DAY + 40, "fail", "abudhabi-g3"),),
))
_add(UseCase(
    key="uc6_offpeak_consolidation",
    title="Off-peak consolidation to save GPU-hours",
    users="FinOps / sustainability lead; operations agent",
    question="Which devices can be released at night without overloading the rest?",
    kpis="GPU-hours released, J kept below the 0.6 target, migrations caused",
    governance="Energy and cost reporting; human approval for capacity changes",
    topology=_topo(), traffic=TRAFFIC, consolidation=True,
))


def spec() -> ForecastSpec:
    return ForecastSpec(window_s=900.0, season=DAY, horizon=4)
