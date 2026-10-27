"""Placement operations agent (proposal feedback 5: "can this not be agentic?").

The agent runs a closed observe -> forecast -> propose -> check -> (approve) -> act -> learn loop on behalf of the
platform team. It is agentic in the operational sense: it pursues a goal (keep bottleneck utilisation low at
minimum transfer cost), chooses among tools, keeps state across decisions, detects when its own forecasts stop
supporting good decisions, refits them, and escalates to a human when an action is large or regulated.

What it deliberately is NOT: an LLM that is allowed to change serving. Planning and authorisation are separated:
* the agent PROPOSES (tools with "read"/"propose" permission);
* a deterministic Authoriser DECIDES with explicit, testable rules;
* only the authoriser can issue the token required by the single "act" tool (apply_placement);
* actions above the approval threshold, or touching a residency site, require a (simulated) human approval.
An optional `explainer` callable can be swapped for an LLM to write explanations; it never sees the act tool.

Everything runs against the analytic multi-site model (multisite.py) on synthetic traffic over measured routing
profiles. It is a simulation of agent behaviour, not a production deployment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import forecast as F
from . import multisite as M

PERMISSIONS = ("read", "propose", "escalate", "act")


@dataclass(frozen=True)
class AgentConfig:
    model: str = "site_ewma"
    min_gain_rel: float = 0.02          # re-place only if forecast J improves by >= 2 % (same as Phase 4E controller)
    max_migration_fraction: float = 0.05  # transfer time must fit in 5 % of the horizon unless mandatory
    approval_gb: float = 50.0           # larger migrations need human approval
    tau_regret: float = 0.02            # realised regret above this triggers a forecaster refit
    refit_days: int = 2
    consolidation: bool = False
    target_util: float = 0.6            # consolidation keeps forecast J <= this
    max_off_per_site: int = 2
    max_steps: int = 10_000
    stickiness: float = 0.05            # hysteresis: prefer devices that already hold an expert


@dataclass
class Tool:
    name: str
    permission: str
    fn: Callable
    description: str


class PermissionDenied(RuntimeError):
    pass


@dataclass
class Proposal:
    t: int
    reason: str
    plan: M.Plan
    topo: M.Topology
    J_current_forecast: float          # planner objective (Phi) of the current plan under the forecast
    J_new_forecast: float              # Phi of the proposed plan under the forecast
    migration: dict
    mandatory: bool
    devices_off: tuple = ()
    util_new_forecast: float = 0.0     # bottleneck utilisation J of the proposed plan


class Authoriser:
    """Deterministic rules. Returns (approve, needs_human, reasons)."""

    def __init__(self, cfg: AgentConfig, horizon_s: float):
        self.cfg, self.horizon_s = cfg, horizon_s

    def check(self, p: Proposal, residency_sites_touched: bool, residency_violation: float) -> tuple[bool, bool, list]:
        c = self.cfg
        gain = (p.J_current_forecast - p.J_new_forecast) / max(p.J_current_forecast, 1e-12)
        limit = c.max_migration_fraction * self.horizon_s
        reasons = [
            ("constraints satisfied (capacity, memory, residency violations = 0)", residency_violation == 0),
            (f"forecast utilisation of new plan {p.util_new_forecast:.3f} < 1 (no overload)", p.util_new_forecast < 1),
        ]
        if p.mandatory:
            reasons.append(("mandatory change (device lost / current plan invalid)", True))
        elif p.devices_off:
            reasons.append((f"consolidation keeps forecast utilisation {p.util_new_forecast:.3f} <= target {c.target_util}",
                            p.util_new_forecast <= c.target_util))
            reasons.append((f"transfer {p.migration['est_time_s']:.0f} s <= {limit:.0f} s", p.migration["est_time_s"] <= limit))
        else:
            reasons.append((f"forecast objective gain {100 * gain:+.1f}% >= {100 * c.min_gain_rel:.0f}%", gain >= c.min_gain_rel))
            reasons.append((f"transfer {p.migration['est_time_s']:.0f} s <= {limit:.0f} s", p.migration["est_time_s"] <= limit))
        ok = all(r for _, r in reasons)
        needs_human = ok and (p.migration["bytes"] / 1e9 > c.approval_gb or residency_sites_touched or bool(p.devices_off))
        return ok, needs_human, reasons


def auto_approver(proposal: Proposal) -> tuple[bool, str]:
    """Stand-in for a human approver in simulation: approves and says so in the audit log."""
    return True, "simulated human approval (auto)"


def template_explainer(entry: dict) -> str:
    verdict = entry["action"]
    return (f"t={entry['t']}: {verdict}. Forecast objective {entry['J_current_forecast']:.3f} -> {entry['J_new_forecast']:.3f}; "
            f"moves {entry['new_copies']} copies ({entry['migration_gb']:.1f} GB, ~{entry['migration_s']:.0f} s). "
            f"{entry['reasons']}")


class PlacementAgent:
    def __init__(self, Y: np.ndarray, topo: M.Topology, spec: F.ForecastSpec, cfg: AgentConfig = AgentConfig(),
                 plan_kw: dict | None = None, approver=auto_approver, explainer=template_explainer):
        self.Y, self.base_topo, self.spec, self.cfg = Y, topo, spec, cfg
        self.plan_kw = plan_kw or {}
        self.cap = self.plan_kw.get("util_cap")
        self.approver, self.explainer = approver, explainer
        self.authoriser = Authoriser(cfg, spec.horizon * topo.cost.window_s)
        self.failed: set[str] = set()
        self.trace: list[dict] = []
        self.audit: list[dict] = []
        self.proposals: list[dict] = []   # every proposal with the authoriser verdict (for inspection / UI)
        self._token = object()
        self.tools = {t.name: t for t in [
            Tool("observe_demand", "read", self._observe, "Demand tensor up to the current window"),
            Tool("forecast_demand", "read", self._forecast, "Site-aware forecast of the next H windows"),
            Tool("propose_placement", "propose", self._propose, "Run the multi-site planner on a forecast"),
            Tool("evaluate_plan", "read", self._evaluate, "Predicted utilisation of a plan under a demand"),
            Tool("estimate_migration", "read", self._migration, "Bytes and time to move between two plans"),
            Tool("request_approval", "escalate", self._approval, "Ask a human to approve a large/regulated change"),
            Tool("apply_placement", "act", self._apply, "Make a plan active (needs authoriser token)"),
        ]}

    # ------------------------------------------------------------------ tool plumbing
    def call(self, name: str, *args, token=None, **kw):
        tool = self.tools[name]
        if tool.permission == "act" and token is not self._token:
            raise PermissionDenied(f"{name} requires the authoriser's token")
        t0 = time.perf_counter()
        out = tool.fn(*args, **kw)
        self.trace.append({"t": self._t, "tool": name, "permission": tool.permission,
                           "ms": round((time.perf_counter() - t0) * 1e3, 3)})
        return out

    def _observe(self, t):
        return self.Y[: t + 1]

    def _forecast(self, t):
        return self.forecaster.predict(t, self.spec.horizon).mean(0)

    def _propose(self, yhat, topo):
        inc = None
        if getattr(self, "plan", None) is not None and self.cfg.stickiness > 0:
            inc = M.incumbent_matrix(self.plan, self.topo, topo)
        return M.plan_sites(yhat, topo, incumbent=inc, stickiness=self.cfg.stickiness if inc is not None else 0.0,
                            **self.plan_kw)

    def _evaluate(self, plan, y, topo):
        return M.evaluate(plan, y, topo, self.cap)

    def _migration(self, old, new, old_topo, new_topo):
        return M.migration_cost(old, new, old_topo, new_topo)

    def _approval(self, proposal):
        return self.approver(proposal)

    def _apply(self, plan, topo, devices_off):
        self.plan, self.topo, self.devices_off = plan, topo, tuple(devices_off)

    # ------------------------------------------------------------------ helpers
    def _active_topo(self, off=()) -> M.Topology:
        drop = set(self.failed) | set(off)
        return self.base_topo.without(drop) if drop else self.base_topo

    def _plan_valid(self) -> bool:
        return not any(d.name in self.failed for d in self.topo.devices)

    def _residency_touched(self, plan_n: M.Plan, topo_n: M.Topology) -> bool:
        """True if the proposal changes which experts are hosted on any device of a residency site."""
        res = set(self.base_topo.residency_sites())
        if not res:
            return False
        cur = {d.name: self.plan.hosts[:, i] for i, d in enumerate(self.topo.devices) if d.site in res}
        for j, d in enumerate(topo_n.devices):
            if d.site in res and (d.name not in cur or (cur[d.name] != plan_n.hosts[:, j]).any()):
                return True
        return False

    def _consolidation_options(self, yhat):
        """Smallest set of switched-off devices (slowest first, per site) whose forecast J stays <= target."""
        best = None
        sites = range(self.base_topo.n_sites)
        avail = [d for d in self.base_topo.devices if d.name not in self.failed]
        for k in range(1, self.cfg.max_off_per_site + 1):
            off = []
            for s in sites:
                devs = sorted([d for d in avail if d.site == s], key=lambda d: (d.speed, d.name))
                off += [d.name for d in devs[:k]]
            topo = self._active_topo(off)
            try:
                plan = self.call("propose_placement", yhat, topo)
            except M.InfeasiblePlan:
                break
            j = self.call("evaluate_plan", plan, yhat, topo)["J"]  # consolidation is about utilisation
            if j <= self.cfg.target_util:
                best = (tuple(off), topo, plan, j)
            else:
                break
        return best

    # ------------------------------------------------------------------ main loop
    def run(self, start: int, end: int | None = None, events=()) -> tuple[pd.DataFrame, pd.DataFrame]:
        H = self.spec.horizon
        end = end or self.Y.shape[0]
        self._t = start
        self.forecaster = F.fit(self.cfg.model, self.Y, self.spec, start)
        y0 = self.call("forecast_demand", start)
        self.plan = None
        self.plan, self.topo, self.devices_off = self.call("propose_placement", y0, self.base_topo), self.base_topo, ()
        static_plan, static_topo = self.plan, self.topo     # baseline: never re-planned (except forced on failure)
        events = sorted(events)
        prev = None
        rows = []
        for step, t in enumerate(F.decision_points(start, end, H)):
            if step >= self.cfg.max_steps:
                break
            self._t = t
            # events up to now
            for (te, kind, name) in events:
                if te <= t and kind == "fail" and name not in self.failed:
                    self.failed.add(name)
                    self.audit.append({"t": t, "event": f"device {name} failed"})
                    if any(d.name == name for d in static_topo.devices):
                        static_topo = self.base_topo.without(self.failed)
                        static_plan = M.plan_sites(self.forecaster.predict(start, H).mean(0), static_topo, **self.plan_kw)
            # learn: realised regret of the previous decision decides whether to refit the forecaster
            refit = False
            if prev is not None:
                past = self.Y[prev["t"] + 1: prev["t"] + H + 1]
                oracle = M.plan_sites(past.mean(0), prev["topo"], **self.plan_kw)
                jp = np.mean([M.objective_of(prev["plan"], y, prev["topo"], self.cap) for y in past])
                jo = np.mean([M.objective_of(oracle, y, prev["topo"], self.cap) for y in past])
                realised_regret = (jp - jo) / max(jo, F.regret_floor(self.cap))
                if realised_regret > self.cfg.tau_regret:
                    self.forecaster = F.fit(self.cfg.model, self.Y[: t + 1], self.spec, t + 1)
                    self.forecaster.prepare(self.Y)
                    refit = True
            else:
                realised_regret = float("nan")

            yhat = self.call("forecast_demand", t)
            mandatory = not self._plan_valid()
            candidates = []
            topo_full = self._active_topo()
            try:
                plan_full = self.call("propose_placement", yhat, topo_full)
                candidates.append(((), topo_full, plan_full))
            except M.InfeasiblePlan as exc:
                self.audit.append({"t": t, "event": f"planner infeasible: {exc}"})
            if self.cfg.consolidation:
                opt = self._consolidation_options(yhat)
                if opt:
                    off, topo_c, plan_c, _ = opt
                    candidates.insert(0, (off, topo_c, plan_c))
            j_cur = self.call("evaluate_plan", self.plan, yhat, self.topo)["objective"] if not mandatory else float("inf")
            chosen, entry = None, None
            for off, topo_n, plan_n in candidates:
                ev = self.call("evaluate_plan", plan_n, yhat, topo_n)
                mig = self.call("estimate_migration", self.plan, plan_n, self.topo, topo_n)
                prop = Proposal(t, "consolidate" if off else ("restore" if mandatory else "rebalance"), plan_n, topo_n,
                                j_cur, ev["objective"], mig, mandatory, off, ev["J"])
                res_touched = self._residency_touched(plan_n, topo_n)
                ok, needs_human, reasons = self.authoriser.check(prop, res_touched, ev["residency_violation_assignments"])
                approved_by = "authoriser"
                human = "not required"
                authoriser_ok = ok
                if ok and needs_human:
                    ok, approved_by = self.call("request_approval", prop)
                    human = "approved" if ok else ("pending" if "pending" in str(approved_by) else "denied")
                    self.audit.append({"t": t, "event": f"human approval {human} for {prop.reason} ({approved_by})"})
                self.proposals.append({"t": t, "proposal": prop, "authoriser_ok": authoriser_ok,
                                       "needs_human": needs_human, "human_decision": human, "reasons": reasons})
                entry = {"t": t, "proposal": prop.reason, "devices_off": len(off),
                         "J_current_forecast": j_cur, "J_new_forecast": ev["objective"], "util_new_forecast": ev["J"],
                         "new_copies": mig["new_copies"], "migration_gb": mig["bytes"] / 1e9,
                         "migration_s": mig["est_time_s"], "needs_human": needs_human, "approved_by": approved_by,
                         "human_decision": human,
                         "reasons": " | ".join(("PASS: " if r else "FAIL: ") + s for s, r in reasons)}
                if ok:
                    chosen = (plan_n, topo_n, off)
                    break
                # if consolidation was rejected, fall through to the full-topology candidate
            if chosen is None and (mandatory or self.devices_off) and candidates:
                # current plan is invalid, or devices are off but consolidation no longer holds: restore full capacity.
                # The restore is still checked: constraints (residency, overload) can never be bypassed.
                off, topo_n, plan_n = candidates[-1]
                ev = self.call("evaluate_plan", plan_n, yhat, topo_n)
                mig = self.call("estimate_migration", self.plan, plan_n, self.topo, topo_n)
                prop = Proposal(t, "restore", plan_n, topo_n, j_cur, ev["objective"], mig, True, off, ev["J"])
                ok, _, reasons = self.authoriser.check(prop, False, ev["residency_violation_assignments"])
                self.proposals.append({"t": t, "proposal": prop, "authoriser_ok": ok, "needs_human": False,
                                       "human_decision": "not required (mandatory restore)", "reasons": reasons})
                if ok:
                    chosen = (plan_n, topo_n, off)
                    entry = {**(entry or {}), "proposal": "restore", "approved_by": "authoriser (mandatory)",
                             "reasons": " | ".join(("PASS: " if r else "FAIL: ") + s for s, r in reasons)}
                else:
                    self.audit.append({"t": t, "event": "mandatory restore BLOCKED by authoriser: " +
                                       "; ".join(s for s, r in reasons if not r)})
            if chosen is not None:
                self.call("apply_placement", chosen[0], chosen[1], chosen[2], token=self._token)
                action = entry["proposal"]
            else:
                action = "keep"
            entry = {**(entry or {"t": t, "J_current_forecast": j_cur, "J_new_forecast": j_cur, "new_copies": 0,
                                   "migration_gb": 0.0, "migration_s": 0.0, "reasons": "no candidate",
                                   "human_decision": "not required"}), "action": action}
            entry["explanation"] = self.explainer(entry)
            self.audit.append({"t": t, "event": entry["explanation"]})

            # outcome on the actual next horizon (evaluation only; the agent never sees this before acting)
            fut = self.Y[t + 1: t + H + 1]
            ev_a = [M.evaluate(self.plan, y, self.topo, self.cap) for y in fut]
            ev_s = [M.evaluate(static_plan, y, static_topo, self.cap) for y in fut]
            oracle_topo = self._active_topo(self.devices_off)
            oracle = M.plan_sites(fut.mean(0), oracle_topo, **self.plan_kw)
            ev_o = [M.evaluate(oracle, y, oracle_topo, self.cap) for y in fut]
            mean = lambda evs, k: float(np.mean([e[k] for e in evs]))
            j_agent, j_static, j_oracle = mean(ev_a, "objective"), mean(ev_s, "objective"), mean(ev_o, "objective")
            rows.append({**{k: v for k, v in entry.items() if k != "explanation"},
                         "refit": refit, "realised_regret_prev": realised_regret,
                         "devices_active": self.topo.n_devices, "devices_off_now": len(self.devices_off),
                         "objective_actual_agent": j_agent, "objective_actual_static": j_static,
                         "objective_actual_oracle": j_oracle,
                         "util_actual_agent": mean(ev_a, "J"), "util_actual_static": mean(ev_s, "J"),
                         "latency_ms_actual_agent": mean(ev_a, "added_latency_ms_per_token"),
                         "latency_ms_actual_static": mean(ev_s, "added_latency_ms_per_token"),
                         "residency_violations_actual": mean(ev_a, "residency_violation_assignments"),
                         "gpu_hours_released": len(self.devices_off) * H * self.base_topo.cost.window_s / 3600})
            prev = {"t": t, "plan": self.plan, "topo": self.topo}
        return pd.DataFrame(rows), pd.DataFrame(self.trace)
