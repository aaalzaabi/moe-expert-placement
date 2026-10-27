"""Cost-aware migration controller (port of scripts/evaluate_migration_policy.py)
and an explicit migration state machine.

Leakage rule: the decision is computed from the TRAINING request ids only.
Held-out (test) requests are simulated afterwards purely for evaluation and are
never passed to `decide()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config as C
from . import data as D
from . import placement as P
from . import simulator as S

# ---------------------------------------------------------------- state machine
STATES = ("proposed", "approved", "rejected", "transferring", "active", "failed")
TRANSITIONS: dict[str, set[str]] = {
    "proposed": {"approved", "rejected"},
    "approved": {"transferring", "failed"},
    "transferring": {"active", "failed"},
    "rejected": {"proposed"},  # a new proposal may be made later
    "failed": {"proposed"},    # after rollback, retry is a new proposal
    "active": {"proposed"},    # next re-placement round
}


class InvalidTransition(RuntimeError):
    pass


@dataclass
class MigrationStateMachine:
    state: str = "proposed"
    history: list[tuple[str, str]] = field(default_factory=list)

    def to(self, new: str, note: str = "") -> "MigrationStateMachine":
        if new not in STATES:
            raise InvalidTransition(f"Unknown state {new!r}")
        if new not in TRANSITIONS[self.state]:
            raise InvalidTransition(f"Illegal transition {self.state} -> {new}")
        self.history.append((self.state, new) if not note else (self.state, f"{new}: {note}"))
        self.state = new
        return self

    @property
    def terminal(self) -> bool:
        return self.state in {"rejected", "active", "failed"}


# ---------------------------------------------------------------- controller
@dataclass
class Decision:
    approve: bool
    forecast_makespan_gain_pct: float
    forecast_p95_regression_pct: float
    migration_feasible: bool
    forecast_ids: tuple[int, ...]
    reasons: list[tuple[str, bool]]
    forecast_static: dict
    forecast_migrate: dict

    @property
    def decision(self) -> str:
        return "approve" if self.approve else "reject"


def decide(train_traces: S.Traces, layouts: dict[str, S.Layout], cfg: C.SimConfig) -> Decision:
    """Forecast never-migrate vs migrate on training requests and apply thresholds."""
    ids = tuple(cfg.train_ids)
    static = S.run(train_traces, layouts, cfg, ids, allow_migration=False).summary
    migrate = S.run(train_traces, layouts, cfg, ids, allow_migration=True).summary
    gain = 100 * (static["makespan_ms"] - migrate["makespan_ms"]) / static["makespan_ms"]
    regression = 100 * (migrate["p95_latency_ms"] - static["p95_latency_ms"]) / static["p95_latency_ms"]
    feasible = migrate["migration_state"] == "active"
    reasons = [
        (f"Migration feasible in forecast (staging memory / copies): {migrate['migration_state']}", feasible),
        (f"Forecast makespan gain {gain:+.2f}% >= required {cfg.min_makespan_gain_pct:.2f}%",
         gain >= cfg.min_makespan_gain_pct),
        (f"Forecast p95 change {regression:+.2f}% <= allowed {cfg.max_p95_regression_pct:+.2f}%",
         regression <= cfg.max_p95_regression_pct),
    ]
    approve = all(ok for _, ok in reasons)
    return Decision(approve, gain, regression, feasible, ids, reasons, static, migrate)


def build_layouts(routing: pd.DataFrame, cfg: C.SimConfig) -> tuple[pd.DataFrame, dict[str, S.Layout]]:
    """Placements from TRAINING demand only (as evaluate_placement.py did)."""
    placements = P.plan(D.demand(routing, cfg.train_ids), cfg, eval_demand=D.demand(routing, cfg.test_ids))
    return placements, {s: P.layout(placements, s) for s in C.STRATEGIES}


def evaluate(routing: pd.DataFrame, cfg: C.SimConfig, layouts: dict[str, S.Layout] | None = None):
    """Decide on train ids, then evaluate never / always / cost-aware on test ids.

    Returns (policies_df, requests_df, decision, always_migrate_events, test_results)."""
    if layouts is None:
        _, layouts = build_layouts(routing, cfg)
    decision = decide(D.build_traces(routing, cfg.train_ids), layouts, cfg)

    test = D.build_traces(routing, cfg.test_ids)
    never = S.run(test, layouts, cfg, cfg.test_ids, allow_migration=False, record=True)
    always = S.run(test, layouts, cfg, cfg.test_ids, allow_migration=True, record=True)
    chosen = always if decision.approve else never

    def row(name, res):
        return {"policy": name, **{k: v for k, v in res.summary.items() if k != "policy"}}

    policies = pd.DataFrame([row("never_migrate", never), row("always_migrate", always), row("cost_aware", chosen)])
    requests = pd.concat([never.requests.assign(policy="never_migrate"),
                          always.requests.assign(policy="always_migrate"),
                          chosen.requests.assign(policy="cost_aware")], ignore_index=True)
    return policies, requests, decision, always.migrations, {"never_migrate": never, "always_migrate": always}


def decision_frame(decision: Decision, cfg: C.SimConfig) -> pd.DataFrame:
    """Same schema as outputs/policy_decision.csv."""
    return pd.DataFrame([{
        "decision": decision.decision,
        "forecast_request_ids": ",".join(map(str, cfg.train_ids)),
        "evaluation_request_ids": ",".join(map(str, cfg.test_ids)),
        "forecast_makespan_gain_pct": decision.forecast_makespan_gain_pct,
        "forecast_p95_regression_pct": decision.forecast_p95_regression_pct,
        "min_makespan_gain_pct": cfg.min_makespan_gain_pct,
        "max_p95_regression_pct": cfg.max_p95_regression_pct,
        "migration_feasible": decision.migration_feasible,
    }])


# ---------------------------------------------------------------- plain-language explanation
@dataclass
class Explanation:
    """Why the controller decided what it did, in plain language, with evidence and counterfactuals.

    Everything here is computed from TRAINING requests only (the same data the decision used), so explaining a
    decision never leaks held-out information."""
    approved: bool
    headline: str
    checks: list[dict]
    cause: list[str]
    per_request: pd.DataFrame
    what_would_change: list[str]


def _staging_needed(layouts: dict[str, S.Layout], cfg: C.SimConfig) -> int:
    old, new = layouts[cfg.source_strategy], layouts[cfg.target_strategy]
    return max(sum(v == g for v in old.values()) + sum(old[k] != g and new[k] == g for k in old)
               for g in range(cfg.num_gpus))


def explain(train_traces: S.Traces, layouts: dict[str, S.Layout], cfg: C.SimConfig,
            decision: Decision | None = None, search: bool = True) -> Explanation:
    decision = decision or decide(train_traces, layouts, cfg)
    ids = tuple(cfg.train_ids)
    never = S.run(train_traces, layouts, cfg, ids, allow_migration=False)
    mig = S.run(train_traces, layouts, cfg, ids, allow_migration=True)
    per = never.requests[["request_id", "arrival_ms", "latency_ms"]].merge(
        mig.requests[["request_id", "latency_ms", "layout"]], on="request_id", suffixes=("_never", "_migrate"))
    per["change_pct"] = 100 * (per.latency_ms_migrate - per.latency_ms_never) / per.latency_ms_never
    act = mig.summary["migration_activation_ms"]
    copy_end = act if act is not None else float("inf")
    per["running_during_copy"] = (per.arrival_ms <= copy_end) & (
        per.arrival_ms + per.latency_ms_migrate >= cfg.migration_start_ms)

    gain, reg = decision.forecast_makespan_gain_pct, decision.forecast_p95_regression_pct
    moved = decision.forecast_migrate["moved_experts"]
    feasible_ok, gain_ok, p95_ok = (ok for _, ok in decision.reasons)
    checks = [
        {"check": "Enough GPU memory during the copy", "passed": feasible_ok,
         "meaning": ("Old and new copies of the moved experts fit on every GPU at the same time." if feasible_ok else
                     f"At least one GPU would run out of room while holding old and new copies "
                     f"({decision.forecast_migrate.get('failure_reason') or 'staging capacity exceeded'}).")},
        {"check": f"Overall time improves by at least {cfg.min_makespan_gain_pct:g}%", "passed": gain_ok,
         "meaning": (f"All requests together finish {gain:.1f}% sooner, which clears the {cfg.min_makespan_gain_pct:g}% bar."
                     if gain_ok else
                     (f"Overall time improves by only {gain:.1f}%, less than the {cfg.min_makespan_gain_pct:g}% needed "
                      f"to justify copying {moved} experts." if gain > 0 else
                      f"Overall time gets {-gain:.1f}% WORSE: copying {moved} experts costs more time than the better "
                      "layout saves."))},
        {"check": f"The slowest request is not slowed by more than {cfg.max_p95_regression_pct:g}%", "passed": p95_ok,
         "meaning": (f"The slowest request changes by {reg:+.1f}%, within the {cfg.max_p95_regression_pct:+g}% limit."
                     if p95_ok else
                     f"The slowest request would be {reg:.1f}% slower; the rule allows at most "
                     f"{cfg.max_p95_regression_pct:+g}%.")},
    ]

    if not feasible_ok:  # the move never happens, so the timing checks say nothing
        for c in checks[1:]:
            c["passed"] = None
            c["meaning"] = "Not evaluated: the move cannot start because the GPUs lack room for the copies."

    cause = []
    worst = per.loc[per.latency_ms_migrate.idxmax()]
    rid = int(worst.request_id)
    if worst.layout == cfg.source_strategy and worst.running_during_copy and worst.change_pct > 0:
        cause.append(f"Question {rid} was already running on the old layout when copying began at "
                     f"{cfg.migration_start_ms:g} ms. Its data had to wait behind the {moved} expert copies on the shared "
                     f"network links, so it took {worst.latency_ms_migrate:.0f} ms instead of {worst.latency_ms_never:.0f} ms "
                     f"({worst.change_pct:+.1f}%).")
    elif worst.layout == cfg.target_strategy and worst.change_pct > 0:
        cause.append(f"Question {rid} ran on the new layout and was {worst.change_pct:.1f}% slower on it, because more of "
                     "its chosen experts sit on GPUs away from the gateway GPU.")
    faster = per[per.change_pct < 0]
    if len(faster):
        cause.append("Questions that arrived after the switch got faster: " + ", ".join(
            f"question {int(r.request_id)} ({r.change_pct:+.1f}%)" for r in faster.itertuples()) + ".")
    if not p95_ok and feasible_ok:
        cause.append(f"With only {len(ids)} questions, the 'slowest request' (p95) is essentially one request, so a "
                     "single request caught in the copy decides the outcome.")

    changes = []
    if not decision.approve and search:
        if not feasible_ok:
            need = _staging_needed(layouts, cfg)
            changes.append(f"Give each GPU room for at least {need} experts during the copy (now {cfg.staging_slots}).")
        if not p95_ok and feasible_ok:
            alt = decide(train_traces, layouts, cfg.with_(max_p95_regression_pct=float(f"{reg + 0.05:.1f}")))
            changes.append(f"Accept the slowest request getting up to {reg + 0.05:.1f}% slower (limit now "
                           f"{cfg.max_p95_regression_pct:+g}%)" + (" — this alone would approve the move." if alt.approve
                                                                     else ", together with the other changes listed."))
        if not gain_ok and gain > 0 and feasible_ok:
            changes.append(f"Require a smaller overall gain (≤ {gain:.1f}%, now {cfg.min_makespan_gain_pct:g}%).")
        faster_bw = [b for b in (3, 6, 12, 25, 50, 100, 200, 400, 1000) if b > cfg.bandwidth_gb_per_s] if feasible_ok else []
        hit = next((b for b in faster_bw if decide(train_traces, layouts, cfg.with_(bandwidth_gb_per_s=float(b))).approve), None)
        if hit:
            changes.append(f"A faster network of about {hit} GB/s (now {cfg.bandwidth_gb_per_s:g}) would get it approved "
                           "with the current limits.")
        elif faster_bw:
            top = decide(train_traces, layouts, cfg.with_(bandwidth_gb_per_s=1000.0))
            changes.append("A faster network alone is not enough: even at 1,000 GB/s the slowest request is "
                           f"{top.forecast_p95_regression_pct:+.1f}% slower, because each copy still briefly occupies "
                           f"a link ({cfg.link_latency_ms:g} ms) while a request is running.")
        start_hit = None if not feasible_ok else next((s for s in range(0, 700, 25) if s != cfg.migration_start_ms and
                          decide(train_traces, layouts, cfg.with_(migration_start_ms=float(s))).approve), None)
        if start_hit is not None:
            changes.append(f"Starting the copy at {start_hit} ms (a quieter moment) instead of "
                           f"{cfg.migration_start_ms:g} ms would get it approved.")

    if decision.approve:
        headline = (f"APPROVED: moving {moved} experts makes all requests finish {gain:.1f}% sooner without slowing the "
                    "slowest request beyond the limit.")
    else:
        short = {0: "the GPUs lack room for the copies", 1: "the overall improvement is too small",
                 2: "the slowest request would get too much slower"}
        failed = [short[i] for i, c in enumerate(checks) if c["passed"] is False]
        headline = "REJECTED because " + " and ".join(failed) + "."
    return Explanation(decision.approve, headline, checks, cause, per, changes)
