"""Proposal lifecycle and its transaction boundaries.

    POST /proposals                    authoriser verdict
        ├── rejected                   (a rule failed; terminal)
        ├── awaiting_approval ──► approve ──► ready      deny ──► denied (terminal)
        └── ready ──► apply ──► applied                  (plan becomes the single active plan)
                         └────► stale  (the active plan changed since the proposal was assessed; terminal)

Each public function below is ONE database transaction that writes the state change and its audit events
together; if anything fails, nothing is written. The active plan row is locked (SELECT ... FOR UPDATE on
PostgreSQL; SQLite serialises writers) while a proposal is assessed or applied, and the partial unique index on
plans.status guarantees a single active plan even if two applies race.
"""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from . import planning as PL
from .db import ApprovalRow, AuditEventRow, PlanRow, ProposalRow, utcnow
from .schemas import Scenario
from .settings import Settings

log = logging.getLogger("service.workflow")


class NotFound(LookupError):
    pass


class Conflict(RuntimeError):
    """The request is valid but not allowed in the current state (HTTP 409)."""


def _audit(s: Session, actor: str, event: str, proposal_id=None, plan_id=None, **detail) -> None:
    s.add(AuditEventRow(actor=actor, event=event, proposal_id=proposal_id, plan_id=plan_id, detail=detail))


def _active_row(s: Session, lock: bool = False) -> PlanRow | None:
    q = select(PlanRow).where(PlanRow.status == "active")
    return s.scalar(q.with_for_update() if lock else q)


def _active(row: PlanRow | None) -> PL.Active | None:
    if row is None:
        return None
    topo = PL.topology(Scenario.model_validate(row.scenario))
    return PL.Active(PL.from_assignment(row.assignment, topo), topo)


def _proposal(s: Session, proposal_id: int, lock: bool = False) -> ProposalRow:
    row = s.get(ProposalRow, proposal_id, with_for_update=lock)
    if row is None:
        raise NotFound(f"proposal {proposal_id} not found")
    return row


# ------------------------------------------------------------------------------------------- reads
def plan_only(sc: Scenario) -> PL.Computed:
    """Stateless planning (POST /plan): no database access."""
    return PL.compute(sc)


def get_plan(factory: sessionmaker, plan_id: int) -> PlanRow:
    with factory() as s:
        row = s.get(PlanRow, plan_id)
        if row is None:
            raise NotFound(f"plan {plan_id} not found")
        return row


def active_plan(factory: sessionmaker) -> PlanRow:
    with factory() as s:
        row = _active_row(s)
        if row is None:
            raise NotFound("no active plan yet: create and apply a proposal first")
        return row


def get_proposal(factory: sessionmaker, proposal_id: int) -> ProposalRow:
    with factory() as s:
        return _proposal(s, proposal_id)


def list_proposals(factory: sessionmaker, status: str | None, limit: int) -> list[ProposalRow]:
    with factory() as s:
        q = select(ProposalRow).order_by(ProposalRow.id.desc()).limit(limit)
        if status:
            q = q.where(ProposalRow.status == status)
        return list(s.scalars(q))


def audit_log(factory: sessionmaker, proposal_id: int | None, limit: int) -> list[AuditEventRow]:
    with factory() as s:
        q = select(AuditEventRow).order_by(AuditEventRow.id).limit(limit)
        if proposal_id is not None:
            q = q.where(AuditEventRow.proposal_id == proposal_id)
        return list(s.scalars(q))


# ------------------------------------------------------------------------------------------- writes
def create_proposal(factory: sessionmaker, sc: Scenario, settings: Settings,
                    idempotency_key: str | None = None) -> tuple[ProposalRow, bool]:
    """Plan the scenario against the active plan and record the authoriser's verdict.

    Returns (proposal, created). Repeating a request with the same Idempotency-Key returns the original proposal
    (created=False) instead of planning again; reusing a key with a different scenario is a Conflict."""
    with factory.begin() as s:
        if idempotency_key:
            old = s.scalar(select(ProposalRow).where(ProposalRow.idempotency_key == idempotency_key))
            if old is not None:
                if s.get(PlanRow, old.plan_id).scenario != sc.model_dump():
                    raise Conflict("Idempotency-Key was already used with a different scenario")
                return old, False
        active_row = _active_row(s, lock=True)
        active = _active(active_row)
        PL.check_compatible(sc, active)
        new = PL.compute(sc, PL.incumbent(PL.topology(sc), active), settings.stickiness)
        a = PL.assess(new, active, settings)
        status = "rejected" if not a.approved else ("awaiting_approval" if a.needs_human else "ready")

        plan = PlanRow(scenario=sc.model_dump(), assignment=PL.to_assignment(new.plan), metrics=new.metrics,
                       devices=new.devices)
        s.add(plan)
        s.flush()
        prop = ProposalRow(idempotency_key=idempotency_key, plan_id=plan.id,
                           base_plan_id=active_row.id if active_row else None, status=status, reason=a.reason,
                           objective_current=a.objective_current, objective_new=a.objective_new,
                           util_new=a.util_new, migration=a.migration, checks=a.checks, needs_human=a.needs_human)
        s.add(prop)
        s.flush()
        _audit(s, "api", "proposal_created", prop.id, plan.id, reason=a.reason, migration_gb=a.migration["gb"])
        failed = [c["check"] for c in a.checks if not c["passed"]]
        _audit(s, "authoriser", {"rejected": "authoriser_rejected", "awaiting_approval": "approval_requested",
                                 "ready": "authoriser_approved"}[status], prop.id, plan.id, failed_checks=failed)
    log.info("proposal %s created: reason=%s status=%s J_new=%.3f migration=%.2f GB failed=%s",
             prop.id, a.reason, status, a.util_new, a.migration["gb"], failed or "-")
    return prop, True


def decide(factory: sessionmaker, proposal_id: int, approve: bool, approver: str, comment: str = "") -> ProposalRow:
    """Record a human decision on a proposal that is awaiting approval."""
    with factory.begin() as s:
        prop = _proposal(s, proposal_id, lock=True)
        if prop.status != "awaiting_approval":
            raise Conflict(f"proposal {proposal_id} is '{prop.status}'; only 'awaiting_approval' can be decided")
        decision = "approve" if approve else "deny"
        s.add(ApprovalRow(proposal_id=prop.id, approver=approver, decision=decision, comment=comment))
        prop.status = "ready" if approve else "denied"
        _audit(s, f"human:{approver}", "human_approved" if approve else "human_denied", prop.id, prop.plan_id,
               comment=comment)
    log.info("proposal %s %s by %s", proposal_id, "approved" if approve else "denied", approver)
    return prop


def apply(factory: sessionmaker, proposal_id: int) -> ProposalRow:
    """Make the proposal's plan the active plan, if the active plan is still the one it was assessed against."""
    stale = False
    with factory.begin() as s:
        prop = _proposal(s, proposal_id, lock=True)
        if prop.status != "ready":
            raise Conflict(f"proposal {proposal_id} is '{prop.status}'; only 'ready' proposals can be applied")
        active = _active_row(s, lock=True)
        active_id = active.id if active else None
        if active_id != prop.base_plan_id:
            # Assessed against a plan that is no longer active: its gain and migration cost are no longer valid.
            prop.status = "stale"
            _audit(s, "api", "proposal_stale", prop.id, prop.plan_id,
                   assessed_against=prop.base_plan_id, active_now=active_id)
            stale = True
        else:
            now = utcnow()
            if active is not None:
                active.status, active.superseded_at = "superseded", now
                s.flush()  # release the single 'active' slot before taking it
            plan = s.get(PlanRow, prop.plan_id)
            plan.status, plan.activated_at = "active", now
            prop.status, prop.applied_at = "applied", now
            _audit(s, "api", "plan_activated", prop.id, plan.id, replaced_plan=active_id,
                   migration_gb=prop.migration.get("gb"))
    if stale:
        log.warning("proposal %s is stale (assessed against plan %s, active is %s)",
                    proposal_id, prop.base_plan_id, active_id)
        raise Conflict(f"proposal {proposal_id} was assessed against plan {prop.base_plan_id}, but plan "
                       f"{active_id} is active now; create a new proposal")
    log.info("proposal %s applied: plan %s active (replaced %s)", proposal_id, prop.plan_id, active_id)
    return prop
