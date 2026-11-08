"""Persistence: SQLAlchemy models for plans, proposals, approvals and the audit log.

The same models run on SQLite (default, zero setup) and PostgreSQL (set MOE_DATABASE_URL). Rules that must hold
even if the application has a bug are enforced by the database itself:

* at most one plan has status 'active'          (partial unique index uq_plans_one_active)
* every status column only takes known values   (CHECK constraints)
* a proposal has at most one human decision      (UNIQUE approvals.proposal_id)
* an idempotency key creates at most one proposal (UNIQUE proposals.idempotency_key)
* references between rows are real              (foreign keys; enabled explicitly on SQLite)

Transaction boundaries are defined in service/workflow.py: each state transition (create proposal, decide,
apply) writes its rows and its audit events in ONE transaction, so the audit log can never miss a change.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, Text,
                        create_engine, event, text)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

PLAN_STATUSES = ("candidate", "active", "superseded")
PROPOSAL_STATUSES = ("rejected", "awaiting_approval", "ready", "applied", "denied", "stale")
PROPOSAL_REASONS = ("initial", "rebalance", "restore")
DECISIONS = ("approve", "deny")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _in(column: str, values) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    pass


class PlanRow(Base):
    """A placement computed by the planner. 'candidate' until a proposal that carries it is applied."""
    __tablename__ = "plans"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String(16), default="candidate")
    scenario: Mapped[dict] = mapped_column(JSON)      # validated request (topology + demand) the plan was made for
    assignment: Mapped[dict] = mapped_column(JSON)    # hosts / primary / serve, device indices of the scenario
    metrics: Mapped[dict] = mapped_column(JSON)       # predicted metrics (analytic model, HYPOTHETICAL hardware)
    devices: Mapped[list] = mapped_column(JSON)       # per device: name, site, experts hosted, utilisation
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint(_in("status", PLAN_STATUSES), name="ck_plans_status"),
        Index("uq_plans_one_active", "status", unique=True,
              sqlite_where=text("status = 'active'"), postgresql_where=text("status = 'active'")),
    )


class ProposalRow(Base):
    """A request to replace the active plan, with the authoriser's verdict."""
    __tablename__ = "proposals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("plans.id"))
    base_plan_id: Mapped[int | None] = mapped_column(ForeignKey("plans.id"))  # active plan when it was assessed
    status: Mapped[str] = mapped_column(String(24))
    reason: Mapped[str] = mapped_column(String(16))
    objective_current: Mapped[float | None] = mapped_column(Float)  # None: no valid active plan
    objective_new: Mapped[float] = mapped_column(Float)
    util_new: Mapped[float] = mapped_column(Float)
    migration: Mapped[dict] = mapped_column(JSON)
    checks: Mapped[list] = mapped_column(JSON)        # [{"check": str, "passed": bool}, ...]
    needs_human: Mapped[bool] = mapped_column(Boolean)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint(_in("status", PROPOSAL_STATUSES), name="ck_proposals_status"),
        CheckConstraint(_in("reason", PROPOSAL_REASONS), name="ck_proposals_reason"),
        Index("ix_proposals_status", "status"),
    )


class ApprovalRow(Base):
    """The human decision on a proposal that needed one."""
    __tablename__ = "approvals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    proposal_id: Mapped[int] = mapped_column(ForeignKey("proposals.id"), unique=True)
    approver: Mapped[str] = mapped_column(String(128))
    decision: Mapped[str] = mapped_column(String(8))
    comment: Mapped[str] = mapped_column(Text, default="")
    __table_args__ = (CheckConstraint(_in("decision", DECISIONS), name="ck_approvals_decision"),)


class AuditEventRow(Base):
    """Append-only log of every state change (who, what, when, why)."""
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    actor: Mapped[str] = mapped_column(String(160))    # 'api', 'authoriser' or 'human:<name>'
    event: Mapped[str] = mapped_column(String(40))
    proposal_id: Mapped[int | None] = mapped_column(ForeignKey("proposals.id"))
    plan_id: Mapped[int | None] = mapped_column(ForeignKey("plans.id"))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    __table_args__ = (Index("ix_audit_events_proposal", "proposal_id"),)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # SQLite ignores foreign keys unless asked
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    else:
        engine = create_engine(url, pool_pre_ping=True)
    return engine


def init_db(engine: Engine) -> sessionmaker:
    """Create missing tables (idempotent) and return a session factory."""
    if engine.url.get_backend_name() == "sqlite" and engine.url.database not in (None, "", ":memory:"):
        from pathlib import Path
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
