"""Planner API: plan, propose, approve and apply expert placements, with a persistent audit log.

Run (from the project root):
    .venv/bin/uvicorn service.app:create_app --factory --port 8000
Interactive API documentation: http://localhost:8000/docs

Error contract (body is always {"detail": ...}):
    404  unknown proposal / plan
    409  not allowed in the current state (wrong status, stale proposal, Idempotency-Key reuse)
    422  invalid input, or a scenario that cannot be served (capacity, memory, residency, unknown category)
    503  database or measured routing data unavailable
    500  unexpected error (logged with its request id; no internals in the response)
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Literal

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

from moeplace.config import ConfigError
from moeplace.multisite import InfeasiblePlan

from . import __version__
from . import workflow as WF
from .db import PROPOSAL_STATUSES, init_db, make_engine
from .planning import ScenarioError, measured_profiles
from .schemas import AuditOut, DecisionIn, PlanOut, ProposalOut, Scenario
from .settings import Settings

log = logging.getLogger("service.api")


def _plan_out(row) -> PlanOut:
    return PlanOut(id=row.id, status=row.status, metrics=row.metrics, devices=row.devices)


def _proposal_out(row) -> ProposalOut:
    return ProposalOut.model_validate(row, from_attributes=True)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    engine = make_engine(settings.database_url)
    factory = init_db(engine)
    log.info("planner API %s using %s", __version__, engine.url.render_as_string(hide_password=True))

    app = FastAPI(title="MoE Expert Placement Planner API", version=__version__,
                  description="Plan, propose, approve and apply expert placements. Expert routing is measured on "
                              "OLMoE; GPU speeds and network numbers are hypothetical inputs.")
    app.state.settings, app.state.engine, app.state.sessions = settings, engine, factory

    # ------------------------------------------------------------------------------ cross-cutting
    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.rid = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info("%s %s %s %.1f ms rid=%s", request.method, request.url.path, response.status_code,
                 (time.perf_counter() - t0) * 1e3, rid)
        return response

    def _error(status: int, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.exception_handler(WF.NotFound)
    async def _404(_, exc):
        return _error(404, exc)

    @app.exception_handler(WF.Conflict)
    async def _409(_, exc):
        return _error(409, exc)

    @app.exception_handler(IntegrityError)
    async def _409_db(_, exc):
        log.warning("integrity error: %s", exc.orig)
        return _error(409, Exception("conflicting concurrent change; retry the request"))

    for exc_type in (ScenarioError, InfeasiblePlan, ConfigError):
        @app.exception_handler(exc_type)
        async def _422(_, exc):
            log.info("unservable scenario: %s", exc)
            return _error(422, exc)

    @app.exception_handler(OperationalError)
    async def _503(request: Request, exc):
        log.error("database unavailable rid=%s: %s", getattr(request.state, "rid", "-"), exc.orig)
        return _error(503, Exception("database unavailable"))

    @app.exception_handler(Exception)
    async def _500(request: Request, exc):
        log.exception("unhandled error rid=%s", getattr(request.state, "rid", "-"))
        return _error(500, Exception(f"internal error (request id {getattr(request.state, 'rid', '-')})"))

    # ------------------------------------------------------------------------------ routes
    @app.get("/health", tags=["service"])
    def health():
        """Liveness and readiness: database reachable and measured routing profiles loadable."""
        checks = {}
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except OperationalError as exc:
            checks["database"] = f"error: {exc.orig}"
        try:
            cats, _ = measured_profiles()
            checks["routing_profiles"] = f"ok ({', '.join(cats)})"
        except Exception as exc:  # missing or invalid outputs/workload_routing.csv
            checks["routing_profiles"] = f"error: {exc}"
        ok = all(v.startswith("ok") for v in checks.values())
        return JSONResponse(status_code=200 if ok else 503,
                            content={"status": "ok" if ok else "degraded", "version": __version__, **checks})

    @app.post("/plan", response_model=PlanOut, tags=["planning"])
    def plan(sc: Scenario):
        """Plan a scenario without storing anything (what-if analysis)."""
        c = WF.plan_only(sc)
        return PlanOut(metrics=c.metrics, devices=c.devices)

    @app.post("/proposals", response_model=ProposalOut, status_code=201, tags=["proposals"])
    def create_proposal(sc: Scenario,
                        idempotency_key: str | None = Header(None, max_length=128,
                                                             description="repeat-safe retries")):
        """Plan the scenario against the active plan and record the authoriser's verdict:
        'ready', 'awaiting_approval' (human needed) or 'rejected'."""
        row, created = WF.create_proposal(factory, sc, settings, idempotency_key)
        return JSONResponse(status_code=201 if created else 200,
                            content=_proposal_out(row).model_dump(mode="json"))

    @app.get("/proposals", response_model=list[ProposalOut], tags=["proposals"])
    def list_proposals(status: Literal[PROPOSAL_STATUSES] | None = None,
                       limit: int = Query(50, ge=1, le=500)):
        return [_proposal_out(r) for r in WF.list_proposals(factory, status, limit)]

    @app.get("/proposals/{proposal_id}", response_model=ProposalOut, tags=["proposals"])
    def get_proposal(proposal_id: int):
        return _proposal_out(WF.get_proposal(factory, proposal_id))

    @app.post("/proposals/{proposal_id}/approve", response_model=ProposalOut, tags=["proposals"])
    def approve(proposal_id: int, body: DecisionIn):
        """Human approval of a proposal in 'awaiting_approval'."""
        return _proposal_out(WF.decide(factory, proposal_id, True, body.approver, body.comment))

    @app.post("/proposals/{proposal_id}/deny", response_model=ProposalOut, tags=["proposals"])
    def deny(proposal_id: int, body: DecisionIn):
        return _proposal_out(WF.decide(factory, proposal_id, False, body.approver, body.comment))

    @app.post("/proposals/{proposal_id}/apply", response_model=ProposalOut, tags=["proposals"])
    def apply(proposal_id: int):
        """Activate a 'ready' proposal's plan. 409 if the active plan changed since it was assessed."""
        return _proposal_out(WF.apply(factory, proposal_id))

    @app.get("/plans/active", response_model=PlanOut, tags=["plans"])
    def active():
        return _plan_out(WF.active_plan(factory))

    @app.get("/plans/{plan_id}", response_model=PlanOut, tags=["plans"])
    def get_plan(plan_id: int):
        return _plan_out(WF.get_plan(factory, plan_id))

    @app.get("/audit", response_model=list[AuditOut], tags=["audit"])
    def audit(proposal_id: int | None = None, limit: int = Query(200, ge=1, le=1000)):
        """Append-only audit log, oldest first."""
        return [AuditOut.model_validate(r, from_attributes=True) for r in WF.audit_log(factory, proposal_id, limit)]

    return app
