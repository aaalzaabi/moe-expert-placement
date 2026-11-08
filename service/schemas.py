"""Request and response models of the planner API (validated by pydantic; invalid input -> HTTP 422).

A Scenario is everything the planner needs for one decision: the sites, their devices and the expected demand of
the next planning horizon (requests per 15-minute window and the category mix). Device speeds and network numbers
are HYPOTHETICAL inputs; the expert-routing profile per category is MEASURED on OLMoE (outputs/workload_routing.csv).
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Name = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeviceIn(_Strict):
    name: str = Name
    speed: float = Field(1.0, gt=0, le=100, description="relative throughput (hypothetical)")
    slots: int = Field(256, ge=1, le=4096, description="expert slots on the device")
    memory_gb: float | None = Field(None, gt=0, description="memory for expert weights; null = slots only")


class SiteIn(_Strict):
    name: str = Name
    region: str = Field("", max_length=64)
    residency: bool = Field(False, description="tokens from this site must be served by experts hosted here")
    devices: list[DeviceIn] = Field(min_length=1, max_length=64)
    requests_per_window: float = Field(ge=0, le=1e7, description="expected requests per 15-minute window")
    mix: dict[str, float] = Field(description="category -> share, e.g. {'ai': 0.5, 'coding': 0.3, 'general': 0.2}")

    @field_validator("mix")
    @classmethod
    def _mix(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("mix must name at least one category")
        if any(x < 0 for x in v.values()):
            raise ValueError("mix shares must be >= 0")
        if sum(v.values()) <= 0:
            raise ValueError("mix shares must sum to more than 0")
        return v


class Scenario(_Strict):
    sites: list[SiteIn] = Field(min_length=1, max_length=16)
    tokens_per_request: float = Field(400.0, gt=0, le=1e5)
    replica_frac: float = Field(0.0, ge=0, le=1, description="extra copies as a share of the 1,024 experts")
    util_cap: float | None = Field(None, gt=0, le=1, description="null = balance mode; a value = locality mode")
    inter_site_gb_per_s: float = Field(1.25, gt=0, le=1e4)
    inter_site_rtt_ms: float = Field(60.0, ge=0, le=1e4)
    intra_site_gb_per_s: float = Field(12.0, gt=0, le=1e4)

    @model_validator(mode="after")
    def _unique_names(self) -> "Scenario":
        sites = [s.name for s in self.sites]
        devices = [d.name for s in self.sites for d in s.devices]
        if len(set(sites)) != len(sites):
            raise ValueError("site names must be unique")
        if len(set(devices)) != len(devices):
            raise ValueError("device names must be unique across all sites")
        return self


class DecisionIn(_Strict):
    approver: str = Field(min_length=1, max_length=128)
    comment: str = Field("", max_length=1000)


class DeviceOut(BaseModel):
    name: str
    site: str
    experts: int
    utilisation: float


class PlanOut(BaseModel):
    id: int | None = None
    status: str | None = None
    metrics: dict
    devices: list[DeviceOut]


class Check(BaseModel):
    check: str
    passed: bool


class ProposalOut(BaseModel):
    id: int
    created_at: datetime
    status: str
    reason: str
    plan_id: int
    base_plan_id: int | None
    needs_human: bool
    objective_current: float | None
    objective_new: float
    util_new: float
    migration: dict
    checks: list[Check]
    applied_at: datetime | None


class AuditOut(BaseModel):
    id: int
    created_at: datetime
    actor: str
    event: str
    proposal_id: int | None
    plan_id: int | None
    detail: dict
