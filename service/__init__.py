"""Planner service: REST API + database around the moeplace core.

Modules
-------
settings   configuration from environment variables
schemas    validated request / response models
planning   scenario -> topology + measured demand -> plan -> authoriser verdict (adapter over moeplace)
workflow   proposal lifecycle; one database transaction per state change, with audit events
db         SQLAlchemy models (plans, proposals, approvals, audit_events) for SQLite or PostgreSQL
app        FastAPI routes and the HTTP error contract
"""
__version__ = "1.0.0"
