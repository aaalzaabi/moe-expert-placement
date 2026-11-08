"""Service configuration, read from environment variables (no hard-coded paths, hosts or credentials).

    MOE_DATABASE_URL            SQLAlchemy URL. Default: SQLite file data/planner.db in the project.
                                PostgreSQL: postgresql+psycopg://user:password@host:5432/dbname (password from the
                                environment or a secret store, never committed).
    MOE_LOG_LEVEL               DEBUG | INFO | WARNING | ERROR (default INFO)
    MOE_APPROVAL_GB             migrations larger than this need human approval (default 50)
    MOE_MIN_GAIN                minimum relative objective gain for a rebalance (default 0.02 = 2 %)
    MOE_MAX_MIGRATION_FRACTION  transfer time must fit in this share of the planning horizon (default 0.05)
    MOE_STICKINESS              planner hysteresis toward the active plan (default 0.05)

Thresholds default to the operations agent's values (moeplace.agent.AgentConfig), so the API and the simulated
agent apply the same rules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from moeplace.agent import AgentConfig
from moeplace.config import ConfigError
from moeplace.paths import PROJECT_ROOT

_AGENT = AgentConfig()
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def default_database_url() -> str:
    return f"sqlite:///{PROJECT_ROOT / 'data' / 'planner.db'}"


@dataclass(frozen=True)
class Settings:
    database_url: str
    log_level: str = "INFO"
    approval_gb: float = _AGENT.approval_gb
    min_gain_rel: float = _AGENT.min_gain_rel
    max_migration_fraction: float = _AGENT.max_migration_fraction
    stickiness: float = _AGENT.stickiness
    horizon_windows: int = 4   # one placement stays active for 4 x 15 min (same horizon as the forecaster)

    def validate(self) -> "Settings":
        errs = []
        if not self.database_url:
            errs.append("MOE_DATABASE_URL must not be empty")
        if self.log_level not in LOG_LEVELS:
            errs.append(f"MOE_LOG_LEVEL must be one of {LOG_LEVELS}, got {self.log_level!r}")
        if self.approval_gb < 0:
            errs.append("MOE_APPROVAL_GB must be >= 0")
        if not 0 <= self.min_gain_rel < 1:
            errs.append("MOE_MIN_GAIN must be in [0, 1)")
        if not 0 < self.max_migration_fraction <= 1:
            errs.append("MOE_MAX_MIGRATION_FRACTION must be in (0, 1]")
        if not 0 <= self.stickiness < 1:
            errs.append("MOE_STICKINESS must be in [0, 1)")
        if errs:
            raise ConfigError("Invalid service settings:\n  - " + "\n  - ".join(errs))
        return self

    def agent_config(self) -> AgentConfig:
        return AgentConfig(min_gain_rel=self.min_gain_rel, max_migration_fraction=self.max_migration_fraction,
                           approval_gb=self.approval_gb, stickiness=self.stickiness)

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env

        def num(key, default):
            raw = env.get(key)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except ValueError:
                raise ConfigError(f"{key} must be a number, got {raw!r}") from None

        return cls(
            database_url=env.get("MOE_DATABASE_URL") or default_database_url(),
            log_level=env.get("MOE_LOG_LEVEL", "INFO").upper(),
            approval_gb=num("MOE_APPROVAL_GB", _AGENT.approval_gb),
            min_gain_rel=num("MOE_MIN_GAIN", _AGENT.min_gain_rel),
            max_migration_fraction=num("MOE_MAX_MIGRATION_FRACTION", _AGENT.max_migration_fraction),
            stickiness=num("MOE_STICKINESS", _AGENT.stickiness),
        ).validate()
