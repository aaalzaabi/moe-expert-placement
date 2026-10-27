"""Centralised configuration for placement, simulation and the migration controller.

Every hardware number here is a HYPOTHETICAL simulation assumption unless it is
marked as checkpoint-derived. Units are explicit in every field name:

* bandwidth is in GB/s (gigaBYTES per second, 1 GB = 1e9 bytes) - never Gbps.
* time is in simulated milliseconds.
* sizes are in bytes.

Defaults reproduce the Phase 4D/4E/4F scripts. Named presets reproduce each
saved legacy experiment exactly (verified by tests/test_reproduction.py).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields, replace

# --------------------------------------------------------------------------
# Real model facts (from models/OLMoE-1B-7B-0125-Instruct/config.json and
# outputs/phase4f/memory/*.csv). These are NOT assumptions.
# --------------------------------------------------------------------------
MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"
NUM_LAYERS = 16
EXPERTS_PER_LAYER = 64
TOP_K = 8
TOTAL_EXPERT_SLOTS = NUM_LAYERS * EXPERTS_PER_LAYER  # 1024 layer-specific experts
HIDDEN_SIZE = 2048
EXPERT_INTERMEDIATE_SIZE = 1024
# gate_proj + up_proj + down_proj, each 2048x1024, stored as BF16 (2 bytes):
CHECKPOINT_EXPERT_BYTES = 3 * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE * 2  # 12,582,912
LEGACY_EXPERT_BYTES = 8_000_000  # original Phase 4B-4E placeholder (not calibrated)

GB = 1e9
MB = 1e6

STRATEGIES = ("static", "balanced", "demand_aware")
LINK_MODELS = ("directed", "shared_port")
ARRIVAL_PROCESSES = ("fixed", "poisson")
POLICIES = ("never_migrate", "always_migrate", "cost_aware")


def transfer_ms(size_bytes: float, bandwidth_gb_per_s: float, latency_ms: float) -> float:
    """Simulated time to move size_bytes over one link (GB/s -> bytes per ms = GB/s * 1e6)."""
    return latency_ms + size_bytes / (bandwidth_gb_per_s * 1e6)


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration values."""


@dataclass(frozen=True)
class SimConfig:
    # ---- simulated cluster (HYPOTHETICAL) ----
    gpu_speeds: tuple[float, ...] = (1.0, 1.3, 0.8, 1.6)
    gateway_gpu: int = 0
    placement_capacity_slots: int | None = 256  # experts a GPU may hold; None -> ceil(1024/n)
    staging_capacity_slots: int | None = 512    # resident + staged copies during migration; None -> 2x placement
    gpu_memory_gb: float | None = None          # optional byte budget for expert weights per GPU

    # ---- compute / communication cost model (HYPOTHETICAL) ----
    compute_ms_per_assignment: float = 0.5
    activation_bytes_per_assignment: int = 4096
    bandwidth_gb_per_s: float = 12.0
    link_latency_ms: float = 0.1
    link_model: str = "directed"  # "directed": one FIFO queue per (src, dst) pair (legacy)
                                  # "shared_port": a transfer occupies src egress AND dst ingress

    # ---- expert weights ----
    expert_bytes: int = CHECKPOINT_EXPERT_BYTES  # checkpoint-derived (BF16)

    # ---- workload ----
    train_ids: tuple[int, ...] = (0, 2, 4)
    test_ids: tuple[int, ...] = (1, 3, 5)
    arrival_gap_ms: float = 200.0
    arrival_process: str = "fixed"
    seed: int = 0

    # ---- migration ----
    source_strategy: str = "static"
    target_strategy: str = "demand_aware"
    migration_start_ms: float = 100.0
    fail_after_copies: int | None = None  # fault injection: abort after N completed copies

    # ---- cost-aware controller thresholds ----
    min_makespan_gain_pct: float = 2.0
    max_p95_regression_pct: float = 0.0

    # ------------------------------------------------------------------
    @property
    def num_gpus(self) -> int:
        return len(self.gpu_speeds)

    @property
    def capacity_slots(self) -> int:
        if self.placement_capacity_slots is not None:
            return int(self.placement_capacity_slots)
        return math.ceil(TOTAL_EXPERT_SLOTS / self.num_gpus)

    @property
    def staging_slots(self) -> int:
        if self.staging_capacity_slots is not None:
            return int(self.staging_capacity_slots)
        return 2 * self.capacity_slots

    @property
    def expert_mb(self) -> float:
        return self.expert_bytes / MB

    def validate(self) -> "SimConfig":
        errs = []
        if self.num_gpus < 1:
            errs.append("gpu_speeds must list at least one GPU")
        if any((not math.isfinite(s)) or s <= 0 for s in self.gpu_speeds):
            errs.append(f"gpu_speeds must be positive, got {self.gpu_speeds}")
        if not 0 <= self.gateway_gpu < max(self.num_gpus, 1):
            errs.append(f"gateway_gpu={self.gateway_gpu} is outside 0..{self.num_gpus - 1}")
        if self.capacity_slots * self.num_gpus < TOTAL_EXPERT_SLOTS:
            errs.append(
                f"{self.num_gpus} GPUs x {self.capacity_slots} slots = "
                f"{self.capacity_slots * self.num_gpus} < {TOTAL_EXPERT_SLOTS} experts: placement infeasible"
            )
        if self.staging_slots < self.capacity_slots:
            errs.append("staging_capacity_slots must be >= placement_capacity_slots")
        if self.gpu_memory_gb is not None and self.gpu_memory_gb <= 0:
            errs.append("gpu_memory_gb must be positive")
        if self.bandwidth_gb_per_s <= 0:
            errs.append("bandwidth_gb_per_s must be positive (units: GB/s, not Gbps)")
        if self.link_latency_ms < 0 or self.compute_ms_per_assignment <= 0:
            errs.append("link_latency_ms must be >= 0 and compute_ms_per_assignment > 0")
        if self.activation_bytes_per_assignment < 0 or self.expert_bytes <= 0:
            errs.append("activation/expert byte sizes must be positive")
        if self.link_model not in LINK_MODELS:
            errs.append(f"link_model must be one of {LINK_MODELS}")
        if self.arrival_process not in ARRIVAL_PROCESSES:
            errs.append(f"arrival_process must be one of {ARRIVAL_PROCESSES}")
        if self.arrival_gap_ms < 0:
            errs.append("arrival_gap_ms must be >= 0")
        for name in ("source_strategy", "target_strategy"):
            if getattr(self, name) not in STRATEGIES:
                errs.append(f"{name} must be one of {STRATEGIES}")
        if set(self.train_ids) & set(self.test_ids):
            errs.append(f"train_ids {self.train_ids} and test_ids {self.test_ids} overlap (evaluation leakage)")
        if not self.train_ids or not self.test_ids:
            errs.append("train_ids and test_ids must both be non-empty")
        if self.fail_after_copies is not None and self.fail_after_copies < 0:
            errs.append("fail_after_copies must be >= 0")
        if errs:
            raise ConfigError("Invalid SimConfig:\n  - " + "\n  - ".join(errs))
        return self

    def with_(self, **changes) -> "SimConfig":
        """Return a validated copy with some fields replaced."""
        for key in ("gpu_speeds", "train_ids", "test_ids"):
            if key in changes and changes[key] is not None:
                changes[key] = tuple(changes[key])
        return replace(self, **changes).validate()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SimConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown config keys: {sorted(unknown)}")
        clean = dict(data)
        for key in ("gpu_speeds", "train_ids", "test_ids"):
            if key in clean and clean[key] is not None:
                clean[key] = tuple(clean[key])
        return cls(**clean).validate()

    def fingerprint(self) -> str:
        """Stable short hash of all parameters (used in reproducible run ids)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]


DEFAULT = SimConfig().validate()

# Presets that regenerate saved legacy experiments exactly.
PRESETS: dict[str, SimConfig] = {
    # Phase 4E saved policy CSVs (policy_*.csv): 8 MB experts, 12 GB/s.
    "phase4e_legacy": SimConfig(expert_bytes=LEGACY_EXPERT_BYTES, bandwidth_gb_per_s=12.0).validate(),
    # Phase 4F network sweep (outputs/phase4f/network): 8 MB experts.
    "phase4f_network": SimConfig(expert_bytes=LEGACY_EXPERT_BYTES).validate(),
    # Phase 4F calibrated sweep (outputs/phase4f/calibrated): checkpoint-derived 12.583 MB.
    "phase4f_calibrated": SimConfig(expert_bytes=CHECKPOINT_EXPERT_BYTES).validate(),
}
PHASE4F_BANDWIDTHS = (3, 12, 25, 100)

# Phase 4A / 4B standalone scripts (simulate_inference.py, simulate_migration.py).
PHASE4A = SimConfig(bandwidth_gb_per_s=12.0, expert_bytes=LEGACY_EXPERT_BYTES).validate()
# Phase 3A (optimize_placement.py) used identical GPUs and all six requests.
PHASE3A_SPEEDS = (1.0, 1.0, 1.0, 1.0)
