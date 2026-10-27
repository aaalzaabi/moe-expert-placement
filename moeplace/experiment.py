"""Reproducible experiment runner.

Each scenario writes to outputs/runs/<run_id>/ where
    run_id = <name>-<10-char hash of (config, routing file sha256)>
so identical parameters always map to the same directory. Results are written
to a temporary directory and renamed on success, so a failed run never leaves a
half-written result. Original files in outputs/ are never touched.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__
from . import config as C
from . import data as D
from . import placement as P
from . import policy as POL
from .paths import OUTPUTS, RUNS

ASSUMPTIONS = ("All GPU speeds, compute cost per assignment, activation sizes, network bandwidth and latency are "
               "HYPOTHETICAL simulation parameters. Routing is measured on OLMoE (Apple MPS). "
               "Results are simulated, not real multi-GPU measurements.")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def routing_path(source: str | Path | None) -> Path:
    if source is None:
        return OUTPUTS / "workload_routing.csv"
    p = Path(source)
    return p / "workload_routing.csv" if p.is_dir() else p


def run_id(name: str, cfg: C.SimConfig, routing_sha: str) -> str:
    blob = json.dumps({"cfg": cfg.to_dict(), "routing": routing_sha}, sort_keys=True, default=list)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return f"{safe}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"


def _versions() -> dict:
    import numpy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pd.__version__,
            "moeplace": __version__, "platform": platform.platform()}


def run_scenario(cfg: C.SimConfig, name: str = "scenario", routing_source=None, out_root: Path = RUNS,
                 overwrite: bool = False, log=print) -> Path:
    cfg.validate()
    rpath = routing_path(routing_source)
    routing = D.load_routing(rpath)
    sha = file_sha256(rpath)
    rid = run_id(name, cfg, sha)
    final = Path(out_root) / rid
    if final.exists() and (final / "metadata.json").exists() and not overwrite:
        log(f"[skip] {rid} already exists (identical parameters). Use --overwrite to recompute.")
        return final

    tmp = Path(out_root) / f".tmp-{rid}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    started = time.perf_counter()
    try:
        placements, layouts = POL.build_layouts(routing, cfg)
        policies, requests, decision, migrations, results = POL.evaluate(routing, cfg, layouts)
        placements.to_csv(tmp / "placements.csv", index=False)
        P.summary(placements, "test_demand", cfg).to_csv(tmp / "placement_heldout_summary.csv", index=False)
        policies.to_csv(tmp / "policy_comparison.csv", index=False)
        requests.to_csv(tmp / "policy_request_latency.csv", index=False)
        dec = POL.decision_frame(decision, cfg)
        dec["reasons"] = " | ".join(f"{'PASS' if ok else 'FAIL'}: {txt}" for txt, ok in decision.reasons)
        dec.to_csv(tmp / "policy_decision.csv", index=False)
        migrations.to_csv(tmp / "migration_events.csv", index=False)
        pd.DataFrame(results["always_migrate"].states).to_csv(tmp / "migration_states.csv", index=False)
        meta = {
            "run_id": rid, "name": name, "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "command": " ".join(sys.argv), "config": cfg.to_dict(),
            "routing_file": str(rpath), "routing_sha256": sha,
            "request_ids": {"train (decision only)": list(cfg.train_ids), "test (evaluation only)": list(cfg.test_ids)},
            "decision": decision.decision, "result_kind": "simulated", "assumptions": ASSUMPTIONS,
            "versions": _versions(),
        }
        (tmp / "metadata.json").write_text(json.dumps(meta, indent=2, default=list))
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
    except Exception:
        (tmp / "ERROR.txt").write_text(traceback.format_exc())
        log(f"[error] {rid} failed; traceback kept in {tmp}")
        raise
    log(f"[done] {rid}: decision={decision.decision}  ({meta['elapsed_s']} s)")
    return final


def sweep(base: C.SimConfig, grid: dict[str, list], name: str = "sweep", routing_source=None,
          out_root: Path = RUNS, overwrite: bool = False, log=print) -> Path:
    """Cartesian sweep over config fields. Writes <out_root>/<sweep_id>/summary.csv."""
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    blob = json.dumps({"base": base.to_dict(), "grid": grid}, sort_keys=True, default=list)
    sweep_dir = Path(out_root) / f"{name}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], 0
    for i, values in enumerate(combos, 1):
        changes = dict(zip(keys, values))
        label = ", ".join(f"{k}={v}" for k, v in changes.items())
        log(f"[{i}/{len(combos)}] {label}")
        try:
            cfg = base.with_(**changes)
            run_dir = run_scenario(cfg, name=name, routing_source=routing_source, out_root=sweep_dir / "runs",
                                   overwrite=overwrite, log=log)
        except Exception as exc:  # keep sweeping; record the failure
            failures += 1
            log(f"    failed: {exc}")
            rows.append({**changes, "status": f"error: {exc}"})
            continue
        pol = pd.read_csv(run_dir / "policy_comparison.csv").set_index("policy")
        dec = pd.read_csv(run_dir / "policy_decision.csv").iloc[0]
        never, always = pol.loc["never_migrate"], pol.loc["always_migrate"]
        rows.append({
            **changes, "status": "ok", "run_id": run_dir.name, "decision": dec.decision,
            "forecast_makespan_gain_pct": dec.forecast_makespan_gain_pct,
            "forecast_p95_regression_pct": dec.forecast_p95_regression_pct,
            "p95_never_ms": never.p95_latency_ms, "p95_always_ms": always.p95_latency_ms,
            "makespan_never_ms": never.makespan_ms, "makespan_always_ms": always.makespan_ms,
            "activation_ms": always.migration_activation_ms, "migration_mb": always.migration_mb,
            "heldout_p95_improvement_pct": 100 * (never.p95_latency_ms - always.p95_latency_ms) / never.p95_latency_ms,
            "heldout_makespan_improvement_pct": 100 * (never.makespan_ms - always.makespan_ms) / never.makespan_ms,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(sweep_dir / "summary.csv", index=False)
    (sweep_dir / "sweep.json").write_text(json.dumps({
        "base_config": base.to_dict(), "grid": grid, "runs": len(combos), "failures": failures,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "assumptions": ASSUMPTIONS,
    }, indent=2, default=list))
    log(f"Sweep complete: {len(combos) - failures}/{len(combos)} ok -> {sweep_dir}")
    return sweep_dir


def list_runs(root: Path = RUNS) -> pd.DataFrame:
    rows = []
    if root.exists():
        for meta in sorted(root.rglob("metadata.json")):
            if any(part.startswith(".tmp-") for part in meta.parts):
                continue
            m = json.loads(meta.read_text())
            cfg = m.get("config", {})
            rows.append({"run_id": m["run_id"], "kind": m.get("result_kind", "simulated"), "decision": m.get("decision"),
                         "created_utc": m.get("created_utc"),
                         "bandwidth_gb_per_s": cfg.get("bandwidth_gb_per_s"),
                         "expert_mb": cfg["expert_bytes"] / C.MB if "expert_bytes" in cfg else None,
                         "path": str(meta.parent)})
    return pd.DataFrame(rows)


def save_frames(name: str, params: dict, frames: dict, out_root: Path = RUNS, result_kind: str = "simulated",
                assumptions: str = ASSUMPTIONS, routing_source=None, overwrite: bool = True, log=print) -> Path:
    """Generic reproducible run writer used by the multi-site, forecasting, agent and scalability experiments.

    Run id = <name>-<hash(params, routing sha256)>. Files are written to a temporary directory and renamed."""
    rpath = routing_path(routing_source)
    sha = file_sha256(rpath)
    blob = json.dumps({"params": params, "routing": sha}, sort_keys=True, default=str)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    rid = f"{safe}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"
    final = Path(out_root) / rid
    if final.exists() and not overwrite:
        log(f"[skip] {rid} exists")
        return final
    tmp = Path(out_root) / f".tmp-{rid}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for fname, df in frames.items():
        path = tmp / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    meta = {"run_id": rid, "name": name, "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv), "params": params, "routing_file": str(rpath), "routing_sha256": sha,
            "result_kind": result_kind, "assumptions": assumptions, "files": sorted(frames), "versions": _versions()}
    (tmp / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    log(f"[done] {final}")
    return final


def latest_run(prefix: str, root: Path = RUNS) -> Path | None:
    """Most recently created run directory whose name starts with `prefix` (None if absent)."""
    if not root.exists():
        return None
    cands = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix + "-") and (p / "metadata.json").exists()]
    if not cands:
        return None
    return max(cands, key=lambda p: json.loads((p / "metadata.json").read_text()).get("created_utc", ""))
