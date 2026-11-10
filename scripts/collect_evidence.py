"""Collect the testing and evaluation evidence for the implementation report into outputs/evidence/.

    .venv/bin/python scripts/collect_evidence.py            # ~40 s; never loads OLMoE
    .venv/bin/python scripts/collect_evidence.py --skip-tests

Writes
    test_results.csv       one row per automated test: module, category, negative?, outcome, duration
    test_summary.csv       counts per category
    placement_vs_random.csv  held-out peak load of each strategy vs 10,000 random placements and the lower bound
    replicates_summary.csv   30 seeded Poisson arrival scenarios per bandwidth: mean, 95% CI, win rate,
                             controller decision accuracy against the held-out outcome
    EVIDENCE.md            the tables above plus saved scalability and use-case results, in one readable file

Inputs it reads (it does not re-run them): outputs/heterogeneous_placements.csv, outputs/runs/replicates30*/,
outputs/runs/scalability-*/, outputs/runs/usecases-*/, outputs/evidence/e2e_*.json and api_bench.csv when present.
Produce the replicate runs first with
    .venv/bin/python scripts/run_experiment.py --name replicates30 --arrival-process poisson \
        --arrival-rate-rps 5 --seed $(seq 0 29) --policy all
    (and the same with --name replicates30_bw100 --bandwidth 100)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moeplace import config as C  # noqa: E402
from moeplace import placement as P  # noqa: E402

OUT = ROOT / "outputs" / "evidence"
RUNS = ROOT / "outputs" / "runs"

# Test level of each module. A module's docstring explains what it exercises; see the report, section 5.
CATEGORY = {
    "test_placement": "unit", "test_simulator_invariants": "unit", "test_policy": "unit", "test_forecast": "unit",
    "test_data_and_schemas": "unit", "test_multisite": "unit", "test_gpu_viz": "unit", "test_assistant": "unit",
    "test_experiment_runner": "integration", "test_reproduction": "integration (regression)",
    "test_agent_usecases": "integration", "test_real_expert_parallel": "integration",
    "test_live_answer": "integration", "test_service": "integration + end-to-end (API/DB)",
    "test_dashboard": "end-to-end (UI)", "test_dashboard_multisite": "end-to-end (UI)",
    "test_presentation_flow": "end-to-end (UI)", "test_extended_prompts": "integration (regression)",
}
NEGATIVE_WORDS = ("reject", "invalid", "error", "fail", "missing", "bad", "unknown", "block", "stale", "conflict",
                  "denied", "deny", "raise", "refuse", "violat", "leak", "corrupt", "empty", "negative",
                  "unavailable", "rollback", "idempot", "infeasible", "overwrite", "guard", "illegal")


class TimingResult(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rows, self._t = [], {}

    def startTest(self, test):
        self._t[test.id()] = time.perf_counter()
        super().startTest(test)

    def _record(self, test, outcome):
        tid = test.id()
        parts = tid.split(".")
        module = parts[1] if len(parts) > 2 else parts[0]
        name = parts[-1]
        self.rows.append({"module": module, "class": parts[-2] if len(parts) > 2 else "", "test": name,
                          "category": CATEGORY.get(module, "other"),
                          "negative": any(w in name.lower() for w in NEGATIVE_WORDS),
                          "outcome": outcome,
                          "duration_s": round(time.perf_counter() - self._t.get(tid, time.perf_counter()), 4)})

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "pass")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "fail")

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "error")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "skip")


def run_tests() -> pd.DataFrame:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    runner = unittest.TextTestRunner(resultclass=TimingResult, verbosity=0, stream=open("/dev/null", "w"))
    t0 = time.perf_counter()
    result = runner.run(suite)
    df = pd.DataFrame(result.rows)
    df.attrs["wall_s"] = time.perf_counter() - t0
    return df


def ci95(x: pd.Series) -> tuple[float, float, float]:
    x = x.dropna().astype(float)
    m, s, n = x.mean(), x.std(ddof=1), len(x)
    h = 2.045 * s / math.sqrt(n) if n == 30 else 1.96 * s / math.sqrt(n)  # t(0.975, 29) = 2.045
    return m, m - h, m + h


def placement_table() -> pd.DataFrame:
    pl = pd.read_csv(ROOT / "outputs" / "heterogeneous_placements.csv")
    r = P.validate_vs_random(pl, C.SimConfig())
    t = r["table"].copy()
    static = float(t.loc[t.strategy == "static", "peak_load"].iloc[0])
    t["vs_static_pct"] = 100 * (t.peak_load - static) / static
    extra = pd.DataFrame([
        {"strategy": "random (best of 10,000)", "peak_load": r["random_best"]},
        {"strategy": "random (median of 10,000)", "peak_load": r["random_median"]},
        {"strategy": "hindsight demand-aware (uses test demand)", "peak_load": r["hindsight"]},
        {"strategy": "lower bound (speed-proportional, no slot limit)", "peak_load": r["lower_bound"]},
    ])
    return pd.concat([t, extra], ignore_index=True)


def replicate_table() -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS.glob("replicates30*")):
        s = pd.read_csv(d / "summary.csv")
        s = s[s.status == "ok"]
        meta = json.loads((d / "sweep.json").read_text()) if (d / "sweep.json").exists() else {}
        bw = float(s.bandwidth_gb_per_s.iloc[0]) if "bandwidth_gb_per_s" in s else C.SimConfig().bandwidth_gb_per_s
        imp = s.heldout_p95_improvement_pct
        m, lo, hi = ci95(imp)
        mk, mlo, mhi = ci95(s.heldout_makespan_improvement_pct)
        helps = imp > 0
        approve = s.decision.str.lower().isin(["approve", "approved", "migrate"])
        rows.append({
            "run": d.name, "bandwidth_gb_per_s": bw, "n_scenarios": len(s),
            "always_migrate_p95_gain_mean_pct": m, "p95_gain_ci95_low": lo, "p95_gain_ci95_high": hi,
            "makespan_gain_mean_pct": mk, "makespan_ci95_low": mlo, "makespan_ci95_high": mhi,
            "scenarios_where_migration_helps_p95": int(helps.sum()),
            "controller_approved": int(approve.sum()),
            "controller_decision_accuracy_pct": 100 * float((approve == helps).mean()),
            "migration_mb": float(s.migration_mb.iloc[0]), "activation_ms_mean": float(s.activation_ms.mean()),
            "sweep_keys": ",".join(sorted(meta)) if isinstance(meta, dict) else "",
        })
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def latest(prefix: str) -> Path | None:
    c = sorted(RUNS.glob(prefix + "*"), key=lambda p: p.stat().st_mtime)
    return c[-1] if c else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    doc = ["# Testing and evaluation evidence", "",
           f"Generated by `scripts/collect_evidence.py` on {time.strftime('%Y-%m-%d %H:%M %Z')}. "
           "Latency, utilisation and regret numbers are **simulated** with hypothetical hardware; routing is "
           "**measured** on OLMoE; planner and API times are **measured** wall clock on the author's Mac.", ""]

    if not a.skip_tests:
        t = run_tests()
        t.to_csv(OUT / "test_results.csv", index=False)
        g = (t.groupby("category")
             .agg(tests=("test", "size"), negative_cases=("negative", "sum"),
                  passed=("outcome", lambda o: int((o == "pass").sum())),
                  failed=("outcome", lambda o: int(o.isin(["fail", "error"]).sum())),
                  total_s=("duration_s", "sum"))
             .reset_index())
        tot = pd.DataFrame([{"category": "TOTAL", "tests": len(t), "negative_cases": int(t.negative.sum()),
                             "passed": int((t.outcome == "pass").sum()),
                             "failed": int(t.outcome.isin(["fail", "error"]).sum()),
                             "total_s": t.duration_s.sum()}])
        g = pd.concat([g, tot], ignore_index=True)
        g.to_csv(OUT / "test_summary.csv", index=False)
        doc += ["## 1. Automated tests", "",
                f"`python -m unittest discover -s tests -t .` — wall time {t.attrs['wall_s']:.1f} s. "
                "'negative_cases' counts tests whose name marks a failure, rejection or invalid-input case.", "",
                md(g), "", "Slowest five tests:", "",
                md(t.sort_values("duration_s", ascending=False).head(5)[["module", "test", "duration_s"]]), ""]
        print(g.to_string(index=False))

    e2e = sorted(OUT.glob("e2e_*.json"))
    if e2e:
        doc += ["## 2. End-to-end workflow against the running containers", "",
                "`scripts/smoke_e2e.py` drives the full operator workflow over HTTP against the Docker image "
                "(fresh database each time).", ""]
        rows = []
        for f in e2e:
            j = json.loads(f.read_text())
            rows.append({"backend": f.stem.replace("e2e_", ""), "url": j["url"],
                         "checks_passed": f"{sum(c['passed'] for c in j['checks'])}/{len(j['checks'])}",
                         "elapsed_s": j["elapsed_s"]})
        doc += [md(pd.DataFrame(rows)), "", "Checks:", ""]
        doc += [f"* {'PASS' if c['passed'] else 'FAIL'} — {c['check']}" for c in json.loads(e2e[0].read_text())["checks"]]
        doc += [""]

    pt = placement_table()
    pt.to_csv(OUT / "placement_vs_random.csv", index=False)
    doc += ["## 3. Placement quality on held-out demand (Phase 3B)", "",
            "Placement fitted on requests 0, 2, 4; evaluated on 1, 3, 5. Peak load = assignments ÷ assumed GPU speed "
            "(speeds 1.0/1.3/0.8/1.6, 256 slots each). 'beats_random_pct' = share of 10,000 random capacity-feasible "
            "placements with a higher peak.", "", md(pt), ""]

    ext = latest("extended_prompts-")
    if ext is not None:
        hl = pd.read_csv(ext / "headline.csv")
        ss = pd.read_csv(ext / "splits_summary.csv")
        cs = pd.read_csv(ext / "curve_summary.csv")
        for f, df in [("extended_prompts_headline.csv", hl), ("extended_prompts_splits.csv", ss),
                      ("extended_prompts_curve.csv", cs)]:
            df.to_csv(OUT / f, index=False)
        cols = ["evaluation", "train_prompts", "test_prompts", "peak_static", "peak_balanced", "peak_demand_aware",
                "demand_aware_vs_static_pct", "demand_aware_vs_balanced_pct", "demand_aware_beats_random_pct",
                "demand_aware_gap_to_lower_bound_pct"]
        doc += ["### 3.1 The same evaluation on 24 measured prompts", "",
                f"Run `{ext.name}` (`scripts/evaluate_extended_prompts.py`). Routing measured on OLMoE for 24 prompts "
                "(8 per category, 231 tokens); ids 0-5 are the original six re-profiled (routing identical), ids 6-23 "
                "are new. Same simulated cluster as above.", "", md(hl[[c for c in cols if c in hl]]), "",
                "Random stratified 50/50 splits (4 train / 4 test prompts per category). The 2.5-97.5 % range is "
                "the spread over splits, which share prompts, so it is not an independent confidence interval.", "",
                md(ss), "", "Learning curve (training prompts per category; test on the remaining prompts):", "",
                md(cs), ""]

    rt = replicate_table()
    if not rt.empty:
        rt.to_csv(OUT / "replicates_summary.csv", index=False)
        show = rt.drop(columns=["sweep_keys", "run"])
        doc += ["## 4. Migration under 30 seeded Poisson arrival scenarios", "",
                "Each scenario replays the six measured traces with Poisson arrivals (5 req/s, seeds 0–29), 776 "
                "experts (9.76 GB) migrating from static to demand-aware placement. 'p95 gain' is the held-out p95 "
                "improvement of always-migrate vs never-migrate (positive = migration helps). The controller decides "
                "from training requests only; accuracy compares its decision with the held-out outcome. These are "
                "replays of the same six measured traces, not new independent measurements.", "", md(show), ""]

    sc = latest("scalability-")
    if sc:
        fits = pd.read_csv(sc / "growth_fits.csv")
        gap = pd.read_csv(sc / "optimality_gap.csv")
        doc += ["## 5. Planner scalability (measured wall time)", "", f"Run `{sc.name}`. log(ms) = k·log(x) + c.", "",
                md(fits), "",
                f"Optimality: greedy matched the exhaustive optimum on {int((gap.gap_rel.abs() < 1e-9).sum())} of "
                f"{len(gap)} tiny instances (max gap {100 * gap.gap_rel.max():.2f}%).", ""]
    uc = latest("usecases-")
    if uc:
        u = pd.read_csv(uc / "summary.csv")
        cols = ["use_case", "best_forecaster", "best_mean_wape", "site_ewma_popularity_error",
                "global_ewma_popularity_error", "agent_replans", "agent_human_approvals", "objective_agent",
                "objective_static", "objective_oracle", "util_agent_max", "util_static_max", "residency_violations",
                "gpu_hours_released"]
        doc += ["## 6. Six operational use cases (simulated, 47 hourly decisions each)", "", f"Run `{uc.name}`. "
                "Objective: lower is better; oracle = same planner with perfect foresight.", "",
                md(u[[c for c in cols if c in u.columns]], "{:.3f}"), ""]
    bench = OUT / "api_bench.csv"
    if bench.exists():
        doc += ["## 7. Planner API response time (measured)", "",
                "`scripts/bench_api.py -n 50` against the Docker container (SQLite), sequential, same host.", "",
                md(pd.read_csv(bench)), ""]

    (OUT / "EVIDENCE.md").write_text("\n".join(doc))
    print(f"wrote {OUT / 'EVIDENCE.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
