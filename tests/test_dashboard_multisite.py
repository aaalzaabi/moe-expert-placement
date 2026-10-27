"""Multi-site, Forecast & Agent page: view models (dashboard/ms_views.py) and headless page behaviour (AppTest).

Numbered comments refer to the integration checklist (docs/FEEDBACK_RESPONSE.md, "Dashboard tests")."""
import hashlib
import json
import logging
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests.helpers import ROOT, routing

sys.path.insert(0, str(ROOT / "dashboard"))
import ms_views as V  # noqa: E402  (same module object the page imports)

from streamlit.testing.v1 import AppTest  # noqa: E402

from moeplace import agent as A  # noqa: E402
from moeplace import forecast as F  # noqa: E402
from moeplace import multisite as M  # noqa: E402
from moeplace import usecases as U  # noqa: E402

logging.getLogger("streamlit").setLevel(logging.ERROR)
APP = str(ROOT / "dashboard" / "app.py")
PAGE = "app_pages/multisite.py"


def open_ms() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300).run()
    return at.switch_page(PAGE).run()


def no_exc(tc, at):
    tc.assertFalse(at.exception, [str(e.value)[:400] for e in at.exception])


def demand(key):
    return U.USE_CASES[key].demand(routing())[0]


def outputs_digest() -> dict:
    out = {}
    for p in sorted((ROOT / "outputs").rglob("*")):
        if p.is_file() and ".partial-usecases" not in p.parts:
            out[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ------------------------------------------------------------------ registration and structure
class TestRegistration(unittest.TestCase):
    def test_01_dashboard_and_view_models_import(self):
        at = AppTest.from_file(APP, default_timeout=300).run()
        no_exc(self, at)
        import importlib
        importlib.import_module("ms_views")

    def test_02_every_page_file_is_registered_in_navigation(self):
        app = (ROOT / "dashboard" / "app.py").read_text()
        registered = set(re.findall(r'st\.Page\("app_pages/(\w+)\.py"', app))
        files = {p.stem for p in (ROOT / "dashboard" / "app_pages").glob("*.py")}
        self.assertEqual(registered, files)
        self.assertIn("multisite", registered)

    def test_03_all_six_use_cases_selectable(self):
        at = open_ms()
        no_exc(self, at)
        self.assertEqual(list(at.selectbox(key="ms_uc").options),
                         [f"{k.split('_')[0].upper()} · {U.USE_CASES[k].title}" for k in V.UC_ORDER])
        self.assertEqual(len(at.tabs), 6)

    def test_17_widget_keys_unique_per_page_and_namespaced(self):
        for path in (ROOT / "dashboard" / "app_pages").glob("*.py"):
            keys = re.findall(r'key="([^"]+)"', path.read_text())
            dupes = {k for k in keys if keys.count(k) > 1}
            self.assertFalse(dupes, f"{path.name}: duplicate widget keys {dupes}")
        ms_keys = re.findall(r'key="([^"]+)"', (ROOT / "dashboard" / "app_pages" / "multisite.py").read_text())
        self.assertTrue(ms_keys and all(k.startswith("ms_") for k in ms_keys), ms_keys)


# ------------------------------------------------------------------ scenario profile
class TestScenario(unittest.TestCase):
    def test_04_profile_reflects_selected_use_case(self):
        p2 = V.scenario_profile("uc2_sovereign_residency")
        self.assertEqual(p2["residency_sites"], ["abudhabi"])
        self.assertIn("0.7", p2["objective"])
        p3 = V.scenario_profile("uc3_heterogeneous_cluster")
        self.assertEqual(len(p3["sites"]), 1)
        self.assertIn("bottleneck utilisation", p3["objective"])
        p5 = V.scenario_profile("uc5_device_failure")
        self.assertTrue(any("abudhabi-g3" in e for e in p5["events"]))

    def test_04_page_shows_parameters_of_selected_scenario(self):
        at = open_ms()
        at.selectbox(key="ms_uc").set_value("uc2_sovereign_residency").run()
        no_exc(self, at)
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("abudhabi", text)
        self.assertIn("capacity+memory+residency", text)


# ------------------------------------------------------------------ forecasting
class TestForecastViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = "uc1_multi_region"
        cls.Y = demand(cls.key)
        cls.fitted = V.fit_models(cls.Y, cls.key)

    def test_05_every_forecasting_method_produces_validated_output(self):
        self.assertEqual(set(self.fitted), set(F.MODELS))
        S, E = self.Y.shape[1:]
        for m, f in self.fitted.items():
            with self.subTest(model=m):
                V.validate_forecast(f.predict(U.USE_CASES[self.key].train_end + 8, 4), 4, S, E)

    def test_05_validation_rejects_bad_forecasts(self):
        good = np.ones((4, 3, 1024))
        for bad in (np.ones((3, 3, 1024)), np.full((4, 3, 1024), np.nan), -good):
            with self.assertRaises(ValueError):
                V.validate_forecast(bad, 4, 3, 1024)

    def test_predicted_vs_actual_and_error_by_site(self):
        df = V.predicted_vs_actual(self.Y, self.fitted, self.key, site=0)
        self.assertTrue({"window", "actual", *F.MODELS} <= set(df.columns))
        ebs = V.error_by_site(self.Y, self.fitted, self.key)
        self.assertEqual(len(ebs), len(F.MODELS) * 3)
        g = ebs.groupby("model").popularity_error.mean()
        self.assertLess(g["site_ewma"], g["global_ewma"])  # site-aware beats global popularity

    def test_06_charts_handle_missing_or_empty_data(self):
        empty = V.predicted_vs_actual(self.Y, self.fitted, self.key, 0, eval_windows=0)
        self.assertTrue(empty.empty)
        self.assertTrue(V.error_by_site(self.Y, self.fitted, self.key, eval_windows=0).empty)
        for fig in (V.fig_predicted_vs_actual(None, F.MODELS, "x"), V.fig_predicted_vs_actual(empty, F.MODELS, "x"),
                    V.fig_heatmap(np.array([]), "x"), V.fig_regret_scatter(pd.DataFrame(), 0.1, 0.02)):
            self.assertTrue(fig.layout.annotations, "empty input must produce an explanatory empty figure")
        self.assertTrue(V.classify(None).empty)
        self.assertTrue(V.capacity_violations(None, 0.7).empty)
        a, r = V.drift_alerts(None)
        self.assertTrue(a.empty and r.empty)


# ------------------------------------------------------------------ regret
class TestRegretViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        uc = U.USE_CASES["uc1_multi_region"]
        cls.rows = F.rolling_evaluation(demand(uc.key), uc.topology, U.spec(), uc.train_end, models=("site_ewma", "naive_last"),
                                        eval_end=uc.train_end + 32, **uc.plan_kw())

    def test_07_regret_is_labelled_hindsight_evaluation_only(self):
        cl = V.classify(self.rows)
        self.assertTrue((cl.regret_basis == V.HINDSIGHT).all())
        self.assertIn("hindsight", V.fig_regret_scatter(self.rows).layout.title.text)
        self.assertEqual(set(cl.quadrant) <= set(V.QUADRANTS.values()), True)

    def test_07_agent_decisions_ignore_future_demand(self):
        key = "uc1_multi_region"
        Y = demand(key)
        a = V.run_agent_cycles(key, Y, 3, start_offset=8)
        Y2 = Y.copy()
        Y2[a["last_t"] + 1:] *= 5.0          # change only demand that arrives after the last decision
        b = V.run_agent_cycles(key, Y2, 3, start_offset=8)
        cols = ["t", "action", "new_copies", "J_new_forecast", "human_decision"]
        pd.testing.assert_frame_equal(a["decisions"][cols], b["decisions"][cols])
        self.assertFalse(np.allclose(a["decisions"].objective_actual_agent, b["decisions"].objective_actual_agent))

    def test_capacity_violation_definitions_and_drift(self):
        cv = V.capacity_violations(self.rows, 0.7)
        self.assertEqual(set(cv.columns), {"model", "decision_points", "overload (util >= 1)", "above cap"})
        alerts, recs = V.drift_alerts(self.rows, 0.02)
        self.assertEqual(set(recs.model), {"site_ewma", "naive_last"})
        self.assertTrue(set(recs.recommendation) <= {"no action", "refit on recent data when alerts occur (the agent does this automatically)",
                                                     "retrain or replace: decisions persistently worse than hindsight"})


# ------------------------------------------------------------------ agent safety
class ViolatingAgent(A.PlacementAgent):
    """After the initial plan, proposes plans that serve the residency site's demand from abroad."""
    def _propose(self, yhat, topo):
        plan = super()._propose(yhat, topo)
        if getattr(self, "plan", None) is None:
            return plan
        r = topo.residency_sites()[0]
        foreign = int(np.flatnonzero(topo.site_of != r)[0])
        plan.serve[r, :50] = foreign
        plan.hosts[:50, foreign] = True
        return plan


class TestAgentSafety(unittest.TestCase):
    def test_08_act_tool_requires_authoriser_token(self):
        res = V.run_agent_cycles("uc1_multi_region", demand("uc1_multi_region"), 1)
        ag = res["agent"]
        with self.assertRaises(A.PermissionDenied):
            ag.call("apply_placement", ag.plan, ag.topo, ())
        self.assertTrue(set(res["trace"].permission) <= set(A.PERMISSIONS))

    def test_09_unsafe_proposals_rejected(self):
        auth = A.Authoriser(A.AgentConfig(), horizon_s=3600)
        plan = M.Plan(np.ones((1, 1), bool), np.zeros(1, int), np.zeros((1, 1), int))
        ok_mig = {"bytes": 1e9, "est_time_s": 10.0, "new_copies": 1}
        slow = {"bytes": 1e9, "est_time_s": 999.0, "new_copies": 1}
        cases = {
            "overload": A.Proposal(0, "rebalance", plan, None, 100.0, 50.0, ok_mig, False, (), 1.2),
            "too slow": A.Proposal(0, "rebalance", plan, None, 100.0, 50.0, slow, False, (), 0.5),
            "no gain": A.Proposal(0, "rebalance", plan, None, 100.0, 99.9, ok_mig, False, (), 0.5),
            "mandatory but overloaded": A.Proposal(0, "restore", plan, None, float("inf"), 50.0, ok_mig, True, (), 1.5),
        }
        for name, p in cases.items():
            with self.subTest(case=name):
                self.assertFalse(auth.check(p, False, 0.0)[0])
        mandatory = A.Proposal(0, "restore", plan, None, float("inf"), 50.0, ok_mig, True, (), 0.5)
        self.assertTrue(auth.check(mandatory, False, 0.0)[0])
        self.assertFalse(auth.check(mandatory, False, 5.0)[0])  # even a mandatory restore cannot violate residency

    def test_10_residency_violations_are_blocked(self):
        key = "uc2_sovereign_residency"
        uc = U.USE_CASES[key]
        Y = demand(key)
        ag = ViolatingAgent(Y, uc.topology, U.spec(), A.AgentConfig(), plan_kw=uc.plan_kw(),
                            approver=lambda p: (True, "auto"))
        dec, _ = ag.run(uc.train_end + 8, uc.train_end + 8 + 4 * 4 + 1)
        self.assertTrue((dec.action == "keep").all(), dec[["t", "action", "reasons"]])
        self.assertTrue(dec.reasons.str.contains("FAIL: constraints satisfied").all())
        self.assertEqual(dec.residency_violations_actual.sum(), 0.0)
        self.assertGreater(len(V.blocked_reasons(dec)), 0)

    def test_11_human_approval_required_and_respected(self):
        key = "uc2_sovereign_residency"
        Y = demand(key)
        res = V.run_agent_cycles(key, Y, 4, start_offset=8)
        self.assertTrue(res["pending"], "residency-site change must wait for a human")
        self.assertEqual(res["decisions"].iloc[-1].action, "keep")
        p = res["pending"][0]
        approved = V.run_agent_cycles(key, Y, 4, start_offset=8, decisions={f"{p['t']}:{p['proposal'].reason}": "approve"})
        self.assertNotEqual(approved["decisions"].iloc[-1].action, "keep")
        denied = V.run_agent_cycles(key, Y, 4, start_offset=8, decisions={f"{p['t']}:{p['proposal'].reason}": "deny"})
        self.assertEqual(denied["decisions"].iloc[-1].human_decision, "denied")
        self.assertEqual(denied["decisions"].iloc[-1].action, "keep")

    def test_12_audit_events_recorded(self):
        res = V.run_agent_cycles("uc2_sovereign_residency", demand("uc2_sovereign_residency"), 4, start_offset=8)
        audit = res["audit"]
        self.assertTrue(set(res["decisions"].t) <= set(audit.t))
        self.assertTrue(audit.event.str.contains("human approval pending").any())


# ------------------------------------------------------------------ scalability and labels
class TestScalabilityViews(unittest.TestCase):
    def make_run(self, root: Path, machine: str) -> Path:
        d = root / "scalability-test01"
        d.mkdir(parents=True)
        pd.DataFrame([
            {"factor": "proposal_point", "experts": 16, "sites": 3, "devices": 3, "replica_frac": 0.25, "constraints": "x",
             "status": "ok", "planner_ms": 0.4, "planner_p95_ms": 0.5, "latency_ms": 700.0, "J": 0.6},
            {"factor": "experts", "experts": 16, "sites": 3, "devices": 12, "replica_frac": 0.25, "constraints": "x",
             "status": "ok", "planner_ms": 0.5, "planner_p95_ms": 0.6, "latency_ms": 650.0, "J": 0.6},
            {"factor": "experts", "experts": 1024, "sites": 3, "devices": 12, "replica_frac": 0.25, "constraints": "x",
             "status": "ok", "planner_ms": 20.0, "planner_p95_ms": 25.0, "latency_ms": 600.0, "J": 0.65},
        ]).to_csv(d / "scalability.csv", index=False)
        (d / "metadata.json").write_text(json.dumps({"created_utc": "2026-10-22T00:00:00+00:00",
                                                     "params": {"machine": machine}}))
        return d

    def test_13_scalability_results_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.make_run(Path(tmp), "macOS-15.7-arm64")
            self.assertEqual(V.latest_run("scalability", Path(tmp)), d)
            df, msg = V.load_csv(d, "scalability.csv", "run_scalability.py")
            self.assertIsNone(msg)
            sv = V.scalability_views(df, V.run_meta(d), "experts")
            self.assertEqual(list(sv["table"].experts), [16, 1024])
            self.assertTrue(sv["table"].meets_2s_target.all())
            self.assertAlmostEqual(V.proposal_point(df)["planner_ms"], 0.4)

    def test_14_planner_timing_and_simulated_metrics_never_mislabeled(self):
        linux = V.platform_label({"params": {"machine": "Linux-6.8.0-138-generic-aarch64-with-glibc2.35"}})
        self.assertIn("not a macOS", linux)
        self.assertNotIn("measured on macOS", linux)
        self.assertIn("measured on macOS", V.platform_label({"params": {"machine": "macOS-15.7.9-arm64-arm-64bit"}}))
        sv = V.scalability_views(None, {"params": {"machine": "Linux-x"}}, "experts")
        self.assertIn("Planner wall time", sv["planner_title"])
        self.assertIn("measured", sv["planner_title"])
        self.assertTrue(sv["latency_title"].startswith("Simulated"))
        self.assertTrue(sv["cost_title"].startswith("Simulated"))
        self.assertNotIn("measured", sv["latency_title"] + sv["cost_title"])

    def test_14_page_states_where_timings_were_measured(self):
        at = open_ms()
        no_exc(self, at)
        meta = V.run_meta(V.latest_run("scalability"))
        if meta:
            self.assertTrue(any(V.platform_label(meta) in c.value for c in at.caption))


# ------------------------------------------------------------------ page behaviour (AppTest)
class TestPageBehaviour(unittest.TestCase):
    def test_15_16_page_and_interactions_never_load_olmoe(self):
        at = open_ms()
        at.button(key="ms_ag_step").click().run()
        no_exc(self, at)
        self.assertNotIn("torch", sys.modules)

    def test_18_session_state_isolated_between_scenarios(self):
        at = open_ms()
        at.button(key="ms_ag_step").click().run()
        at.selectbox(key="ms_uc").set_value("uc5_device_failure").run()
        no_exc(self, at)
        states = at.session_state["ms_state"]
        self.assertEqual(states["uc1_multi_region"]["cycles"], 1)
        self.assertEqual(states["uc5_device_failure"]["cycles"], 0)
        self.assertTrue(any("Run next cycle" in i.value for i in at.info))

    def test_11_approve_button_applies_change_in_session_only(self):
        before = outputs_digest()
        at = open_ms()
        at.selectbox(key="ms_uc").set_value("uc2_sovereign_residency").run()
        for _ in range(4):
            at.button(key="ms_ag_step").click().run()
        no_exc(self, at)
        self.assertTrue(any("Human approval pending" in w.value for w in at.warning))
        at.button(key="ms_ag_approve").click().run()
        no_exc(self, at)
        self.assertTrue(any("Final action:** rebalance" in s.value for s in at.success))
        self.assertEqual(before, outputs_digest())

    def test_19_missing_saved_results_show_messages_not_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = V.RUNS_DIR
            V.RUNS_DIR = Path(tmp)
            try:
                at = open_ms()
                no_exc(self, at)
                self.assertTrue(any("No saved run found" in i.value for i in at.info))
            finally:
                V.RUNS_DIR = old
            bad = Path(tmp) / "usecases-x"
            bad.mkdir()
            (bad / "f.csv").write_text("")
            df, msg = V.load_csv(bad, "f.csv", "run_usecases.py")
            self.assertIsNone(df)
            self.assertIn("could not be read", msg)
            df, msg = V.load_csv(bad, "missing.csv", "run_usecases.py")
            self.assertIn("missing", msg)

    def test_20_no_dashboard_action_writes_experiment_outputs(self):
        before = outputs_digest()
        at = open_ms()
        at.selectbox(key="ms_uc").set_value("uc3_heterogeneous_cluster").run()
        at.button(key="ms_run").click().run()
        at.button(key="ms_ag_step").click().run()
        at.button(key="ms_ag_reset").click().run()
        no_exc(self, at)
        self.assertIsNotNone(at.session_state["ms_state"]["uc3_heterogeneous_cluster"]["run"])
        self.assertEqual(before, outputs_digest())


class TestGovernanceViews(unittest.TestCase):
    def test_governance_from_single_source(self):
        g = V.governance_views("uc2_sovereign_residency")
        self.assertTrue(g["found"])
        self.assertEqual(len(g["compliance"]), 1)
        self.assertIn("UC2", g["compliance"].iloc[0, 0])
        self.assertGreaterEqual(len(g["risks"]), 9)
        rules = V.approval_rules()
        self.assertTrue((rules.type == "human approval").sum() >= 3)


if __name__ == "__main__":
    unittest.main()
