"""Planner API + database: unit, integration, end-to-end and failure-case tests.

Every test uses its own temporary SQLite database through the real HTTP layer (FastAPI TestClient), so routes,
validation, the workflow's transactions and the database constraints are exercised together.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from moeplace import config as C
from moeplace.config import ConfigError
from moeplace.paths import PROJECT_ROOT
from service import planning as PL
from service.app import create_app
from service.db import PlanRow
from service.schemas import Scenario
from service.settings import Settings

SCENARIO = json.loads((PROJECT_ROOT / "examples" / "scenario.json").read_text())


def scenario(**changes) -> dict:
    sc = copy.deepcopy(SCENARIO)
    sc.update(changes)
    return sc


class ServiceCase(unittest.TestCase):
    settings_kw: dict = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_url = f"sqlite:///{Path(self.tmp.name) / 'planner.db'}"
        self.client = self.make_client()

    def tearDown(self):
        self.client.app.state.engine.dispose()
        self.tmp.cleanup()

    def make_client(self) -> TestClient:
        settings = Settings(database_url=self.db_url, log_level="WARNING", **self.settings_kw).validate()
        return TestClient(create_app(settings))

    def propose(self, sc=None, **headers) -> dict:
        r = self.client.post("/proposals", json=sc or SCENARIO, headers=headers)
        self.assertIn(r.status_code, (200, 201), r.text)
        return r.json()

    def bootstrap(self) -> dict:
        """Create and apply the initial plan."""
        p = self.propose()
        self.assertEqual(p["status"], "ready", p["checks"])
        self.assertEqual(self.client.post(f"/proposals/{p['id']}/apply").status_code, 200)
        return p

    def events(self, proposal_id=None) -> list[str]:
        params = {} if proposal_id is None else {"proposal_id": proposal_id}
        return [e["event"] for e in self.client.get("/audit", params=params).json()]


# ------------------------------------------------------------------------------------------- unit
class TestSettings(unittest.TestCase):
    def test_defaults_match_the_operations_agent(self):
        s = Settings.from_env({})
        self.assertTrue(s.database_url.startswith("sqlite:///"))
        self.assertEqual((s.approval_gb, s.min_gain_rel, s.max_migration_fraction), (50.0, 0.02, 0.05))

    def test_environment_overrides(self):
        s = Settings.from_env({"MOE_DATABASE_URL": "sqlite:///x.db", "MOE_APPROVAL_GB": "5", "MOE_LOG_LEVEL": "debug"})
        self.assertEqual((s.database_url, s.approval_gb, s.log_level), ("sqlite:///x.db", 5.0, "DEBUG"))

    def test_invalid_values_are_rejected(self):
        for env in ({"MOE_APPROVAL_GB": "lots"}, {"MOE_MIN_GAIN": "1.5"}, {"MOE_LOG_LEVEL": "LOUD"}):
            with self.subTest(env=env), self.assertRaises(ConfigError):
                Settings.from_env(env)


class TestPlanningAdapter(unittest.TestCase):
    def test_demand_volume_follows_the_measured_routing(self):
        sc = Scenario.model_validate(SCENARIO)
        Y = PL.demand(sc)
        self.assertEqual(Y.shape, (3, C.TOTAL_EXPERT_SLOTS))
        for s, site in enumerate(sc.sites):  # every token goes to 8 experts in each of 16 layers
            self.assertAlmostEqual(Y[s].sum(), site.requests_per_window * sc.tokens_per_request * C.TOP_K * C.NUM_LAYERS,
                                   delta=1e-6 * Y[s].sum())

    def test_unknown_category_is_a_scenario_error(self):
        sc = scenario()
        sc["sites"][0]["mix"] = {"poetry": 1.0}
        with self.assertRaises(PL.ScenarioError):
            PL.demand(Scenario.model_validate(sc))

    def test_assignment_round_trip_and_reindex(self):
        c = PL.compute(Scenario.model_validate(SCENARIO))
        back = PL.from_assignment(json.loads(json.dumps(PL.to_assignment(c.plan))), c.topo)
        np.testing.assert_array_equal(back.hosts, c.plan.hosts)
        np.testing.assert_array_equal(back.serve, c.plan.serve)
        same = PL.reindex(back, c.topo, c.topo)
        self.assertAlmostEqual(PL.M.evaluate(same, c.Y, c.topo)["J"], c.metrics["J"])


# ------------------------------------------------------------------------------------------- integration / e2e
class TestHealthAndPlan(ServiceCase):
    def test_health_reports_database_and_measured_data(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["database"], "ok")
        self.assertTrue(r.json()["routing_profiles"].startswith("ok"))
        self.assertIn("X-Request-ID", r.headers)

    def test_health_is_degraded_without_routing_data(self):
        with patch("service.app.measured_profiles", side_effect=FileNotFoundError("workload_routing.csv")):
            r = self.client.get("/health")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["status"], "degraded")

    def test_plan_is_stateless_and_places_every_expert(self):
        r = self.client.post("/plan", json=SCENARIO)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(sum(d["experts"] for d in body["devices"]), C.TOTAL_EXPERT_SLOTS)  # no replicas requested
        self.assertLess(body["metrics"]["J"], 1)
        self.assertEqual(self.client.get("/proposals").json(), [])
        self.assertEqual(self.client.get("/plans/active").status_code, 404)

    def test_api_plan_matches_the_core_planner(self):
        sc = Scenario.model_validate(SCENARIO)
        core = PL.M.plan_sites(PL.demand(sc), PL.topology(sc))
        api = self.client.post("/plan", json=SCENARIO).json()
        self.assertAlmostEqual(api["metrics"]["J"], PL.M.evaluate(core, PL.demand(sc), PL.topology(sc))["J"])


class TestProposalWorkflow(ServiceCase):
    def test_initial_proposal_is_applied_end_to_end(self):
        p = self.propose()
        self.assertEqual((p["status"], p["reason"], p["base_plan_id"]), ("ready", "initial", None))
        # the first plan loads every expert from checkpoint storage: 1,024 x 12.58 MB
        self.assertAlmostEqual(p["migration"]["gb"], C.TOTAL_EXPERT_SLOTS * C.CHECKPOINT_EXPERT_BYTES / C.GB, places=6)
        applied = self.client.post(f"/proposals/{p['id']}/apply").json()
        self.assertEqual(applied["status"], "applied")
        active = self.client.get("/plans/active").json()
        self.assertEqual((active["id"], active["status"]), (p["plan_id"], "active"))
        self.assertEqual(self.events(p["id"]), ["proposal_created", "authoriser_approved", "plan_activated"])

    def test_identical_demand_is_rejected_as_no_gain(self):
        self.bootstrap()
        p = self.propose()
        self.assertEqual(p["status"], "rejected")
        failed = [c["check"] for c in p["checks"] if not c["passed"]]
        self.assertEqual(len(failed), 1)
        self.assertIn("gain", failed[0])
        self.assertEqual(self.client.post(f"/proposals/{p['id']}/apply").status_code, 409)

    def test_slower_device_is_scored_on_current_hardware_and_rebalanced(self):
        first = self.bootstrap()
        sc = scenario()
        sc["sites"][0]["devices"][0]["speed"] = 0.3
        p = self.propose(sc)
        self.assertEqual((p["status"], p["reason"]), ("ready", "rebalance"), p["checks"])
        self.assertGreater(p["objective_current"], 1.0)  # the old plan overloads the slowed device
        self.assertLess(p["objective_new"], p["objective_current"])
        self.assertEqual(p["base_plan_id"], first["plan_id"])
        self.client.post(f"/proposals/{p['id']}/apply")
        self.assertEqual(self.client.get(f"/plans/{first['plan_id']}").json()["status"], "superseded")
        self.assertEqual(self.client.get("/plans/active").json()["id"], p["plan_id"])

    def test_lost_device_forces_a_restore_from_storage(self):
        self.bootstrap()
        sc = scenario()
        del sc["sites"][1]["devices"][3]
        p = self.propose(sc)
        self.assertEqual((p["status"], p["reason"], p["objective_current"]), ("ready", "restore", None))
        self.assertGreater(p["migration"]["storage_bytes"], 0)

    def test_stale_proposal_cannot_be_applied(self):
        self.bootstrap()
        sc = scenario()
        sc["sites"][0]["devices"][0]["speed"] = 0.3
        a, b = self.propose(sc), self.propose(sc)  # both assessed against the same active plan
        self.assertEqual(self.client.post(f"/proposals/{a['id']}/apply").status_code, 200)
        r = self.client.post(f"/proposals/{b['id']}/apply")
        self.assertEqual(r.status_code, 409)
        self.assertIn("create a new proposal", r.json()["detail"])
        self.assertEqual(self.client.get(f"/proposals/{b['id']}").json()["status"], "stale")
        self.assertIn("proposal_stale", self.events(b["id"]))

    def test_idempotency_key_returns_the_original_proposal(self):
        r1 = self.client.post("/proposals", json=SCENARIO, headers={"Idempotency-Key": "abc"})
        r2 = self.client.post("/proposals", json=SCENARIO, headers={"Idempotency-Key": "abc"})
        self.assertEqual((r1.status_code, r2.status_code), (201, 200))
        self.assertEqual(r1.json()["id"], r2.json()["id"])
        self.assertEqual(len(self.client.get("/proposals").json()), 1)
        r3 = self.client.post("/proposals", json=scenario(tokens_per_request=500), headers={"Idempotency-Key": "abc"})
        self.assertEqual(r3.status_code, 409)

    def test_state_survives_a_restart(self):
        p = self.bootstrap()
        self.client.app.state.engine.dispose()
        self.client = self.make_client()  # new process, same database
        self.assertEqual(self.client.get("/plans/active").json()["id"], p["plan_id"])
        self.assertEqual(self.propose()["status"], "rejected")  # still compared with the persisted plan

    def test_documented_schema_is_up_to_date(self):
        import subprocess
        import sys
        r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "export_schema.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_single_active_plan_is_enforced_by_the_database(self):
        p = self.bootstrap()
        with self.client.app.state.sessions() as s:
            row = s.get(PlanRow, p["plan_id"])
            s.add(PlanRow(status="active", scenario=row.scenario, assignment=row.assignment, metrics=row.metrics,
                          devices=row.devices))
            with self.assertRaises(IntegrityError):
                s.commit()


class TestHumanApproval(ServiceCase):
    settings_kw = {"approval_gb": 1.0}  # the 12.9 GB initial load now needs a human

    def test_approve_then_apply(self):
        p = self.propose()
        self.assertEqual((p["status"], p["needs_human"]), ("awaiting_approval", True))
        self.assertEqual(self.client.post(f"/proposals/{p['id']}/apply").status_code, 409)  # not before approval
        r = self.client.post(f"/proposals/{p['id']}/approve", json={"approver": "ops-lead", "comment": "window ok"})
        self.assertEqual(r.json()["status"], "ready")
        self.assertEqual(self.client.post(f"/proposals/{p['id']}/apply").json()["status"], "applied")
        audit = self.client.get("/audit", params={"proposal_id": p["id"]}).json()
        self.assertEqual([e["event"] for e in audit],
                         ["proposal_created", "approval_requested", "human_approved", "plan_activated"])
        self.assertEqual(audit[2]["actor"], "human:ops-lead")

    def test_deny_is_terminal(self):
        p = self.propose()
        self.assertEqual(self.client.post(f"/proposals/{p['id']}/deny", json={"approver": "ops-lead"}).json()["status"],
                         "denied")
        self.assertEqual(self.client.post(f"/proposals/{p['id']}/approve", json={"approver": "x"}).status_code, 409)
        self.assertEqual(self.client.post(f"/proposals/{p['id']}/apply").status_code, 409)
        self.assertEqual(self.client.get("/plans/active").status_code, 404)


class TestResidency(ServiceCase):
    def test_plan_touching_a_residency_site_needs_a_human(self):
        sc = scenario()
        sc["sites"][0]["residency"] = True
        p = self.propose(sc)
        self.assertEqual(p["status"], "awaiting_approval")
        self.assertTrue(p["migration"]["residency_site_touched"])
        plan = self.client.get(f"/plans/{p['plan_id']}").json()
        self.assertEqual(plan["metrics"]["residency_violation_assignments"], 0)


class TestInvalidRequests(ServiceCase):
    def assertStatus(self, method, path, code, **kw):
        r = getattr(self.client, method)(path, **kw)
        self.assertEqual(r.status_code, code, r.text)
        self.assertIn("detail", r.json())
        return r.json()["detail"]

    def test_schema_violations_return_422(self):
        bad = []
        sc = scenario(); sc["sites"][0]["devices"][0]["speed"] = 0; bad.append(sc)
        sc = scenario(); sc["sites"][0]["devices"][1]["name"] = sc["sites"][0]["devices"][0]["name"]; bad.append(sc)
        sc = scenario(); sc["sites"][1]["name"] = sc["sites"][0]["name"]; bad.append(sc)
        sc = scenario(); sc["sites"][0]["mix"] = {"ai": -1}; bad.append(sc)
        sc = scenario(); sc["sites"][0]["devices"] = []; bad.append(sc)
        bad += [scenario(sites=[]), scenario(replica_frac=2), scenario(unexpected_field=1), {"sites": "x"}]
        for sc in bad:
            with self.subTest(sc=str(sc)[:80]):
                self.assertStatus("post", "/plan", 422, json=sc)

    def test_unservable_scenarios_return_422_with_a_reason(self):
        sc = scenario()
        for site in sc["sites"]:
            for d in site["devices"]:
                d["slots"] = 10  # 12 devices x 10 slots < 1,024 experts
        self.assertIn("slot", self.assertStatus("post", "/plan", 422, json=sc))
        sc = scenario(); sc["sites"][0]["mix"] = {"poetry": 1}
        self.assertIn("unknown categories", self.assertStatus("post", "/proposals", 422, json=sc))

    def test_changing_the_sites_of_the_active_plan_is_rejected(self):
        self.bootstrap()
        sc = scenario(sites=SCENARIO["sites"][:2])
        self.assertIn("differ from the active plan", self.assertStatus("post", "/proposals", 422, json=sc))

    def test_unknown_ids_and_bad_decisions(self):
        self.assertStatus("get", "/proposals/999", 404)
        self.assertStatus("post", "/proposals/999/apply", 404)
        self.assertStatus("get", "/plans/999", 404)
        p = self.propose()
        self.assertStatus("post", f"/proposals/{p['id']}/approve", 409, json={"approver": "x"})  # it is 'ready'
        self.assertStatus("post", f"/proposals/{p['id']}/approve", 422, json={"approver": ""})
        self.assertStatus("get", "/proposals", 422, params={"status": "unknown"})

    def test_failed_request_writes_nothing(self):
        sc = scenario(); sc["sites"][0]["mix"] = {"poetry": 1}
        self.client.post("/proposals", json=sc)
        self.assertEqual(self.client.get("/proposals").json(), [])
        self.assertEqual(self.client.get("/audit").json(), [])


if __name__ == "__main__":
    unittest.main()
