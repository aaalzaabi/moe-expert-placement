"""Operations agent: permissions, authoriser, failure handling, residency, consolidation; use-case registry;
scalability experiment."""
import unittest

import numpy as np

from tests.helpers import routing
from moeplace import agent as A
from moeplace import multisite as M
from moeplace import scalability as SC
from moeplace import usecases as U


def run_agent(key, start_offset=0, windows=16, **cfg):
    uc = U.USE_CASES[key]
    Y, _ = uc.demand(routing())
    ag = A.PlacementAgent(Y, uc.topology, U.spec(), A.AgentConfig(consolidation=uc.consolidation, **cfg),
                          plan_kw=uc.plan_kw())
    start = uc.train_end + start_offset
    dec, trace = ag.run(start, start + windows, events=uc.events)
    return uc, ag, dec, trace


class TestUseCases(unittest.TestCase):
    def test_six_use_cases_with_users_and_governance(self):
        self.assertEqual(len(U.USE_CASES), 6)
        for uc in U.USE_CASES.values():
            self.assertTrue(uc.users and uc.governance and uc.kpis and uc.question)
            uc.topology.validate()


class TestAgent(unittest.TestCase):
    def test_act_tool_requires_authoriser_token(self):
        uc, ag, _, _ = run_agent("uc1_multi_region", windows=5)
        with self.assertRaises(A.PermissionDenied):
            ag.call("apply_placement", ag.plan, ag.topo, ())

    def test_every_tool_call_is_traced_with_permission(self):
        _, ag, _, trace = run_agent("uc1_multi_region", windows=9)
        self.assertTrue(set(trace.permission) <= set(A.PERMISSIONS))
        self.assertIn("propose_placement", set(trace.tool))

    def test_device_failure_restores_feasible_plan(self):
        uc = U.USE_CASES["uc5_device_failure"]
        t_fail = uc.events[0][0]
        offset = t_fail - uc.train_end - 4
        _, ag, dec, _ = run_agent("uc5_device_failure", start_offset=offset, windows=13)
        self.assertIn("restore", set(dec.action))
        self.assertNotIn(uc.events[0][2], [d.name for d in ag.topo.devices])
        after = dec[dec.t >= t_fail]
        self.assertTrue((after.devices_active == uc.topology.n_devices - 1).all())

    def test_residency_never_violated_and_changes_need_approval(self):
        _, ag, dec, _ = run_agent("uc2_sovereign_residency", windows=24)
        self.assertEqual(dec.residency_violations_actual.sum(), 0.0)
        acted = dec[dec.action != "keep"]
        if len(acted):
            self.assertTrue(acted.needs_human.fillna(False).any() or (acted.approved_by == "authoriser").all())

    def test_denied_approval_keeps_current_plan(self):
        uc = U.USE_CASES["uc6_offpeak_consolidation"]
        Y, _ = uc.demand(routing())
        ag = A.PlacementAgent(Y, uc.topology, U.spec(), A.AgentConfig(consolidation=True, approval_gb=0.0),
                              plan_kw=uc.plan_kw(), approver=lambda p: (False, "denied in test"))
        dec, _ = ag.run(uc.train_end, uc.train_end + 12)
        self.assertTrue((dec.devices_off_now == 0).all())

    def test_authoriser_rules(self):
        auth = A.Authoriser(A.AgentConfig(), horizon_s=3600)
        plan = M.Plan(np.ones((1, 1), bool), np.zeros(1, int), np.zeros((1, 1), int))
        small = {"bytes": 1e9, "est_time_s": 10.0, "new_copies": 1}
        ok, human, _ = auth.check(A.Proposal(0, "rebalance", plan, None, 100.0, 90.0, small, False, (), 0.5), False, 0.0)
        self.assertTrue(ok and not human)
        ok, _, _ = auth.check(A.Proposal(0, "rebalance", plan, None, 100.0, 99.5, small, False, (), 0.5), False, 0.0)
        self.assertFalse(ok)  # gain below 2 %
        ok, _, _ = auth.check(A.Proposal(0, "rebalance", plan, None, 100.0, 50.0, small, False, (), 0.5), False, 5.0)
        self.assertFalse(ok)  # residency violation
        big = {"bytes": 80e9, "est_time_s": 10.0, "new_copies": 9}
        ok, human, _ = auth.check(A.Proposal(0, "rebalance", plan, None, 100.0, 50.0, big, False, (), 0.5), False, 0.0)
        self.assertTrue(ok and human)


class TestScalability(unittest.TestCase):
    def test_small_grid_and_fits(self):
        grid = {"experts": [16, 64, 256], "sites": [1, 2, 3]}
        df = SC.run(routing(), repeats=1, seeds=(0,), grid=grid, log=lambda *a: None)
        self.assertTrue((df.status == "ok").all())
        self.assertTrue({"planner_ms", "candidates", "J", "latency_ms"} <= set(df.columns))
        fits = SC.growth_fits(df)
        self.assertEqual(set(fits.factor), {"experts", "sites"})
        prop = df[df.factor == "proposal_point"].iloc[0]
        self.assertLess(prop.planner_ms, 2000)  # proposal target: 16 experts, 3 sites, < 2 s


if __name__ == "__main__":
    unittest.main()
