"""Multi-site planner, cost model, constraints, replicas and migration cost."""
import unittest

import numpy as np

from tests.helpers import routing
from moeplace import config as C
from moeplace import data as D
from moeplace import multisite as M
from moeplace import placement as P
from moeplace import usecases as U


def tiny_topo(residency=False, slots=4, speeds=(1.0, 1.0)):
    sites = (M.Site("a", residency=residency), M.Site("b"))
    devs = (M.Device("a0", 0, speeds[0], slots), M.Device("b0", 1, speeds[1], slots))
    cost = M.CostModel(window_s=1.0, assignments_per_s=100.0, activation_bytes=1000, inter_site_gb_per_s=1e-6,
                       inter_site_rtt_ms=10.0)
    return M.Topology(sites, devs, cost).validate()


class TestPlanner(unittest.TestCase):
    def test_single_site_equals_demand_aware(self):
        d = D.demand(routing(), (0, 2, 4)).to_numpy(float)
        topo = M.Topology((M.Site("x"),), tuple(M.Device(f"g{i}", 0, s, 256) for i, s in enumerate(C.DEFAULT.gpu_speeds)))
        plan = M.plan_sites(d[None, :], topo, constraints=M.Constraints(memory=False, residency=False))
        ref = P.assign(d, "demand_aware", C.DEFAULT.gpu_speeds, 256)
        self.assertTrue((plan.primary == ref).all())

    def test_every_expert_hosted_and_served_by_a_host(self):
        uc = U.USE_CASES["uc1_multi_region"]
        Y, _ = uc.demand(routing())
        plan = M.plan_sites(Y[200], uc.topology, **uc.plan_kw())
        self.assertTrue(plan.hosts.any(axis=1).all())
        for s in range(uc.topology.n_sites):
            self.assertTrue(plan.hosts[np.arange(plan.n_experts), plan.serve[s]].all())

    def test_capacity_respected(self):
        uc = U.USE_CASES["uc1_multi_region"]
        Y, _ = uc.demand(routing())
        plan = M.plan_sites(Y[230], uc.topology, **uc.plan_kw())
        self.assertTrue((plan.hosts.sum(axis=0) <= uc.topology.capacity(uc.constraints)).all())

    def test_memory_constraint_limits_slots(self):
        topo = M.make_topology([M.Site("a")], devices_per_site=4, slots=1000, memory_gb=3.3)
        cap = topo.capacity(M.Constraints())
        self.assertEqual(int(cap[0]), int(3.3e9 // C.CHECKPOINT_EXPERT_BYTES))
        Y = np.ones((1, 1024))
        plan = M.plan_sites(Y, topo)
        self.assertTrue((plan.hosts.sum(axis=0) <= cap).all())

    def test_residency_zero_violations_and_full_local_copy(self):
        uc = U.USE_CASES["uc2_sovereign_residency"]
        Y, _ = uc.demand(routing())
        plan = M.plan_sites(Y[210], uc.topology, **uc.plan_kw())
        r = uc.topology.residency_sites()[0]
        local = plan.hosts[:, uc.topology.site_of == r].any(axis=1)
        self.assertTrue(local.all())
        for t in (205, 215, 300):  # also under demand the plan was not built for
            self.assertEqual(M.evaluate(plan, Y[t], uc.topology)["residency_violation_assignments"], 0.0)

    def test_residency_infeasible_when_site_too_small(self):
        sites = (M.Site("a", residency=True), M.Site("b"))
        topo = M.make_topology(list(sites), devices_per_site=2, slots=100)
        with self.assertRaises(M.InfeasiblePlan):
            M.plan_sites(np.ones((2, 1024)), topo)

    def test_balance_mode_replicas_never_increase_J(self):
        uc = U.USE_CASES["uc1_multi_region"]
        Y, _ = uc.demand(routing())
        for t in (200, 240):
            a = M.plan_sites(Y[t], uc.topology, replica_frac=0.0, constraints=uc.constraints)
            b = M.plan_sites(Y[t], uc.topology, replica_frac=0.5, constraints=uc.constraints)
            self.assertLessEqual(M.J_of(b, Y[t], uc.topology), M.J_of(a, Y[t], uc.topology) + 1e-12)

    def test_locality_mode_cuts_latency_within_cap(self):
        uc = U.USE_CASES["uc1_multi_region"]
        Y, _ = uc.demand(routing())
        y = Y[200]
        a = M.evaluate(M.plan_sites(y, uc.topology, replica_frac=0.0, constraints=uc.constraints), y, uc.topology, 0.7)
        b = M.evaluate(M.plan_sites(y, uc.topology, **uc.plan_kw()), y, uc.topology, 0.7)
        self.assertLess(b["added_latency_ms_per_token"], a["added_latency_ms_per_token"])
        self.assertLessEqual(b["J"], max(0.7, a["J"]) + 1e-9)

    def test_greedy_matches_exhaustive_on_tiny_instance(self):
        topo = M.make_topology([M.Site("a"), M.Site("b"), M.Site("c")], devices_per_site=1, speeds=(1.0, 1.3, 0.8), slots=3)
        rng = np.random.default_rng(3)
        Y = rng.uniform(1e6, 5e7, size=(3, 6))
        cons = M.Constraints(capacity=True, memory=False, residency=False)
        g = M.J_of(M.plan_sites(Y, topo, constraints=cons), Y, topo)
        o = M.J_of(M.exact_small(Y, topo, cons), Y, topo)
        self.assertLessEqual(o, g + 1e-12)


class TestCostModel(unittest.TestCase):
    def test_hand_computed_example(self):
        topo = tiny_topo()
        # expert 0 on device a0, expert 1 on b0; site a sends 10 to e0 and 20 to e1; site b sends 30 to e1
        plan = M.plan_from_primary(np.array([0, 1]), topo)
        Y = np.array([[10.0, 20.0], [0.0, 30.0]])
        ev = M.evaluate(plan, Y, topo)
        self.assertAlmostEqual(ev["rho_compute_max"], 50 / 100)          # b0 serves 20 + 30 per 100 capacity
        wan = 20 * 1000 / (1e-6 * 1e9 * 1.0)                                # 20 remote assignments * 1000 B
        self.assertAlmostEqual(ev["rho_wan_max"], wan)
        self.assertAlmostEqual(ev["remote_share"], 20 / 60)
        r_a = 20 / 30
        lat_a = 16 * 10.0 * (1 - (1 - r_a) ** 8)
        self.assertAlmostEqual(ev["added_latency_ms_per_token"], 30 * lat_a / 60)

    def test_objective_penalises_cap(self):
        self.assertEqual(M.objective_value(0.5, 100.0, None), 0.5)
        self.assertEqual(M.objective_value(0.6, 100.0, 0.7), 100.0)
        self.assertAlmostEqual(M.objective_value(0.8, 100.0, 0.7), 100.0 + M.PENALTY * 0.1)


class TestMigration(unittest.TestCase):
    def setUp(self):
        self.uc = U.USE_CASES["uc1_multi_region"]
        self.Y, _ = self.uc.demand(routing())

    def test_identical_plans_cost_nothing(self):
        p = M.plan_sites(self.Y[200], self.uc.topology, **self.uc.plan_kw())
        m = M.migration_cost(p, p, self.uc.topology)
        self.assertEqual((m["new_copies"], m["bytes"], m["est_time_s"]), (0, 0.0, 0.0))

    def test_lost_device_copies_reload_from_storage_when_unique(self):
        topo = self.uc.topology
        p = M.plan_sites(self.Y[200], topo, replica_frac=0.0, constraints=self.uc.constraints)
        lost = topo.devices[0].name
        t2 = topo.without([lost])
        q = M.plan_sites(self.Y[200], t2, replica_frac=0.0, constraints=self.uc.constraints)
        m = M.migration_cost(p, q, topo, t2)
        self.assertGreater(m["storage_bytes"], 0)
        self.assertNotIn(lost, [d.name for d in t2.devices])

    def test_stickiness_reduces_churn(self):
        topo, kw = self.uc.topology, self.uc.plan_kw()
        p = M.plan_sites(self.Y[200], topo, **kw)
        loose = M.plan_sites(self.Y[204], topo, **kw)
        sticky = M.plan_sites(self.Y[204], topo, incumbent=M.incumbent_matrix(p, topo, topo), stickiness=0.05, **kw)
        self.assertLess(M.migration_cost(p, sticky, topo)["new_copies"], M.migration_cost(p, loose, topo)["new_copies"])


if __name__ == "__main__":
    unittest.main()
