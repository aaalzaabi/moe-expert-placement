"""Placement feasibility and algorithm properties."""
import unittest

import numpy as np

from tests.helpers import C, D, routing
from moeplace import placement as P


class TestFeasibility(unittest.TestCase):
    def test_every_expert_placed_once_within_capacity(self):
        for speeds in [(1.0, 1.3, 0.8, 1.6), (1, 1), (1, 2, 3), (0.5, 1, 1, 1, 2, 2, 2, 4)]:
            cfg = C.DEFAULT.with_(gpu_speeds=speeds, placement_capacity_slots=None, staging_capacity_slots=None)
            pl = P.plan(D.demand(routing(), cfg.train_ids), cfg)
            for s in C.STRATEGIES:
                with self.subTest(speeds=speeds, strategy=s):
                    rows = pl[pl.strategy == s]
                    self.assertEqual(len(rows), C.TOTAL_EXPERT_SLOTS)
                    self.assertFalse(rows.duplicated(["layer_id", "expert_id"]).any())
                    self.assertTrue(rows.gpu_id.between(0, cfg.num_gpus - 1).all())
                    self.assertLessEqual(rows.groupby("gpu_id").size().max(), cfg.capacity_slots)
                    self.assertTrue(P.gpu_metrics(pl, s, "train_demand", cfg).feasible.all())

    def test_insufficient_capacity_raises(self):
        with self.assertRaises(ValueError):
            P.assign(np.ones(10), "balanced", [1, 1], capacity=4)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            P.assign(np.ones(4), "random", [1, 1], capacity=4)

    def test_memory_budget_marks_infeasible(self):
        cfg = C.DEFAULT.with_(gpu_memory_gb=2.0)  # 256 experts x 12.58 MB = 3.2 GB > 2 GB
        pl = P.plan(D.demand(routing(), cfg.train_ids), cfg)
        m = P.gpu_metrics(pl, "static", "train_demand", cfg)
        self.assertFalse(m.feasible.any())
        self.assertAlmostEqual(m.weight_gb.iloc[0], 256 * C.CHECKPOINT_EXPERT_BYTES / C.GB)


class TestAlgorithms(unittest.TestCase):
    def test_static_is_round_robin(self):
        out = P.assign(np.arange(12), "static", [1, 1, 1], capacity=4)
        self.assertEqual(out.tolist(), [i % 3 for i in range(12)])

    def test_balanced_equals_demand_aware_on_identical_gpus(self):
        d = D.demand(routing()).to_numpy()
        self.assertTrue((P.assign(d, "balanced", [1] * 4, 256) == P.assign(d, "demand_aware", [1] * 4, 256)).all())

    def test_demand_aware_uses_speed(self):
        # Without capacity pressure, a 3x faster GPU should receive about 3x the demand.
        d = np.ones(400)
        out = P.assign(d, "demand_aware", [1.0, 3.0], capacity=400)
        share = (out == 1).mean()
        self.assertAlmostEqual(share, 0.75, delta=0.01)

    def test_demand_aware_reduces_train_peak(self):
        cfg = C.DEFAULT
        pl = P.plan(D.demand(routing(), cfg.train_ids), cfg)
        s = P.summary(pl, "train_demand", cfg).set_index("strategy")
        self.assertLess(s.loc["demand_aware", "peak_load"], s.loc["static", "peak_load"])

    def test_placement_ignores_test_demand(self):
        """Leakage guard: changing held-out routing must not change placements."""
        r = routing().copy()
        test_rows = r.request_id.isin(C.DEFAULT.test_ids)
        r.loc[test_rows, "token_assignments"] = r.loc[test_rows, "token_assignments"].to_numpy()[::-1]
        a = P.plan(D.demand(routing(), C.DEFAULT.train_ids), C.DEFAULT)
        b = P.plan(D.demand(r, C.DEFAULT.train_ids), C.DEFAULT)
        self.assertTrue((a.gpu_id.to_numpy() == b.gpu_id.to_numpy()).all())

    def test_moved_experts_count(self):
        pl = P.plan(D.demand(routing(), C.DEFAULT.train_ids), C.DEFAULT)
        self.assertEqual(len(P.moved_experts(pl, "static", "demand_aware")), 776)
        self.assertEqual(len(P.moved_experts(pl, "static", "static")), 0)



class TestBetterThanRandom(unittest.TestCase):
    """Evidence that the placement is a real method, not chance."""

    @classmethod
    def setUpClass(cls):
        cls.pl = P.plan(D.demand(routing(), C.DEFAULT.train_ids), C.DEFAULT,
                        eval_demand=D.demand(routing(), C.DEFAULT.test_ids))
        cls.v = P.validate_vs_random(cls.pl, C.DEFAULT, n=2000, seed=0)

    def test_demand_aware_beats_random_on_heldout(self):
        t = self.v["table"].set_index("strategy")
        self.assertGreaterEqual(t.loc["demand_aware", "beats_random_pct"], 99.0)
        self.assertLess(t.loc["demand_aware", "peak_load"], self.v["random_best"])

    def test_lower_bound_is_never_beaten(self):
        self.assertLessEqual(self.v["lower_bound"], self.v["random"].min())
        self.assertLessEqual(self.v["lower_bound"], self.v["table"].peak_load.min() + 1e-9)
        self.assertLessEqual(self.v["lower_bound"], self.v["hindsight"] + 1e-9)

    def test_random_baseline_respects_capacity_and_is_seeded(self):
        a = P.random_baseline(np.ones(1024), C.DEFAULT, n=50, seed=3)
        self.assertTrue(np.array_equal(a, P.random_baseline(np.ones(1024), C.DEFAULT, n=50, seed=3)))
        # equal demand and 256 slots per GPU: every random layout fills each GPU exactly
        self.assertTrue(np.allclose(a, 256 / min(C.DEFAULT.gpu_speeds)))


if __name__ == "__main__":
    unittest.main()
