"""The real expert-parallel script plans without torch, and every expert gets exactly one device."""
import importlib.util
import unittest

from tests.helpers import ROOT

spec = importlib.util.spec_from_file_location("rep", ROOT / "scripts" / "run_real_expert_parallel.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)


class TestPlanForRealDevices(unittest.TestCase):
    def test_plan_covers_every_expert_once_per_strategy(self):
        for n, speeds in [(2, [1.0, 1.0]), (3, [1.0, 2.0, 1.5]), (8, [1.0] * 8)]:
            with self.subTest(n=n):
                cfg, pl = rep.make_plan(n, speeds, 0)
                for s in rep.C.STRATEGIES:
                    rows = pl[pl.strategy == s]
                    self.assertEqual(len(rows), 1024)
                    self.assertFalse(rows.duplicated(["layer_id", "expert_id"]).any())
                    self.assertTrue(rows.gpu_id.between(0, n - 1).all())
                    self.assertLessEqual(rows.groupby("gpu_id").size().max(), cfg.capacity_slots)

    def test_identical_gpus_make_demand_aware_equal_balanced(self):
        _, pl = rep.make_plan(4, [1.0] * 4, 0)
        a = pl[pl.strategy == "balanced"].gpu_id.to_numpy()
        b = pl[pl.strategy == "demand_aware"].gpu_id.to_numpy()
        self.assertTrue((a == b).all())


if __name__ == "__main__":
    unittest.main()
