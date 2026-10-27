"""The visual GPU simulator replays the real placement and its numbers agree with the rest of the dashboard."""
import logging
import sys
import unittest

import numpy as np

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "dashboard"))
logging.getLogger("streamlit").setLevel(logging.ERROR)

import gpu_viz as V  # noqa: E402
import ui  # noqa: E402
from moeplace import placement as P  # noqa: E402
from ui import C  # noqa: E402


class TestGpuViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = C.DEFAULT
        cls.pl = ui.plan(ui.cfg_key(cls.cfg))
        cls.tr = {s: V.placement_trace(cls.pl, s, cls.cfg) for s in C.STRATEGIES}

    def test_trace_is_the_real_placement(self):
        for s, t in self.tr.items():
            with self.subTest(strategy=s):
                rows = self.pl[self.pl.strategy == s].set_index(["layer_id", "expert_id"]).gpu_id
                self.assertEqual(len(t), C.TOTAL_EXPERT_SLOTS)
                self.assertTrue((t.set_index(["layer_id", "expert_id"]).gpu_id.reindex(rows.index) == rows).all())
                self.assertLess(t.slot.max(), self.cfg.capacity_slots)
                self.assertFalse(t.duplicated(["gpu_id", "slot"]).any())

    def test_replaying_the_rule_in_trace_order_gives_the_same_choices(self):
        t, speeds = self.tr["demand_aware"], np.asarray(self.cfg.gpu_speeds)
        loads, counts = np.zeros(self.cfg.num_gpus), np.zeros(self.cfg.num_gpus, int)
        for d, g in zip(t.train_demand, t.gpu_id):
            free = [k for k in range(self.cfg.num_gpus) if counts[k] < self.cfg.capacity_slots]
            self.assertEqual(min(free, key=lambda k: ((loads[k] + d) / speeds[k], k)), g)
            loads[g] += d
            counts[g] += 1

    def test_busy_time_matches_placement_page(self):
        for s, t in self.tr.items():
            m = P.gpu_metrics(self.pl, s, "train_demand", self.cfg)
            self.assertTrue(np.allclose(V.busy_time(t, self.cfg), m.normalized_load))

    def test_decision_table_marks_the_minimum(self):
        row, dt = V.decision_table(self.tr["demand_aware"], 1, self.cfg)
        self.assertEqual(dt["busy time if placed here"].idxmin(), row.gpu_id)
        self.assertEqual(int(dt.chosen.sum()), 1)

    def test_question_layers_and_summary(self):
        active, per = V.question_work(ui.routing(), 1, self.pl, "demand_aware", self.cfg)
        r = ui.routing()
        self.assertEqual(active.token_assignments.sum(), r[r.request_id == 1].token_assignments.sum())
        self.assertEqual(per.shape, (C.NUM_LAYERS, self.cfg.num_gpus))
        s = V.question_summary(per)
        self.assertAlmostEqual(s["total"], per.max(axis=1).sum())
        self.assertTrue(0 <= s["waiting_pct"] < 100)

    def test_figures_have_frames(self):
        f = V.fig_placement_animation(self.tr["static"], self.cfg, "x")
        self.assertGreater(len(f.frames), 10)
        active, per = V.question_work(ui.routing(), 3, self.pl, "static", self.cfg)
        self.assertEqual(len(V.fig_question_animation(self.tr["static"], active, per, self.cfg, "q").frames), C.NUM_LAYERS)


if __name__ == "__main__":
    unittest.main()
