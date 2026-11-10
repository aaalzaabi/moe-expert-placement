"""The 24-prompt routing measurement and the held-out placement evaluation run on it."""
import importlib.util
import unittest

import pandas as pd

from tests.helpers import ROOT
from moeplace import config as C
from moeplace import data as D

MEAS = ROOT / "outputs" / "measurements" / "routing-20261110T123026Z"
SCRIPT = ROOT / "scripts" / "evaluate_extended_prompts.py"


def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_extended_prompts", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(MEAS.exists(), "24-prompt measurement not present")
class TestExtendedPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routing = D.load_routing(MEAS)  # schema + top-8 invariant
        cls.ev = load_script()

    def test_reprofiled_original_prompts_match_published_routing_exactly(self):
        new = self.routing[self.routing.request_id < 6]
        old = D.load_routing()
        key = ["request_id", "layer_id", "expert_id"]
        m = old.merge(new, on=key, suffixes=("_old", "_new"))
        self.assertEqual(len(m), 6 * C.NUM_LAYERS * C.EXPERTS_PER_LAYER)
        self.assertTrue((m.token_assignments_old == m.token_assignments_new).all())

    def test_original_split_reproduces_published_held_out_gain(self):
        cfg = C.SimConfig()
        pk = self.ev.peaks(self.routing, cfg.train_ids, cfg.test_ids, cfg)
        self.assertAlmostEqual(100 * (pk["demand_aware"] - pk["static"]) / pk["static"], -29.428, places=2)

    def test_unseen_prompts_keep_the_gain(self):
        cfg = C.SimConfig()
        new_ids = list(range(6, 24))
        pk = self.ev.peaks(self.routing, self.ev.ORIGINAL_IDS, new_ids, cfg)
        self.assertLess(pk["demand_aware"], pk["balanced"])
        self.assertLess(pk["balanced"], pk["static"])
        self.assertAlmostEqual(100 * (pk["demand_aware"] - pk["static"]) / pk["static"], -29.806, places=2)

    def test_overlapping_train_and_test_prompts_raise(self):
        with self.assertRaises(ValueError):
            self.ev.peaks(self.routing, [0, 1, 2], [2, 3], C.SimConfig())

    def test_stratified_split_is_balanced_and_disjoint(self):
        import numpy as np
        cats = self.routing.groupby("request_id").category.first()
        tr, te = self.ev.stratified(cats, 4, np.random.default_rng(0))
        self.assertFalse(set(tr) & set(te))
        self.assertEqual(len(tr) + len(te), len(cats))
        self.assertEqual(pd.Series(cats[tr]).value_counts().tolist(), [4, 4, 4])


if __name__ == "__main__":
    unittest.main()
