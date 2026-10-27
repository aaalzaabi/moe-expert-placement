"""Demand tensor, forecasters, regret and drift analysis."""
import unittest

import numpy as np

from tests.helpers import routing
from moeplace import forecast as F
from moeplace import multisite as M
from moeplace import usecases as U
from moeplace import workload as WL


class TestWorkload(unittest.TestCase):
    def test_profiles_sum_to_128_assignments_per_token(self):
        cats, P = WL.category_profiles(routing())
        self.assertEqual(cats, ["ai", "coding", "general"])
        self.assertTrue(np.allclose(P.sum(axis=1), 128))
        self.assertTrue((P > 0).all())  # smoothing keeps unseen experts possible

    def test_expected_tensor_matches_token_volume(self):
        cats, P = WL.category_profiles(routing())
        tr = [WL.SiteTraffic(1000, {"ai": 1.0}, tokens_per_request=10)]
        _, EX = WL.synthesize(tr, cats, P, 96, seed=0)
        lam = 1000 * WL.diurnal(np.arange(96), 900, 0, 0.25, 14.0)
        self.assertTrue(np.allclose(EX[:, 0, :].sum(axis=1), lam * 10 * 128))

    def test_shift_changes_mix_only_inside_window(self):
        cats, P = WL.category_profiles(routing())
        tr = WL.SiteTraffic(1000, {"ai": 1.0}, shifts=(WL.Shift(10, 20, {"coding": 1.0}, 2.0),))
        mix, mult = WL.expected_mix(tr, cats, 30)
        self.assertEqual(mix[5].tolist(), [1, 0, 0])
        self.assertEqual(mix[15].tolist(), [0, 1, 0])
        self.assertEqual(mult[15], 2.0)


class TestForecasters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cats, P = WL.category_profiles(routing())
        tr = [WL.SiteTraffic(1000, {"ai": 0.7, "coding": 0.3}, utc_offset_h=4, volume_cv=1e-6, mix_concentration=1e9),
              WL.SiteTraffic(800, {"general": 0.8, "ai": 0.2}, utc_offset_h=8, volume_cv=1e-6, mix_concentration=1e9)]
        _, cls.EX = WL.synthesize(tr, cats, P, 3 * 96, seed=0)
        cls.spec = F.ForecastSpec()

    def test_causal(self):
        Y = self.EX.copy()
        f = F.Forecaster("site_ewma", self.spec).prepare(Y)
        a = f.predict(150, 4)
        Y2 = Y.copy()
        Y2[151:] *= 7
        b = F.Forecaster("site_ewma", self.spec).prepare(Y2).predict(150, 4)
        self.assertTrue(np.allclose(a, b))

    def test_seasonal_models_exact_on_periodic_noise_free_demand(self):
        for m in ("seasonal_naive", "site_ewma"):
            f = F.Forecaster(m, self.spec).prepare(self.EX)
            self.assertLess(F.wape(self.EX[201:205], f.predict(200, 4)), 1e-9, m)

    def test_site_aware_beats_global_popularity(self):
        site = F.Forecaster("site_ewma", self.spec).prepare(self.EX).predict(200, 4)
        glob = F.Forecaster("global_ewma", self.spec).prepare(self.EX).predict(200, 4)
        actual = self.EX[201:205]
        self.assertLess(F.popularity_error(actual, site), F.popularity_error(actual, glob))

    def test_fit_ignores_data_after_train_end(self):
        Y = self.EX.copy()
        a = F.fit("ewma", Y, self.spec, 192).params
        Y[192:] = Y[192:] * np.random.default_rng(0).uniform(0, 5, Y[192:].shape)
        b = F.fit("ewma", Y, self.spec, 192).params
        self.assertEqual(a, b)

    def test_metrics(self):
        Y = np.array([[10.0, 0.0], [5.0, 5.0]])
        self.assertAlmostEqual(F.wape(Y, Y * 1.1), 0.1)
        self.assertAlmostEqual(F.popularity_error(Y, Y * 3), 0.0)
        self.assertAlmostEqual(F.volume_error(Y, Y * 1.5), 0.5)


class TestRegretAndDrift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.uc = U.USE_CASES["uc1_multi_region"]
        cls.Y, _ = cls.uc.demand(routing())

    def test_perfect_forecast_has_zero_regret(self):
        class Oracle(F.Forecaster):
            def predict(self, t, H):
                return self.Y[t + 1:t + H + 1]
        f = Oracle("naive_last", U.spec()).prepare(self.Y)
        row, _, _ = F.evaluate_point(200, self.Y, f, self.uc.topology, 4, **self.uc.plan_kw())
        self.assertAlmostEqual(row["regret_abs"], 0.0, places=9)
        self.assertEqual(row["hosts_jaccard"], 1.0)

    def test_rolling_evaluation_and_drift_table(self):
        rows = F.rolling_evaluation(self.Y, self.uc.topology, U.spec(), self.uc.train_end, models=("site_ewma", "naive_last"),
                                    eval_end=self.uc.train_end + 20, **self.uc.plan_kw())
        self.assertEqual(set(rows.model), {"site_ewma", "naive_last"})
        dt = F.drift_table(rows)
        n = rows.groupby("model").size()
        for _, r in dt.iterrows():
            total = r.harmful_and_high_error + r.harmless_high_error + r.harmful_low_error + r.harmless_low_error
            self.assertEqual(total, n[r.model])
        s = F.summarize(rows)
        self.assertIn("aggregate_regret_rel", s.columns)


if __name__ == "__main__":
    unittest.main()
