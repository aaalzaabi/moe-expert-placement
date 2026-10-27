"""Integration: the refactored core regenerates every saved simulated result bit-for-bit.

This both protects the original findings from accidental change and documents
exactly which parameters produced each saved file.
"""
import unittest

import pandas as pd

from tests.helpers import C, D, POL, assert_frames_close, default_layouts, routing
from moeplace import placement as P
from moeplace import simulator as S

SUMMARY_COLS = ["policy", "migration_state", "migration_activation_ms", "moved_experts", "migration_mb",
                "total_link_mb", "remote_expert_calls", "p50_latency_ms", "p95_latency_ms", "makespan_ms"]
REQ_COLS = ["request_id", "arrival_ms", "finish_ms", "latency_ms", "layout", "policy"]


class TestPlacementReproduction(unittest.TestCase):
    def test_heterogeneous_placements_phase3b(self):
        pl, _ = POL.build_layouts(routing(), C.DEFAULT)
        saved = D.read_csv("heterogeneous_placements.csv")
        assert_frames_close(self, pl, saved, list(saved.columns), "heterogeneous_placements")

    def test_heldout_comparison_phase3b(self):
        pl, _ = POL.build_layouts(routing(), C.DEFAULT)
        got = P.summary(pl, "test_demand", C.DEFAULT)
        saved = D.read_csv("heldout_comparison.csv")
        assert_frames_close(self, got.rename(columns={"peak_load": "peak_test_load", "mean_load": "mean_test_load",
                                                      "imbalance": "test_imbalance"}),
                            saved, list(saved.columns), "heldout_comparison")

    def test_gpu_placements_phase3a(self):
        cfg = C.DEFAULT.with_(gpu_speeds=C.PHASE3A_SPEEDS)
        got = P.plan(D.demand(routing()), cfg).rename(columns={"train_demand": "demand"})
        saved = D.read_csv("gpu_placements.csv")
        assert_frames_close(self, got, saved, list(saved.columns), "gpu_placements")
        cmp_ = P.summary(got, "demand", cfg).rename(columns={"peak_load": "max_normalized_load",
                                                            "mean_load": "mean_normalized_load", "imbalance": "load_imbalance"})
        saved = D.read_csv("placement_comparison.csv")
        assert_frames_close(self, cmp_, saved, list(saved.columns), "placement_comparison")


class TestSimulationReproduction(unittest.TestCase):
    def test_phase4a_batch(self):
        lay = default_layouts()
        rows = []
        for s in C.STRATEGIES:
            req, m = S.run_batch_at_zero(routing(), lay[s], C.PHASE4A)
            rows.append({"strategy": s, **m})
        saved = D.read_csv("simulated_inference_comparison.csv")
        assert_frames_close(self, pd.DataFrame(rows), saved, list(saved.columns), "phase4a")

    def test_phase4b_migration_events(self):
        lay = default_layouts()
        ev, m = S.migration_only(lay["static"], lay["demand_aware"], C.PHASE4A)
        saved = D.read_csv("migration_events.csv")
        assert_frames_close(self, ev, saved, list(saved.columns), "migration_events")
        self.assertEqual(m["moved_experts"], 776)

    def test_phase4c_migration_benefit(self):
        lay = default_layouts()
        s = S.run_batch_at_zero(routing(), lay["static"], C.PHASE4A)[1]["makespan_ms"]
        o = S.run_batch_at_zero(routing(), lay["demand_aware"], C.PHASE4A)[1]["makespan_ms"]
        mig = S.migration_only(lay["static"], lay["demand_aware"], C.PHASE4A)[1]["activation_ms"]
        saved = D.read_csv("migration_benefit.csv")
        for r in saved.itertuples():
            self.assertAlmostEqual(r.static_total_ms, r.batches * s, places=6)
            self.assertAlmostEqual(r.optimized_total_ms, mig + r.batches * o, places=6)

    def test_phase4f_sweeps(self):
        lay = default_layouts()
        test = D.build_traces(routing(), C.DEFAULT.test_ids)
        for kind, preset in [("network", "phase4f_network"), ("calibrated", "phase4f_calibrated")]:
            for bw in C.PHASE4F_BANDWIDTHS:
                with self.subTest(kind=kind, bw=bw):
                    cfg = C.PRESETS[preset].with_(bandwidth_gb_per_s=float(bw))
                    got = pd.DataFrame([S.run(test, lay, cfg, allow_migration=m).summary for m in (False, True)])
                    saved = D.read_csv(f"phase4f/{kind}/bandwidth_{bw}.csv")
                    assert_frames_close(self, got, saved, SUMMARY_COLS, f"{kind}/{bw}")

    def test_phase4f_requests_and_migration_at_3gbps(self):
        lay = default_layouts()
        test = D.build_traces(routing(), C.DEFAULT.test_ids)
        cfg = C.PRESETS["phase4f_network"].with_(bandwidth_gb_per_s=3.0)
        req = pd.concat([S.run(test, lay, cfg, allow_migration=m).requests.assign(policy="migrate" if m else "static")
                         for m in (False, True)])
        assert_frames_close(self, req, D.read_csv("phase4f/network/requests_3.csv"), REQ_COLS, "requests_3")
        saved = D.read_csv("phase4f/network/migration_3.csv")
        assert_frames_close(self, S.run(test, lay, cfg).migrations, saved, list(saved.columns), "migration_3")

    def test_integrated_outputs_are_calibrated_100gbps(self):
        lay = default_layouts()
        test = D.build_traces(routing(), C.DEFAULT.test_ids)
        cfg = C.PRESETS["phase4f_calibrated"].with_(bandwidth_gb_per_s=100.0)
        runs = {m: S.run(test, lay, cfg, allow_migration=m) for m in (False, True)}
        assert_frames_close(self, pd.DataFrame([runs[m].summary for m in (False, True)]),
                            D.read_csv("integrated_comparison.csv"), SUMMARY_COLS, "integrated_comparison")
        req = pd.concat([runs[m].requests.assign(policy="migrate" if m else "static") for m in (False, True)])
        assert_frames_close(self, req, D.read_csv("integrated_request_latency.csv"), REQ_COLS, "integrated_requests")
        saved = D.read_csv("integrated_migration_events.csv")
        assert_frames_close(self, runs[True].migrations, saved, list(saved.columns), "integrated_migration_events")

    def test_phase4e_policy_files(self):
        cfg = C.PRESETS["phase4e_legacy"]
        pol, req, dec, _, _ = POL.evaluate(routing(), cfg, default_layouts())
        saved = D.read_csv("policy_comparison.csv")
        assert_frames_close(self, pol, saved, list(saved.columns), "policy_comparison")
        assert_frames_close(self, req, D.read_csv("policy_request_latency.csv"), REQ_COLS, "policy_request_latency")
        saved = D.read_csv("policy_decision.csv")
        assert_frames_close(self, POL.decision_frame(dec, cfg), saved, list(saved.columns), "policy_decision")


if __name__ == "__main__":
    unittest.main()
