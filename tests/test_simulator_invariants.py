"""Scientific-validity invariants of the discrete-event simulator."""
import unittest

import numpy as np
import pandas as pd

from tests.helpers import C, D, default_layouts, routing
from moeplace import simulator as S

EPS = 1e-9


def run(cfg=C.DEFAULT, ids=None, migrate=True, **kw):
    ids = ids or cfg.test_ids
    return S.run(D.build_traces(routing(), ids), default_layouts(), cfg, ids, allow_migration=migrate, record=True, **kw)


def no_overlap(intervals: pd.DataFrame) -> bool:
    iv = intervals.sort_values(["start_ms", "end_ms"]).to_numpy()
    return bool((iv[1:, 0] >= iv[:-1, 1] - EPS).all()) if len(iv) > 1 else True


class TestLayerDependencies(unittest.TestCase):
    def test_layers_are_sequential_per_request(self):
        res = run(C.DEFAULT.with_(bandwidth_gb_per_s=3.0))
        for rid, lay in res.layers.groupby("request_id"):
            lay = lay.sort_values("layer_id")
            self.assertEqual(lay.layer_id.tolist(), list(range(16)))
            self.assertTrue((lay.start_ms.to_numpy()[1:] >= lay.end_ms.to_numpy()[:-1] - EPS).all())

    def test_no_expert_starts_before_its_layer_and_all_finish_within_it(self):
        res = run()
        m = res.compute.merge(res.layers, on=["request_id", "layer_id"], suffixes=("", "_layer"))
        self.assertTrue((m.ready_ms >= m.start_ms_layer - EPS).all())
        self.assertTrue((m.end_ms <= m.end_ms_layer + EPS).all())
        self.assertTrue((m.start_ms >= m.inbound_ms - EPS).all(), "compute before activations arrived")

    def test_request_latency_equals_last_layer_end(self):
        res = run()
        last = res.layers.groupby("request_id").end_ms.max()
        self.assertTrue(np.allclose(res.requests.set_index("request_id").finish_ms, last))


class TestResourceExclusivity(unittest.TestCase):
    def test_gpu_queues_never_overlap(self):
        res = run(C.DEFAULT.with_(arrival_gap_ms=0.0))
        for g, d in res.compute.groupby("gpu_id"):
            self.assertTrue(no_overlap(d[["start_ms", "end_ms"]]), f"GPU {g} runs two tasks at once")

    def test_directed_links_never_overlap(self):
        res = run(C.DEFAULT.with_(bandwidth_gb_per_s=3.0))
        for link, d in res.transfers.groupby(["src_gpu", "dst_gpu"]):
            self.assertTrue(no_overlap(d[["start_ms", "end_ms"]]), f"link {link} carries two transfers at once")

    def test_shared_ports_never_overlap(self):
        res = run(C.DEFAULT.with_(bandwidth_gb_per_s=3.0, link_model="shared_port"))
        tr = res.transfers
        for g in range(4):
            self.assertTrue(no_overlap(tr[tr.src_gpu == g][["start_ms", "end_ms"]]), f"egress {g}")
            self.assertTrue(no_overlap(tr[tr.dst_gpu == g][["start_ms", "end_ms"]]), f"ingress {g}")

    def test_shared_port_is_never_faster_than_directed(self):
        for bw in (3.0, 12.0, 100.0):
            a = run(C.DEFAULT.with_(bandwidth_gb_per_s=bw)).summary
            b = run(C.DEFAULT.with_(bandwidth_gb_per_s=bw, link_model="shared_port")).summary
            self.assertGreaterEqual(b["migration_activation_ms"] + EPS, a["migration_activation_ms"])


class TestMigrationSemantics(unittest.TestCase):
    def test_requests_keep_layout_active_at_arrival(self):
        for bw in (3.0, 12.0, 100.0):
            res = run(C.DEFAULT.with_(bandwidth_gb_per_s=bw))
            act = res.summary["migration_activation_ms"]
            for r in res.requests.itertuples():
                expected = "demand_aware" if act is not None and r.arrival_ms >= act else "static"
                self.assertEqual(r.layout, expected, f"bw={bw} request {r.request_id}")

    def test_compute_uses_pinned_layout(self):
        res = run(C.DEFAULT.with_(bandwidth_gb_per_s=3.0))
        lay = default_layouts()
        pinned = res.requests.set_index("request_id").layout
        for t in res.compute.itertuples():
            self.assertEqual(t.gpu_id, lay[pinned[t.request_id]][(t.layer_id, t.expert_id)])

    def test_activation_only_after_every_copy(self):
        res = run()
        self.assertEqual(res.summary["migration_state"], "active")
        self.assertEqual(len(res.migrations), 776)
        self.assertAlmostEqual(res.summary["migration_activation_ms"], res.migrations.finish_ms.max())
        self.assertTrue((res.migrations.finish_ms <= res.summary["migration_activation_ms"] + EPS).all())

    def test_state_sequence(self):
        states = [s["state"] for s in run().states]
        self.assertEqual(states, ["proposed", "approved", "transferring", "active"])
        self.assertEqual([s["state"] for s in run(migrate=False).states], ["disabled"])

    def test_staging_capacity_counts_resident_plus_incoming(self):
        lay = default_layouts()
        old, new = lay["static"], lay["demand_aware"]
        need = max(sum(v == g for v in old.values()) + sum(old[k] != g and new[k] == g for k in old) for g in range(4))
        ok = run(C.DEFAULT.with_(staging_capacity_slots=need))
        bad = run(C.DEFAULT.with_(staging_capacity_slots=need - 1))
        self.assertEqual(ok.summary["migration_state"], "active")
        self.assertEqual(bad.summary["migration_state"], "failed")
        self.assertIn("staging capacity", bad.summary["failure_reason"])
        self.assertTrue((bad.requests.layout == "static").all())
        self.assertEqual(len(bad.migrations), 0)

    def test_byte_based_memory_budget(self):
        per_gpu_peak = 512 * C.CHECKPOINT_EXPERT_BYTES / C.GB  # upper bound on resident+staged
        self.assertEqual(run(C.DEFAULT.with_(gpu_memory_gb=per_gpu_peak)).summary["migration_state"], "active")
        small = run(C.DEFAULT.with_(gpu_memory_gb=3.3)).summary  # 256 resident = 3.22 GB, no room to stage
        self.assertEqual(small["migration_state"], "failed")

    def test_injected_copy_failure_rolls_back(self):
        res = run(C.DEFAULT.with_(fail_after_copies=100, bandwidth_gb_per_s=3.0))
        self.assertEqual(res.summary["migration_state"], "failed")
        self.assertIsNone(res.summary["migration_activation_ms"])
        self.assertTrue((res.requests.layout == "static").all(), "no request may use a half-copied layout")
        self.assertEqual([s["state"] for s in res.states][-1], "failed")
        self.assertIn("rolled back", res.states[-1]["note"])
        # failure after the last copy has no effect
        ok = run(C.DEFAULT.with_(fail_after_copies=776))
        self.assertEqual(ok.summary["migration_state"], "active")

    def test_inference_and_migration_share_links(self):
        """Copies delay in-flight inference on shared links (Phase 4F mechanism)."""
        quiet = run(C.DEFAULT.with_(bandwidth_gb_per_s=3.0), migrate=False)
        busy = run(C.DEFAULT.with_(bandwidth_gb_per_s=3.0), migrate=True)
        r1_quiet = quiet.requests.set_index("request_id").latency_ms[1]
        r1_busy = busy.requests.set_index("request_id").latency_ms[1]
        self.assertGreater(r1_busy, r1_quiet * 1.5)  # request 1 is in flight during copying, still on static layout
        mig_links = busy.transfers[busy.transfers.kind == "migration"]
        inf_links = busy.transfers[busy.transfers.kind != "migration"]
        shared = set(zip(mig_links.src_gpu, mig_links.dst_gpu)) & set(zip(inf_links.src_gpu, inf_links.dst_gpu))
        self.assertTrue(shared)


class TestAccountingAndDeterminism(unittest.TestCase):
    def test_traffic_conservation(self):
        res = run()
        tr = res.transfers
        self.assertAlmostEqual(tr.bytes.sum() / 1e6, res.summary["total_link_mb"])
        self.assertAlmostEqual(tr[tr.kind == "migration"].bytes.sum() / 1e6, res.summary["migration_mb"])
        self.assertEqual(int(res.compute.remote.sum()), res.summary["remote_expert_calls"])
        # every remote call has one dispatch and one return of equal size
        self.assertEqual((tr.kind == "dispatch").sum(), res.summary["remote_expert_calls"])
        self.assertEqual((tr.kind == "return").sum(), res.summary["remote_expert_calls"])

    def test_all_routed_work_is_computed(self):
        res = run(migrate=False)
        expected = routing()[routing().request_id.isin(C.DEFAULT.test_ids)].token_assignments.sum()
        self.assertEqual(int(res.compute.assignments.sum()), int(expected))

    def test_deterministic(self):
        a, b = run().summary, run().summary
        self.assertEqual(a, b)

    def test_poisson_arrivals_seeded(self):
        p = lambda seed: run(C.DEFAULT.with_(arrival_process="poisson", seed=seed)).requests.arrival_ms.tolist()
        self.assertEqual(p(1), p(1))
        self.assertNotEqual(p(1), p(2))
        self.assertEqual(p(1)[0], 0.0)

    def test_higher_bandwidth_never_slows_migration(self):
        acts = [run(C.DEFAULT.with_(bandwidth_gb_per_s=bw)).summary["migration_activation_ms"] for bw in (3, 6, 12, 25, 100)]
        self.assertTrue(all(x >= y - EPS for x, y in zip(acts, acts[1:])))

    def test_gpu_count_mismatch_is_reported(self):
        cfg = C.DEFAULT.with_(gpu_speeds=(1.0, 1.0), placement_capacity_slots=None, staging_capacity_slots=None)
        with self.assertRaises(ValueError):
            S.run(D.build_traces(routing(), (1,)), default_layouts(), cfg, (1,))


if __name__ == "__main__":
    unittest.main()
