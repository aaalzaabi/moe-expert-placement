"""Experiment runner, CLI and legacy-script safety (all writes go to a temp directory)."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests.helpers import ROOT, C
from moeplace import experiment as E
from moeplace import legacy


def quiet(*_a, **_k):
    pass


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_writes_csvs_and_metadata(self):
        d = E.run_scenario(C.PRESETS["phase4e_legacy"], "t", out_root=self.out, log=quiet)
        for f in ["placements.csv", "placement_heldout_summary.csv", "policy_comparison.csv",
                  "policy_request_latency.csv", "policy_decision.csv", "migration_events.csv",
                  "migration_states.csv", "metadata.json"]:
            self.assertTrue((d / f).exists(), f)
        meta = json.loads((d / "metadata.json").read_text())
        self.assertEqual(meta["result_kind"], "simulated")
        self.assertEqual(meta["config"]["expert_bytes"], C.LEGACY_EXPERT_BYTES)
        self.assertEqual(len(meta["routing_sha256"]), 64)
        self.assertEqual(meta["decision"], "reject")
        saved = pd.read_csv(ROOT / "outputs" / "policy_comparison.csv")
        got = pd.read_csv(d / "policy_comparison.csv")
        pd.testing.assert_frame_equal(got[saved.columns], saved)

    def test_run_id_is_deterministic_and_parameter_sensitive(self):
        a = E.run_id("x", C.DEFAULT, "abc")
        self.assertEqual(a, E.run_id("x", C.DEFAULT, "abc"))
        self.assertNotEqual(a, E.run_id("x", C.DEFAULT.with_(bandwidth_gb_per_s=13), "abc"))
        self.assertNotEqual(a, E.run_id("x", C.DEFAULT, "abd"))

    def test_rerun_skips_unless_overwrite(self):
        logs = []
        E.run_scenario(C.DEFAULT, "t", out_root=self.out, log=logs.append)
        E.run_scenario(C.DEFAULT, "t", out_root=self.out, log=logs.append)
        self.assertTrue(logs[-1].startswith("[skip]"))
        E.run_scenario(C.DEFAULT, "t", out_root=self.out, overwrite=True, log=logs.append)
        self.assertTrue(logs[-1].startswith("[done]"))
        self.assertFalse(any(p.name.startswith(".tmp-") for p in self.out.iterdir()))

    def test_sweep_summary(self):
        d = E.sweep(C.PRESETS["phase4f_network"], {"bandwidth_gb_per_s": [3.0, 12.0, 25.0, 100.0]},
                    name="s", out_root=self.out, log=quiet)
        s = pd.read_csv(d / "summary.csv")
        self.assertEqual(len(s), 4)
        self.assertTrue((s.status == "ok").all())
        saved = pd.read_csv(ROOT / "outputs/phase4f/network/network_improvement.csv")
        self.assertTrue(((s.heldout_p95_improvement_pct - saved.p95_improvement_pct).abs() < 1e-9).all())
        self.assertTrue((d / "sweep.json").exists())

    def test_sweep_records_invalid_points_and_continues(self):
        d = E.sweep(C.DEFAULT, {"bandwidth_gb_per_s": [12.0, -1.0]}, name="bad", out_root=self.out, log=quiet)
        s = pd.read_csv(d / "summary.csv")
        self.assertEqual(s.status.tolist()[0], "ok")
        self.assertTrue(s.status.tolist()[1].startswith("error"))


class TestCLI(unittest.TestCase):
    def _cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "scripts/run_experiment.py"), *args],
                              capture_output=True, text=True, cwd=tempfile.gettempdir())

    def test_invalid_config_message(self):
        r = self._cli("--gpu-speeds", "1", "1", "--capacity-slots", "10")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("placement infeasible", r.stderr)

    def test_parse_arguments(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment
        a = run_experiment.parse(["--bandwidth", "3", "12", "--expert-mb", "8", "--policy", "cost_aware"])
        self.assertEqual(a.bandwidth, [3.0, 12.0])
        self.assertEqual(a.policy, "cost_aware")

    def test_profile_routing_dry_run_loads_nothing(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/profile_routing.py"), "--dry-run"],
                           capture_output=True, text=True, cwd=tempfile.gettempdir())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Nothing loaded", r.stdout)


class TestLegacyGuard(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)

    def test_refuses_to_overwrite_measurements(self):
        argv = sys.argv
        sys.argv = ["profile_workloads.py"]
        try:
            with self.assertRaises(SystemExit) as ctx:
                legacy.prepare(["outputs/workload_routing.csv"], measured=True)
            self.assertIn("REAL OLMoE measurements", str(ctx.exception.code))
        finally:
            sys.argv = argv

    def test_force_and_new_files_pass(self):
        legacy.prepare(["outputs/definitely_new_file.csv"])
        self.assertEqual(Path.cwd(), ROOT)
        argv = sys.argv
        sys.argv = ["x.py", "--force"]
        try:
            legacy.prepare(["outputs/workload_routing.csv"])
        finally:
            sys.argv = argv

    def test_legacy_scripts_refuse_without_force(self):
        for script in ("simulate_integrated.py", "evaluate_migration_policy.py", "profile_workloads.py"):
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], capture_output=True, text=True,
                               cwd=tempfile.gettempdir(), timeout=120)
            with self.subTest(script=script):
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("refusing to overwrite", r.stderr)


if __name__ == "__main__":
    unittest.main()
