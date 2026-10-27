"""Controller decisions, leakage guards and migration state transitions."""
import unittest

from tests.helpers import C, D, POL, default_layouts, routing


class TestStateMachine(unittest.TestCase):
    def test_happy_path(self):
        sm = POL.MigrationStateMachine()
        sm.to("approved").to("transferring").to("active")
        self.assertEqual(sm.state, "active")
        self.assertTrue(sm.terminal)
        self.assertEqual(len(sm.history), 3)

    def test_reject_and_repropose(self):
        sm = POL.MigrationStateMachine().to("rejected")
        self.assertTrue(sm.terminal)
        sm.to("proposed")
        self.assertEqual(sm.state, "proposed")

    def test_failure_paths(self):
        POL.MigrationStateMachine().to("approved").to("failed")            # staging memory
        POL.MigrationStateMachine().to("approved").to("transferring").to("failed")  # copy error

    def test_illegal_transitions(self):
        illegal = [("proposed", "active"), ("proposed", "transferring"), ("proposed", "failed"),
                   ("approved", "active"), ("approved", "rejected"), ("rejected", "active"),
                   ("active", "transferring"), ("failed", "active")]
        for start, target in illegal:
            with self.subTest(start=start, target=target), self.assertRaises(POL.InvalidTransition):
                POL.MigrationStateMachine(state=start).to(target)
        with self.assertRaises(POL.InvalidTransition):
            POL.MigrationStateMachine().to("teleported")

    def test_every_state_reachable(self):
        reach, frontier = {"proposed"}, ["proposed"]
        while frontier:
            for nxt in POL.TRANSITIONS[frontier.pop()]:
                if nxt not in reach:
                    reach.add(nxt)
                    frontier.append(nxt)
        self.assertEqual(reach, set(POL.STATES))


class TestController(unittest.TestCase):
    def test_saved_phase4e_decision_is_reject_for_p95(self):
        dec = POL.decide(D.build_traces(routing(), (0, 2, 4)), default_layouts(), C.PRESETS["phase4e_legacy"])
        self.assertFalse(dec.approve)
        self.assertTrue(dec.reasons[0][1], "migration should be feasible")
        self.assertTrue(dec.reasons[1][1], "makespan threshold should pass")
        self.assertFalse(dec.reasons[2][1], "p95 limit should fail")

    def test_thresholds_drive_decision(self):
        traces = D.build_traces(routing(), (0, 2, 4))
        lenient = C.PRESETS["phase4e_legacy"].with_(min_makespan_gain_pct=-100, max_p95_regression_pct=1000)
        strict = C.PRESETS["phase4e_legacy"].with_(min_makespan_gain_pct=50)
        self.assertTrue(POL.decide(traces, default_layouts(), lenient).approve)
        self.assertFalse(POL.decide(traces, default_layouts(), strict).approve)

    def test_infeasible_migration_is_rejected(self):
        cfg = C.DEFAULT.with_(staging_capacity_slots=256, min_makespan_gain_pct=-100, max_p95_regression_pct=1000)
        dec = POL.decide(D.build_traces(routing(), cfg.train_ids), default_layouts(), cfg)
        self.assertFalse(dec.migration_feasible)
        self.assertFalse(dec.approve)

    def test_decision_never_uses_heldout_requests(self):
        """Leakage guard: corrupt the held-out routing; the decision must not change."""
        cfg = C.PRESETS["phase4e_legacy"]
        r = routing().copy()
        test_rows = r.request_id.isin(cfg.test_ids)
        r.loc[test_rows, "token_assignments"] = r.loc[test_rows, "token_assignments"] * 7
        a = POL.decide(D.build_traces(routing(), cfg.train_ids), default_layouts(), cfg)
        b = POL.decide(D.build_traces(r, cfg.train_ids), default_layouts(), cfg)
        self.assertEqual((a.approve, a.forecast_makespan_gain_pct, a.forecast_p95_regression_pct),
                         (b.approve, b.forecast_makespan_gain_pct, b.forecast_p95_regression_pct))
        self.assertEqual(a.forecast_ids, cfg.train_ids)

    def test_cost_aware_row_equals_chosen_policy(self):
        for cfg in (C.PRESETS["phase4e_legacy"],
                    C.DEFAULT.with_(min_makespan_gain_pct=-100, max_p95_regression_pct=1000)):
            pol, req, dec, _, _ = POL.evaluate(routing(), cfg, default_layouts())
            pol = pol.set_index("policy").drop(columns=["failure_reason"])
            chosen = "always_migrate" if dec.approve else "never_migrate"
            self.assertTrue(pol.loc["cost_aware"].equals(pol.loc[chosen]))
            self.assertEqual(set(req.policy), {"never_migrate", "always_migrate", "cost_aware"})



class TestExplanation(unittest.TestCase):
    def _explain(self, cfg):
        _, lay = POL.build_layouts(routing(), cfg)
        return POL.explain(D.build_traces(routing(), cfg.train_ids), lay, cfg), lay

    def test_saved_rejection_is_explained(self):
        ex, _ = self._explain(C.PRESETS["phase4e_legacy"])
        self.assertFalse(ex.approved)
        self.assertIn("slowest request", ex.headline)
        self.assertEqual([c["passed"] for c in ex.checks], [True, True, False])
        self.assertTrue(any("Question 0" in c and "already running" in c for c in ex.cause))
        self.assertTrue(ex.what_would_change)
        self.assertEqual(set(ex.per_request.request_id), {0, 2, 4})  # training only: no held-out leakage

    def test_suggested_change_really_approves(self):
        cfg = C.PRESETS["phase4e_legacy"]
        ex, lay = self._explain(cfg)
        tip = next(c for c in ex.what_would_change if "this alone would approve" in c)
        limit = float(tip.split("up to ")[1].split("%")[0])
        dec = POL.decide(D.build_traces(routing(), cfg.train_ids), lay, cfg.with_(max_p95_regression_pct=limit))
        self.assertTrue(dec.approve)

    def test_infeasible_move_only_blames_memory(self):
        ex, _ = self._explain(C.DEFAULT.with_(staging_capacity_slots=300))
        self.assertEqual([c["passed"] for c in ex.checks], [False, None, None])
        self.assertIn("lack room", ex.headline)
        self.assertTrue(any("room for at least" in c for c in ex.what_would_change))

    def test_approval_is_explained_without_changes(self):
        ex, _ = self._explain(C.DEFAULT.with_(max_p95_regression_pct=100.0))
        self.assertTrue(ex.approved)
        self.assertTrue(ex.headline.startswith("APPROVED"))
        self.assertEqual(ex.what_would_change, [])


if __name__ == "__main__":
    unittest.main()
